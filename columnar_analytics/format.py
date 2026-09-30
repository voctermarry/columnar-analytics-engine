"""Self-describing columnar file format.

Public API:

* :class:`ColumnSchema` / :class:`Schema` / :class:`Table` -- in-memory model
* :func:`write_file` -- deterministic, atomic write
* :func:`read_file` -- strict, validating read with optional projection
* :func:`inspect_file` -- metadata-only read
* :class:`ColumnarFormatError` -- raised for every malformed-file condition

File layout (little-endian)::

    b"CAEF"            magic
    uint8              format version
    uint32             header byte length
    <header bytes>     UTF-8 JSON, schema + per-column stats/chunk pointers
    <data section>     raw concatenated column payloads, or zlib of the same
    uint32             CRC-32 of every preceding byte
    b"END1"            end marker

The same schema, values and options always produce identical bytes: the JSON
header uses a fixed key order and compact separators, there are no timestamps,
and zlib is invoked with a fixed level.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "FORMAT_VERSION",
    "ColumnarFormatError",
    "ColumnSchema",
    "Schema",
    "Table",
    "write_file",
    "read_file",
    "inspect_file",
]

FORMAT_VERSION = 1

_MAGIC = b"CAEF"
_FOOTER_MARKER = b"END1"
_TYPE_NAMES = ("bool", "int64", "float64", "utf8")
_ENCODING_NAMES = ("plain", "dictionary")
_COMPRESSION_NAMES = ("none", "zlib")
_HEADER_KEYS = frozenset(
    ("compression", "row_count", "data_length", "data_crc32", "uncompressed_length", "columns")
)
_COLUMN_KEYS = frozenset(
    ("name", "type", "nullable", "encoding", "offset", "length", "crc32",
     "null_count", "min", "max")
)
_PREFIX_LEN = 9  # magic(4) + version(1) + header length uint32
_FOOTER_LEN = 8  # crc32 uint32 + marker
_MAX_HEADER_BYTES = 256 * 1024 * 1024


class ColumnarFormatError(Exception):
    """Raised when a columnar file is structurally invalid or unsupported."""


# ---------------------------------------------------------------------------
# In-memory model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnSchema:
    """One column: a non-empty unique Unicode ``name``, a type and nullability."""

    name: str
    type: str
    nullable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("column name must be a non-empty string")
        try:
            self.name.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("column name must contain valid Unicode scalars") from None
        if self.type not in _TYPE_NAMES:
            raise ValueError(f"unsupported column type: {self.type!r}")
        if not isinstance(self.nullable, bool):
            raise ValueError("nullable must be a bool")


@dataclass(frozen=True)
class Schema:
    """An ordered, non-empty collection of :class:`ColumnSchema`."""

    columns: tuple[ColumnSchema, ...]

    def __init__(self, columns: Sequence[ColumnSchema]):
        cols = tuple(columns)
        if not cols:
            raise ValueError("schema must contain at least one column")
        for col in cols:
            if not isinstance(col, ColumnSchema):
                raise ValueError("schema entries must be ColumnSchema instances")
        seen: set[str] = set()
        for col in cols:
            if col.name in seen:
                raise ValueError(f"duplicate column name: {col.name!r}")
            seen.add(col.name)
        object.__setattr__(self, "columns", cols)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(col.name for col in self.columns)

    def index(self, name: str) -> int:
        for i, col in enumerate(self.columns):
            if col.name == name:
                return i
        raise KeyError(name)


class Table:
    """A schema together with equal-length, column-oriented Python lists."""

    def __init__(self, schema: Schema, columns: Mapping[str, Sequence] | Sequence[Sequence]):
        if not isinstance(schema, Schema):
            raise TypeError("schema must be a Schema instance")
        aligned = _align_columns(schema, columns)
        row_count = _validate_columns(schema, aligned)
        self.schema = schema
        self._row_count = row_count
        self._columns = tuple(tuple(col) for col in aligned)

    @classmethod
    def _from_storage(cls, schema: Schema, columns: Sequence[Sequence]) -> "Table":
        """Build without re-validating (used by the decoder)."""
        obj = cls.__new__(cls)
        obj.schema = schema
        obj._row_count = len(columns[0]) if columns else 0
        obj._columns = tuple(tuple(col) for col in columns)
        return obj

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def column_names(self) -> tuple[str, ...]:
        return self.schema.names

    def column(self, name: str) -> list:
        """Return a fresh list of the named column's values."""
        return list(self._columns[self.schema.index(name)])

    @property
    def columns(self) -> dict[str, list]:
        """Column name to value list, in schema order."""
        return {name: list(self._columns[i]) for i, name in enumerate(self.schema.names)}

    def project(self, names: Sequence[str]) -> "Table":
        """Return a table restricted to ``names``, in the requested order.

        Unknown names raise :class:`KeyError`; repeated names raise
        :class:`ValueError`.
        """
        if isinstance(names, str):
            raise ValueError("project names must be a sequence of strings, not a single string")
        requested = tuple(names)
        if len(set(requested)) != len(requested):
            raise ValueError("projection contains duplicate column names")
        indices = []
        for name in requested:
            indices.append(self.schema.index(name))  # KeyError on unknown name
        sub_schema = Schema([self.schema.columns[i] for i in indices])
        sub_cols = [self._columns[i] for i in indices]
        return Table._from_storage(sub_schema, sub_cols)


def _align_columns(
    schema: Schema, columns: Mapping[str, Sequence] | Sequence[Sequence]
) -> list[Sequence]:
    if isinstance(columns, Mapping):
        missing = [name for name in schema.names if name not in columns]
        if missing:
            raise ValueError(f"missing column(s): {', '.join(missing)!r}")
        extra = [name for name in columns if name not in set(schema.names)]
        if extra:
            raise ValueError(f"unexpected column(s): {', '.join(map(str, extra))!r}")
        return [columns[name] for name in schema.names]
    if isinstance(columns, Sequence) and not isinstance(columns, (str, bytes)):
        if len(columns) != len(schema.columns):
            raise ValueError(
                f"expected {len(schema.columns)} columns, got {len(columns)}"
            )
        return list(columns)
    raise TypeError("columns must be a mapping or a sequence aligned to the schema")


def _validate_columns(schema: Schema, columns: Sequence[Sequence]) -> int:
    """Check shape, nullability, per-value Python types and float finiteness."""
    row_count: int | None = None
    for col, raw in zip(schema.columns, columns):
        try:
            values = list(raw)
        except TypeError:
            raise ValueError(f"column {col.name!r} is not a sequence") from None
        if row_count is None:
            row_count = len(values)
        elif len(values) != row_count:
            raise ValueError(
                f"column {col.name!r} has {len(values)} rows, expected {row_count}"
            )
        for position, value in enumerate(values):
            where = f"column {col.name!r} row {position}"
            if value is None:
                if not col.nullable:
                    raise ValueError(f"{where}: NULL in non-nullable column")
                continue
            if col.type == "bool":
                if not isinstance(value, bool):
                    raise ValueError(f"{where}: expected bool, got {type(value).__name__}")
            elif col.type == "int64":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{where}: expected int64, got {type(value).__name__}")
                if not (-(2**63) <= value <= 2**63 - 1):
                    raise ValueError(f"{where}: value outside int64 range")
            elif col.type == "float64":
                if isinstance(value, bool) or not isinstance(value, float):
                    raise ValueError(f"{where}: expected float64, got {type(value).__name__}")
                if value != value or value in (float("inf"), float("-inf")):
                    raise ValueError(f"{where}: float64 must be finite (no NaN or infinity)")
            elif col.type == "utf8":
                if not isinstance(value, str):
                    raise ValueError(f"{where}: expected utf8 string, got {type(value).__name__}")
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError:
                    raise ValueError(f"{where}: string is not valid UTF-8 Unicode") from None
    assert row_count is not None
    return row_count


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _pack_bits(flags: Sequence[bool]) -> bytes:
    out = bytearray((len(flags) + 7) // 8)
    for i, flag in enumerate(flags):
        if flag:
            out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


def _check_padding(bits: bytes, count: int, what: str) -> None:
    if count % 8:
        unused = bits[-1] >> (count % 8)
        if unused:
            raise ColumnarFormatError(f"non-zero padding bits in {what}")


def _encode_payload(
    col: ColumnSchema, values: Sequence, use_dictionary: bool
) -> tuple[bytes, int, Any, Any]:
    n = len(values)
    null_count = 0
    minimum: Any = None
    maximum: Any = None
    validity = bytearray((n + 7) // 8) if col.nullable else None

    if col.type == "bool":
        flags: list[bool] = []
        for i, value in enumerate(values):
            if value is None:
                flags.append(False)
                null_count += 1
                continue
            if validity is not None:
                validity[i >> 3] |= 1 << (i & 7)
            flags.append(value)
            if minimum is None or value < minimum:
                minimum = value
            if maximum is None or value > maximum:
                maximum = value
        body = _pack_bits(flags)
    elif col.type == "int64":
        buf = bytearray()
        for i, value in enumerate(values):
            if value is None:
                buf.extend(b"\x00" * 8)
                null_count += 1
                continue
            if validity is not None:
                validity[i >> 3] |= 1 << (i & 7)
            buf.extend(struct.pack("<q", value))
            if minimum is None or value < minimum:
                minimum = value
            if maximum is None or value > maximum:
                maximum = value
        body = bytes(buf)
    elif col.type == "float64":
        buf = bytearray()
        for i, value in enumerate(values):
            if value is None:
                buf.extend(b"\x00" * 8)
                null_count += 1
                continue
            if validity is not None:
                validity[i >> 3] |= 1 << (i & 7)
            buf.extend(struct.pack("<d", value))
            if minimum is None or value < minimum:
                minimum = value
            if maximum is None or value > maximum:
                maximum = value
        body = bytes(buf)
    else:  # utf8
        if use_dictionary:
            dictionary: list[str] = []
            dictionary_index: dict[str, int] = {}
            indices = bytearray()
            for i, value in enumerate(values):
                if value is None:
                    indices.extend(struct.pack("<i", -1))
                    null_count += 1
                    continue
                if validity is not None:
                    validity[i >> 3] |= 1 << (i & 7)
                idx = dictionary_index.get(value)
                if idx is None:
                    # Dictionary ids follow first-occurrence order.
                    idx = len(dictionary)
                    dictionary_index[value] = idx
                    dictionary.append(value)
                indices.extend(struct.pack("<i", idx))
                if minimum is None or value < minimum:
                    minimum = value
                if maximum is None or value > maximum:
                    maximum = value
            body = (
                struct.pack("<I", len(dictionary))
                + _encode_utf8_strings(dictionary)
                + bytes(indices)
            )
        else:
            ordered: list[str] = []
            for i, value in enumerate(values):
                if value is None:
                    ordered.append("")
                    null_count += 1
                    continue
                if validity is not None:
                    validity[i >> 3] |= 1 << (i & 7)
                ordered.append(value)
                if minimum is None or value < minimum:
                    minimum = value
                if maximum is None or value > maximum:
                    maximum = value
            body = _encode_utf8_strings(ordered)

    payload = (bytes(validity) if validity is not None else b"") + body
    return payload, null_count, minimum, maximum


def _encode_utf8_strings(strings: Sequence[str]) -> bytes:
    offsets = bytearray()
    blob = bytearray()
    cursor = 0
    offsets.extend(struct.pack("<I", 0))
    for value in strings:
        raw = value.encode("utf-8")
        cursor += len(raw)
        blob.extend(raw)
        offsets.extend(struct.pack("<I", cursor))
    return bytes(offsets) + bytes(blob)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _decode_column(
    col: ColumnSchema, encoding: str, row_count: int, payload: bytes
) -> list:
    cursor = 0
    validity: bytes | None = None
    if col.nullable:
        size = (row_count + 7) // 8
        if len(payload) < cursor + size:
            raise ColumnarFormatError(f"truncated validity bitmap for {col.name!r}")
        validity = payload[cursor : cursor + size]
        _check_padding(validity, row_count, f"validity bitmap of {col.name!r}")
        cursor += size
    body = payload[cursor:]

    if col.type == "bool":
        size = (row_count + 7) // 8
        if len(body) != size:
            raise ColumnarFormatError(f"bad bool payload length for {col.name!r}")
        _check_padding(body, row_count, f"bool payload of {col.name!r}")
        values = [
            bool((body[i >> 3] >> (i & 7)) & 1) for i in range(row_count)
        ]
    elif col.type == "int64":
        if len(body) != 8 * row_count:
            raise ColumnarFormatError(f"bad int64 payload length for {col.name!r}")
        values = list(struct.unpack(f"<{row_count}q", body)) if row_count else []
    elif col.type == "float64":
        if len(body) != 8 * row_count:
            raise ColumnarFormatError(f"bad float64 payload length for {col.name!r}")
        values = list(struct.unpack(f"<{row_count}d", body)) if row_count else []
        for i, value in enumerate(values):
            if value != value or value in (float("inf"), float("-inf")):
                raise ColumnarFormatError(
                    f"non-finite float64 stored in {col.name!r} row {i}"
                )
    elif encoding == "dictionary":
        values = _decode_dict_utf8(col, row_count, body, validity)
    else:
        values = _decode_plain_utf8(col, row_count, body)

    if validity is not None:
        for i in range(row_count):
            if not ((validity[i >> 3] >> (i & 7)) & 1):
                values[i] = None
    return values


def _decode_utf8_blob(col: ColumnSchema, body: bytes, count: int, *, what: str) -> list[str]:
    offset_bytes = 4 * (count + 1)
    if len(body) < offset_bytes:
        raise ColumnarFormatError(f"truncated {what} offsets for {col.name!r}")
    offsets = struct.unpack(f"<{count + 1}I", body[:offset_bytes])
    blob = body[offset_bytes:]
    if offsets[0] != 0:
        raise ColumnarFormatError(f"{what} offsets must start at 0 for {col.name!r}")
    for prev, nxt in zip(offsets, offsets[1:]):
        if nxt < prev:
            raise ColumnarFormatError(f"non-monotonic {what} offsets for {col.name!r}")
    if offsets[-1] != len(blob):
        raise ColumnarFormatError(f"{what} byte length mismatch for {col.name!r}")
    strings = []
    for i in range(count):
        raw = blob[offsets[i] : offsets[i + 1]]
        try:
            strings.append(raw.decode("utf-8"))
        except UnicodeDecodeError:
            raise ColumnarFormatError(
                f"invalid UTF-8 in {what} entry {i} of {col.name!r}"
            ) from None
    return strings


def _decode_plain_utf8(col: ColumnSchema, row_count: int, body: bytes) -> list[str]:
    return _decode_utf8_blob(col, body, row_count, what="utf8")


def _decode_dict_utf8(
    col: ColumnSchema, row_count: int, body: bytes, validity: bytes | None
) -> list[str]:
    if len(body) < 4:
        raise ColumnarFormatError(f"truncated dictionary header for {col.name!r}")
    (dict_size,) = struct.unpack("<I", body[:4])
    rest = body[4:]
    offset_bytes = 4 * (dict_size + 1)
    # Split dictionary section from the trailing row indices.
    if len(rest) < offset_bytes:
        raise ColumnarFormatError(f"truncated dictionary offsets for {col.name!r}")
    tentative_offsets = struct.unpack(f"<{dict_size + 1}I", rest[:offset_bytes])
    dict_section_len = offset_bytes + tentative_offsets[-1]
    if dict_section_len > len(rest) - 4 * row_count:
        raise ColumnarFormatError(f"dictionary overruns payload for {col.name!r}")
    dictionary = _decode_utf8_blob(
        col, rest[:dict_section_len], dict_size, what="dictionary"
    )
    index_bytes = rest[dict_section_len:]
    if len(index_bytes) != 4 * row_count:
        raise ColumnarFormatError(f"bad dictionary index length for {col.name!r}")
    raw_indices = struct.unpack(f"<{row_count}i", index_bytes) if row_count else ()
    values: list[str] = []
    for i, idx in enumerate(raw_indices):
        is_valid = validity is None or bool((validity[i >> 3] >> (i & 7)) & 1)
        if idx == -1:
            if not col.nullable or is_valid:
                raise ColumnarFormatError(
                    f"NULL dictionary index contradicts validity in {col.name!r} row {i}"
                )
            values.append("")  # replaced with None via validity bitmap below
        elif not 0 <= idx < dict_size:
            raise ColumnarFormatError(
                f"dictionary index {idx} out of range in {col.name!r} row {i}"
            )
        else:
            if not is_valid:
                raise ColumnarFormatError(
                    f"valid dictionary index in NULL slot of {col.name!r} row {i}"
                )
            values.append(dictionary[idx])
    return values


# ---------------------------------------------------------------------------
# Header (JSON metadata)
# ---------------------------------------------------------------------------


def _stats_json(value: Any) -> Any:
    return value


def _build_header(
    table: Table,
    compression: str,
    data_section: bytes,
    raw_section: bytes,
    chunk_info: Sequence[dict],
) -> dict:
    return {
        "compression": compression,
        "row_count": table.row_count,
        "data_length": len(data_section),
        "data_crc32": zlib.crc32(data_section) & 0xFFFFFFFF,
        "uncompressed_length": len(raw_section),
        "columns": list(chunk_info),
    }


def _dump_json(header: dict) -> bytes:
    return json.dumps(
        header, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict:
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise ColumnarFormatError(f"duplicate JSON key in header: {key!r}")
        seen.add(key)
    return dict(pairs)


def _reject_json_constant(value: str):
    raise ColumnarFormatError(f"illegal JSON constant in header: {value}")


def _is_uint32(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 0xFFFFFFFF


def _is_nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_header(header: Any) -> dict:
    if not isinstance(header, dict):
        raise ColumnarFormatError("header must be a JSON object")
    unknown = set(header) - _HEADER_KEYS
    if unknown:
        raise ColumnarFormatError(f"unknown header key(s): {sorted(unknown)!r}")
    compression = header.get("compression")
    if compression not in _COMPRESSION_NAMES:
        raise ColumnarFormatError(f"unknown compression in header: {compression!r}")
    row_count = header.get("row_count")
    if not _is_nonneg_int(row_count):
        raise ColumnarFormatError("header row_count must be a non-negative integer")
    data_length = header.get("data_length")
    if not _is_nonneg_int(data_length):
        raise ColumnarFormatError("header data_length must be a non-negative integer")
    data_crc = header.get("data_crc32")
    if not _is_uint32(data_crc):
        raise ColumnarFormatError("header data_crc32 must be a uint32")
    uncompressed_length = header.get("uncompressed_length")
    if not _is_nonneg_int(uncompressed_length):
        raise ColumnarFormatError("header uncompressed_length must be a non-negative integer")
    entries = header.get("columns")
    if not isinstance(entries, list) or not entries:
        raise ColumnarFormatError("header must list at least one column")

    seen: set[str] = set()
    next_offset = 0
    for pos, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ColumnarFormatError(f"column header #{pos} must be an object")
        unknown_col = set(entry) - _COLUMN_KEYS
        if unknown_col:
            raise ColumnarFormatError(
                f"column header #{pos}: unknown key(s) {sorted(unknown_col)!r}"
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ColumnarFormatError(f"column header #{pos}: name must be a non-empty string")
        if name in seen:
            raise ColumnarFormatError(f"duplicate column in header: {name!r}")
        seen.add(name)
        col_type = entry.get("type")
        if col_type not in _TYPE_NAMES:
            raise ColumnarFormatError(f"column {name!r}: unknown type {col_type!r}")
        nullable = entry.get("nullable")
        if not isinstance(nullable, bool):
            raise ColumnarFormatError(f"column {name!r}: nullable must be true or false")
        encoding = entry.get("encoding")
        if encoding not in _ENCODING_NAMES:
            raise ColumnarFormatError(f"column {name!r}: unknown encoding {encoding!r}")
        if encoding == "dictionary" and col_type != "utf8":
            raise ColumnarFormatError(f"column {name!r}: dictionary encoding requires utf8")
        for key in ("offset", "length", "null_count"):
            if not _is_nonneg_int(entry.get(key)):
                raise ColumnarFormatError(f"column {name!r}: {key} must be a non-negative integer")
        if not _is_uint32(entry.get("crc32")):
            raise ColumnarFormatError(f"column {name!r}: crc32 must be a uint32")
        if "min" not in entry or "max" not in entry:
            raise ColumnarFormatError(f"column {name!r}: min and max keys are required")
        if entry["offset"] != next_offset:
            raise ColumnarFormatError(f"column {name!r}: chunks are not contiguous")
        next_offset += entry["length"]
        null_count = entry["null_count"]
        if null_count > row_count:
            raise ColumnarFormatError(f"column {name!r}: null_count exceeds row_count")
        if not nullable and null_count:
            raise ColumnarFormatError(f"column {name!r}: NULL count set for non-nullable column")
        _validate_stats(entry, col_type, row_count, null_count)
    if next_offset != uncompressed_length:
        raise ColumnarFormatError("declared uncompressed length does not match column chunks")
    return header


def _validate_stats(
    entry: dict, col_type: str, row_count: int, null_count: int
) -> None:
    name = entry["name"]
    minimum = entry.get("min")
    maximum = entry.get("max")
    if null_count == row_count:
        if minimum is not None or maximum is not None:
            raise ColumnarFormatError(f"column {name!r}: all-NULL column needs null min/max")
        return
    if minimum is None or maximum is None:
        raise ColumnarFormatError(f"column {name!r}: min and max must both be present")
    if col_type == "bool":
        if not isinstance(minimum, bool) or not isinstance(maximum, bool):
            raise ColumnarFormatError(f"column {name!r}: bool stats must be booleans")
    elif col_type == "int64":
        if not _is_int64_json(minimum) or not _is_int64_json(maximum):
            raise ColumnarFormatError(f"column {name!r}: int64 stats must be int64 integers")
    elif col_type == "float64":
        if not _is_finite_json_float(minimum) or not _is_finite_json_float(maximum):
            raise ColumnarFormatError(f"column {name!r}: float64 stats must be finite numbers")
        minimum = float(minimum)
        maximum = float(maximum)
    elif col_type == "utf8":
        if not isinstance(minimum, str) or not isinstance(maximum, str):
            raise ColumnarFormatError(f"column {name!r}: utf8 stats must be strings")
    if minimum > maximum:
        raise ColumnarFormatError(f"column {name!r}: min exceeds max")


def _is_int64_json(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and -(2**63) <= value <= 2**63 - 1
    )


def _is_finite_json_float(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    value = float(value)
    return value == value and value not in (float("inf"), float("-inf"))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_file(
    path: str | os.PathLike,
    table: Table,
    *,
    compression: str = "none",
    dictionary_encoding: Sequence[str] = (),
) -> None:
    """Write ``table`` to ``path`` deterministically and atomically.

    All validation happens before any file is created, so invalid arguments
    never create or alter the destination.
    """
    if not isinstance(table, Table):
        raise TypeError("table must be a Table instance")
    if compression not in _COMPRESSION_NAMES:
        raise ValueError(f"unknown compression: {compression!r}")
    if isinstance(dictionary_encoding, str):
        raise ValueError("dictionary_encoding must be a sequence of column names")
    dict_names = tuple(dictionary_encoding)
    if len(set(dict_names)) != len(dict_names):
        raise ValueError("dictionary_encoding contains duplicate column names")
    unknown = [name for name in dict_names if name not in table.schema.names]
    if unknown:
        raise ValueError(f"dictionary_encoding references unknown column(s): {unknown!r}")
    for name in dict_names:
        col = table.schema.columns[table.schema.index(name)]
        if col.type != "utf8":
            raise ValueError(
                f"dictionary encoding is only valid for utf8 columns, not {col.name!r} ({col.type})"
            )
    dict_set = set(dict_names)

    raw_section = bytearray()
    chunk_info: list[dict] = []
    for i, col in enumerate(table.schema.columns):
        values = table._columns[i]
        payload, null_count, minimum, maximum = _encode_payload(
            col, values, use_dictionary=col.name in dict_set
        )
        chunk_info.append(
            {
                "name": col.name,
                "type": col.type,
                "nullable": col.nullable,
                "encoding": "dictionary" if col.name in dict_set else "plain",
                "offset": len(raw_section),
                "length": len(payload),
                "crc32": zlib.crc32(payload) & 0xFFFFFFFF,
                "null_count": null_count,
                "min": _stats_json(minimum),
                "max": _stats_json(maximum),
            }
        )
        raw_section.extend(payload)

    if compression == "zlib":
        data_section = zlib.compress(bytes(raw_section), level=6)
    else:
        data_section = bytes(raw_section)
    header = _build_header(table, compression, data_section, bytes(raw_section), chunk_info)
    header_bytes = _dump_json(header)
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise ValueError("serialized header exceeds the size limit")

    prefix = _MAGIC + bytes([FORMAT_VERSION]) + struct.pack("<I", len(header_bytes))
    body = prefix + header_bytes + data_section
    crc = zlib.crc32(body) & 0xFFFFFFFF
    blob = body + struct.pack("<I", crc) + _FOOTER_MARKER

    _atomic_write(path, blob)


def _atomic_write(path: str | os.PathLike, blob: bytes) -> None:
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_name = tempfile.mkstemp(prefix=".columnar-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _read_exact(handle, size: int, what: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ColumnarFormatError(f"truncated file while reading {what}")
    return data


def _read_header(path: str | os.PathLike) -> tuple[bytes, bytes, dict, int]:
    """Read prefix + JSON header and fstat the declared total size.

    Metadata-only: no data bytes are touched. Returns the raw prefix, raw
    header bytes, the validated header dict and the on-disk file size.
    """
    with open(path, "rb") as handle:
        prefix = handle.read(_PREFIX_LEN)
        if len(prefix) < _PREFIX_LEN:
            if not prefix:
                raise ColumnarFormatError("empty file")
            if len(prefix) >= 4 and prefix[:4] != _MAGIC:
                raise ColumnarFormatError("bad magic number")
            raise ColumnarFormatError("truncated file prefix")
        if prefix[:4] != _MAGIC:
            raise ColumnarFormatError("bad magic number")
        version = prefix[4]
        if version != FORMAT_VERSION:
            raise ColumnarFormatError(f"unsupported format version: {version}")
        (header_length,) = struct.unpack("<I", prefix[5:9])
        if header_length == 0 or header_length > _MAX_HEADER_BYTES:
            raise ColumnarFormatError("header length out of bounds")
        header_bytes = _read_exact(handle, header_length, "header")
        actual_size = os.fstat(handle.fileno()).st_size

    try:
        header = json.loads(
            header_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ColumnarFormatError(f"invalid header JSON: {exc}") from None
    header = _validate_header(header)

    expected_size = _PREFIX_LEN + header_length + header["data_length"] + _FOOTER_LEN
    if actual_size != expected_size:
        raise ColumnarFormatError(
            f"file size {actual_size} does not match declared size {expected_size}"
        )
    return prefix, header_bytes, header, actual_size


def read_file(
    path: str | os.PathLike, *, columns: Sequence[str] | None = None
) -> Table:
    """Read a table, optionally projecting ``columns`` in the given order."""
    prefix, header_bytes, header, _ = _read_header(path)
    with open(path, "rb") as handle:
        handle.seek(_PREFIX_LEN + len(header_bytes))
        data_section = _read_exact(handle, header["data_length"], "data section")
        footer = _read_exact(handle, _FOOTER_LEN, "footer")
        if handle.read(1):
            raise ColumnarFormatError("trailing bytes after footer")

    (footer_crc,) = struct.unpack("<I", footer[:4])
    if footer[4:] != _FOOTER_MARKER:
        raise ColumnarFormatError("bad end marker")
    covered = prefix + header_bytes + data_section
    if (zlib.crc32(covered) & 0xFFFFFFFF) != footer_crc:
        raise ColumnarFormatError("overall checksum mismatch")
    raw_section = _inflate(header, data_section)

    schema_cols = []
    decoded_columns = []
    for entry in header["columns"]:
        col = ColumnSchema(entry["name"], entry["type"], entry["nullable"])
        schema_cols.append(col)
        start = entry["offset"]
        end = start + entry["length"]
        if end > len(raw_section):
            raise ColumnarFormatError(f"column {col.name!r}: chunk outside data section")
        payload = raw_section[start:end]
        if (zlib.crc32(payload) & 0xFFFFFFFF) != entry["crc32"]:
            raise ColumnarFormatError(f"column {col.name!r}: checksum mismatch")
        values = _decode_column(col, entry["encoding"], header["row_count"], payload)
        _verify_stats(col, values, entry)
        decoded_columns.append(values)

    schema = Schema(schema_cols)
    table = Table._from_storage(schema, decoded_columns)
    if columns is not None:
        table = table.project(columns)
    return table


def _inflate(header: dict, data_section: bytes) -> bytes:
    if (zlib.crc32(data_section) & 0xFFFFFFFF) != header["data_crc32"]:
        raise ColumnarFormatError("data section checksum mismatch")
    if header["compression"] == "zlib":
        try:
            raw_section = zlib.decompress(data_section)
        except zlib.error as exc:
            raise ColumnarFormatError(f"cannot decompress data section: {exc}") from None
    else:
        raw_section = data_section
    if len(raw_section) != header["uncompressed_length"]:
        raise ColumnarFormatError("uncompressed data length mismatch")
    return raw_section


def _verify_stats(col: ColumnSchema, values: Sequence, entry: dict) -> None:
    null_count = sum(1 for value in values if value is None)
    if null_count != entry["null_count"]:
        raise ColumnarFormatError(f"column {col.name!r}: null_count mismatch")
    present = [value for value in values if value is not None]
    if not present:
        if entry["min"] is not None or entry["max"] is not None:
            raise ColumnarFormatError(f"column {col.name!r}: stats mismatch")
        return
    minimum = min(present)
    maximum = max(present)
    meta_min = entry["min"]
    meta_max = entry["max"]
    if col.type == "float64":
        meta_min = float(meta_min)
        meta_max = float(meta_max)
    if minimum != meta_min or maximum != meta_max:
        raise ColumnarFormatError(f"column {col.name!r}: min/max mismatch")


def inspect_file(path: str | os.PathLike) -> dict:
    """Read only the file metadata; no data bytes are read, decompressed or decoded.

    Returns a dict with fixed key order: ``format_version``, ``row_count``,
    ``columns``; columns stay in schema order, each with keys ``name``,
    ``type``, ``nullable``, ``row_count``, ``null_count``, ``min``, ``max``.
    """
    _, _, header, _ = _read_header(path)
    return {
        "format_version": FORMAT_VERSION,
        "row_count": header["row_count"],
        "columns": [
            {
                "name": entry["name"],
                "type": entry["type"],
                "nullable": entry["nullable"],
                "row_count": header["row_count"],
                "null_count": entry["null_count"],
                "min": entry["min"],
                "max": entry["max"],
            }
            for entry in header["columns"]
        ],
    }
