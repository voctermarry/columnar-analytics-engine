"""Self-describing columnar file format.

Public API:

* :class:`ColumnSchema` / :class:`Schema` / :class:`Table` -- in-memory model
* :func:`write_file` -- deterministic, atomic write (CAEF v1)
* :func:`write_partitioned_file` -- deterministic, atomic write of contiguous
  row groups (CAEF v2)
* :func:`read_file` -- strict, validating read with optional projection
* :func:`inspect_file` -- metadata-only read
* :func:`inspect_row_groups` -- per-row-group metadata (v2 only)
* :class:`ColumnarFormatError` -- raised for every malformed-file condition

File layout (little-endian)::

    b"CAEF"            magic
    uint8              format version (1 or 2)
    uint32             header byte length
    <header bytes>     UTF-8 JSON, schema + per-column stats/chunk pointers
    <data section>     raw concatenated column payloads, or zlib of the same
    uint32             CRC-32 of every preceding byte
    b"END1"            end marker

Version 1 stores one chunk per column over the whole file.  Version 2 stores a
grid of independent chunks: each row group (a contiguous slice of the original
rows) carries its own ``row_count`` and, per column, ``null_count``/``min``/
``max`` statistics and an independently checksummed, decodable chunk.  The
chunks are laid out row group by row group, column by column.

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
    "FORMAT_VERSION_V1",
    "ColumnarFormatError",
    "ColumnSchema",
    "Schema",
    "Table",
    "write_file",
    "write_partitioned_file",
    "read_file",
    "inspect_file",
    "inspect_row_groups",
]

FORMAT_VERSION = 1
FORMAT_VERSION_V1 = 1
FORMAT_VERSION_V2 = 2
_SUPPORTED_VERSIONS = (1, 2)

_MAGIC = b"CAEF"
_FOOTER_MARKER = b"END1"
_TYPE_NAMES = ("bool", "int64", "float64", "utf8")
_ENCODING_NAMES = ("plain", "dictionary")
_COMPRESSION_NAMES = ("none", "zlib")
_HEADER_KEYS = frozenset(
    ("compression", "row_count", "data_length", "data_crc32", "uncompressed_length", "columns")
)
_V2_EXTRA_HEADER_KEYS = frozenset(("row_group_size", "row_groups"))
_COLUMN_KEYS = frozenset(
    ("name", "type", "nullable", "encoding", "offset", "length", "crc32",
     "null_count", "min", "max")
)
_V2_COLUMN_EXTRA_KEYS = frozenset(("chunks",))
_CHUNK_KEYS = frozenset(
    ("offset", "length", "crc32", "uncompressed_length", "null_count", "min", "max")
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
    def _from_storage(
        cls, schema: Schema, columns: Sequence[Sequence], row_count: int | None = None
    ) -> "Table":
        """Build without re-validating (used by the decoder).

        ``row_count`` is only supplied for the internal zero-column scan used
        by stats-only aggregate queries; otherwise the count comes from the
        first stored column.
        """
        obj = cls.__new__(cls)
        obj.schema = schema
        if row_count is None:
            row_count = len(columns[0]) if columns else 0
        obj._row_count = row_count
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


def _validate_header(header: Any, version: int) -> dict:
    if version == FORMAT_VERSION_V2:
        return _validate_v2_header(header)
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
        _validate_stats(name, entry, col_type, row_count, null_count)
    if next_offset != uncompressed_length:
        raise ColumnarFormatError("declared uncompressed length does not match column chunks")
    return header


def _validate_v2_header(header: Any) -> dict:
    if not isinstance(header, dict):
        raise ColumnarFormatError("header must be a JSON object")
    allowed_keys = _HEADER_KEYS | _V2_EXTRA_HEADER_KEYS
    unknown = set(header) - allowed_keys
    if unknown:
        raise ColumnarFormatError(f"unknown header key(s): {sorted(unknown)!r}")
    compression = header.get("compression")
    if compression not in _COMPRESSION_NAMES:
        raise ColumnarFormatError(f"unknown compression in header: {compression!r}")
    row_count = header.get("row_count")
    if not _is_nonneg_int(row_count):
        raise ColumnarFormatError("header row_count must be a non-negative integer")
    for key in ("data_length", "uncompressed_length"):
        if not _is_nonneg_int(header.get(key)):
            raise ColumnarFormatError(f"header {key} must be a non-negative integer")
    if not _is_uint32(header.get("data_crc32")):
        raise ColumnarFormatError("header data_crc32 must be a uint32")
    group_size = header.get("row_group_size")
    if not _is_positive_int(group_size):
        raise ColumnarFormatError("header row_group_size must be a positive integer")
    groups = header.get("row_groups")
    if not isinstance(groups, list):
        raise ColumnarFormatError("header row_groups must be a list")
    entries = header.get("columns")
    if not isinstance(entries, list) or not entries:
        raise ColumnarFormatError("header must list at least one column")

    seen: set[str] = set()
    column_specs: list[tuple[str, str, bool, str, list]] = []
    for pos, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ColumnarFormatError(f"column header #{pos} must be an object")
        allowed_col_keys = _COLUMN_KEYS - frozenset(
            ("offset", "length", "crc32", "null_count", "min", "max")
        ) | _V2_COLUMN_EXTRA_KEYS
        unknown_col = set(entry) - allowed_col_keys
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
        chunks = entry.get("chunks")
        if not isinstance(chunks, list) or len(chunks) != len(groups):
            raise ColumnarFormatError(
                f"column {name!r}: one chunk per row group is required"
            )
        column_specs.append((name, col_type, nullable, encoding, chunks))

    # Group row counts: contiguous slices of the original rows, so every group
    # except the final one is full and the counts sum to the file row count.
    total = 0
    for g, group in enumerate(groups):
        if not isinstance(group, dict) or set(group) != {"row_count"}:
            raise ColumnarFormatError(f"row group #{g}: must carry only row_count")
        group_rows = group["row_count"]
        if not _is_positive_int(group_rows):
            raise ColumnarFormatError(f"row group #{g}: row_count must be a positive integer")
        if group_rows > group_size:
            raise ColumnarFormatError(f"row group #{g}: row_count exceeds row_group_size")
        if g < len(groups) - 1 and group_rows != group_size:
            raise ColumnarFormatError(f"row group #{g}: only the final group may be partial")
        total += group_rows
    if total != row_count:
        raise ColumnarFormatError("row group row counts do not sum to row_count")
    if row_count == 0:
        if groups:
            raise ColumnarFormatError("an empty file must not declare row groups")
        if header["data_length"] != 0 or header["uncompressed_length"] != 0:
            raise ColumnarFormatError("an empty file must not carry a data section")

    # Chunks are laid out group by group, column by column; offsets are
    # contiguous across the whole stored data section and every chunk matches
    # its group's row count / nullability / statistics.
    next_offset = 0
    uncompressed_total = 0
    for g, group in enumerate(groups):
        group_rows = group["row_count"]
        for name, col_type, nullable, encoding, chunks in column_specs:
            chunk = chunks[g]
            if not isinstance(chunk, dict):
                raise ColumnarFormatError(f"column {name!r} group #{g}: chunk must be an object")
            unknown_chunk = set(chunk) - _CHUNK_KEYS
            if unknown_chunk:
                raise ColumnarFormatError(
                    f"column {name!r} group #{g}: unknown key(s) {sorted(unknown_chunk)!r}"
                )
            for key in ("offset", "length", "null_count"):
                if not _is_nonneg_int(chunk.get(key)):
                    raise ColumnarFormatError(
                        f"column {name!r} group #{g}: {key} must be a non-negative integer"
                    )
            if not _is_nonneg_int(chunk.get("uncompressed_length")):
                raise ColumnarFormatError(
                    f"column {name!r} group #{g}: uncompressed_length must be a non-negative integer"
                )
            if not _is_uint32(chunk.get("crc32")):
                raise ColumnarFormatError(
                    f"column {name!r} group #{g}: crc32 must be a uint32"
                )
            if "min" not in chunk or "max" not in chunk:
                raise ColumnarFormatError(
                    f"column {name!r} group #{g}: min and max keys are required"
                )
            if chunk["offset"] != next_offset:
                raise ColumnarFormatError(f"column {name!r} group #{g}: chunks are not contiguous")
            next_offset += chunk["length"]
            uncompressed_total += chunk["uncompressed_length"]
            null_count = chunk["null_count"]
            if null_count > group_rows:
                raise ColumnarFormatError(
                    f"column {name!r} group #{g}: null_count exceeds group row_count"
                )
            if not nullable and null_count:
                raise ColumnarFormatError(
                    f"column {name!r} group #{g}: NULL count set for non-nullable column"
                )
            _validate_stats(name, chunk, col_type, group_rows, null_count)
    if next_offset != header["data_length"]:
        raise ColumnarFormatError("declared data length does not match column chunks")
    if uncompressed_total != header["uncompressed_length"]:
        raise ColumnarFormatError("declared uncompressed length does not match column chunks")
    return header


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_stats(
    name: str, entry: dict, col_type: str, row_count: int, null_count: int
) -> None:
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


def _validate_write_options(
    table: Table, compression: Any, dictionary_encoding: Any
) -> set[str]:
    """Validate the shared write options; return the dictionary-encoded column set."""
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
    return set(dict_names)


def write_file(
    path: str | os.PathLike,
    table: Table,
    *,
    compression: str = "none",
    dictionary_encoding: Sequence[str] = (),
) -> None:
    """Write ``table`` to ``path`` deterministically and atomically (CAEF v1).

    All validation happens before any file is created, so invalid arguments
    never create or alter the destination.
    """
    dict_set = _validate_write_options(table, compression, dictionary_encoding)

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
    _write_final(path, header, data_section, FORMAT_VERSION_V1)


def write_partitioned_file(
    path: str | os.PathLike,
    table: Table,
    row_group_size: int,
    *,
    compression: str = "none",
    dictionary_encoding: Sequence[str] = (),
) -> None:
    """Write ``table`` as contiguous row groups (CAEF v2), atomically.

    ``row_group_size`` is a positive integer: the table's rows are split in
    their original order into groups of that size, the final group being
    shorter.  Each column chunk is stored, checksummed, compressed and decoded
    independently and carries its own ``null_count``/``min``/``max``.
    Identical inputs and options always produce identical bytes, and all
    validation happens before any file is created.

    A non-positive or non-integer ``row_group_size`` (or any illegal encoding
    option) raises :class:`ValueError` and never creates or alters the
    destination.
    """
    dict_set = _validate_write_options(table, compression, dictionary_encoding)
    if not _is_positive_int(row_group_size):
        raise ValueError("row_group_size must be a positive integer")

    row_count = table.row_count
    group_bounds = list(range(0, row_count, row_group_size))
    groups_meta: list[dict] = []
    # chunks_meta[i] holds the chunk descriptors of column i in group order.
    chunks_meta: list[list[dict]] = [[] for _ in table.schema.columns]
    data_section = bytearray()
    uncompressed_length = 0

    for start in group_bounds:
        end = min(start + row_group_size, row_count)
        groups_meta.append({"row_count": end - start})
        for i, col in enumerate(table.schema.columns):
            values = table._columns[i][start:end]
            payload, null_count, minimum, maximum = _encode_payload(
                col, values, use_dictionary=col.name in dict_set
            )
            if compression == "zlib":
                stored = zlib.compress(bytes(payload), level=6)
            else:
                stored = payload
            chunks_meta[i].append(
                {
                    "offset": len(data_section),
                    "length": len(stored),
                    "crc32": zlib.crc32(stored) & 0xFFFFFFFF,
                    "uncompressed_length": len(payload),
                    "null_count": null_count,
                    "min": _stats_json(minimum),
                    "max": _stats_json(maximum),
                }
            )
            data_section.extend(stored)
            uncompressed_length += len(payload)

    column_info = []
    for i, col in enumerate(table.schema.columns):
        column_info.append(
            {
                "name": col.name,
                "type": col.type,
                "nullable": col.nullable,
                "encoding": "dictionary" if col.name in dict_set else "plain",
                "chunks": chunks_meta[i],
            }
        )
    header = {
        "compression": compression,
        "row_count": row_count,
        "row_group_size": row_group_size,
        "row_groups": groups_meta,
        "data_length": len(data_section),
        "data_crc32": zlib.crc32(bytes(data_section)) & 0xFFFFFFFF,
        "uncompressed_length": uncompressed_length,
        "columns": column_info,
    }
    _write_final(path, header, bytes(data_section), FORMAT_VERSION_V2)


def _write_final(
    path: str | os.PathLike, header: dict, data_section: bytes, version: int
) -> None:
    header_bytes = _dump_json(header)
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise ValueError("serialized header exceeds the size limit")

    prefix = _MAGIC + bytes([version]) + struct.pack("<I", len(header_bytes))
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


def _read_header(path: str | os.PathLike) -> tuple[bytes, bytes, dict, int, int]:
    """Read prefix + JSON header and fstat the declared total size.

    Metadata-only: no data bytes are touched. Returns the raw prefix, raw
    header bytes, the validated header dict, the on-disk file size and the
    format version.
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
        if version not in _SUPPORTED_VERSIONS:
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
    header = _validate_header(header, version)

    expected_size = _PREFIX_LEN + header_length + header["data_length"] + _FOOTER_LEN
    if actual_size != expected_size:
        raise ColumnarFormatError(
            f"file size {actual_size} does not match declared size {expected_size}"
        )
    return prefix, header_bytes, header, actual_size, version


def _read_file_handle(path, data_length: int) -> tuple[bytes, bytes, bytes]:
    """Read the data section and footer, verifying the end marker and CRC.

    Returns ``(prefix_and_header, data_section, footer)``; the combined CRC of
    every byte preceding the footer is checked here.
    """
    prefix_and_header = b""
    with open(path, "rb") as handle:
        prefix_and_header = handle.read(_PREFIX_LEN)
        (header_length,) = struct.unpack("<I", prefix_and_header[5:9])
        prefix_and_header += _read_exact(handle, header_length, "header")
        data_section = _read_exact(handle, data_length, "data section")
        footer = _read_exact(handle, _FOOTER_LEN, "footer")
        if handle.read(1):
            raise ColumnarFormatError("trailing bytes after footer")

    (footer_crc,) = struct.unpack("<I", footer[:4])
    if footer[4:] != _FOOTER_MARKER:
        raise ColumnarFormatError("bad end marker")
    covered = prefix_and_header + data_section
    if (zlib.crc32(covered) & 0xFFFFFFFF) != footer_crc:
        raise ColumnarFormatError("overall checksum mismatch")
    return prefix_and_header, data_section, footer


def read_file(
    path: str | os.PathLike,
    *,
    columns: Sequence[str] | None = None,
    row_groups: Sequence[int] | None = None,
) -> Table:
    """Read a table, optionally projecting ``columns`` in the given order.

    For v2 files ``row_groups`` may further restrict decoding to a subset
    of row-group indices (file order); the returned rows are the
    concatenation of those groups in file order.  Unreferenced column chunks
    and excluded row groups are never decompressed or decoded.
    """
    _prefix, _header_bytes, header, _size, version = _read_header(path)
    if version == FORMAT_VERSION_V2:
        return _read_v2(path, header, columns, row_groups)

    _, data_section, _ = _read_file_handle(path, header["data_length"])
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


def _selected_group_indices(header: dict, row_groups: Sequence[int] | None) -> list[int]:
    count = len(header["row_groups"])
    if row_groups is None:
        return list(range(count))
    if isinstance(row_groups, (str, bytes)) or not isinstance(row_groups, Sequence):
        raise ValueError("row_groups must be a sequence of indices")
    selected: list[int] = []
    for index in row_groups:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("row_group indices must be integers")
        if not 0 <= index < count:
            raise ValueError(f"row_group index {index} out of range 0..{count - 1}")
        selected.append(index)
    return selected


def _read_v2(path, header: dict, columns, row_groups) -> Table:
    schema = Schema(
        tuple(
            ColumnSchema(entry["name"], entry["type"], entry["nullable"])
            for entry in header["columns"]
        )
    )
    if columns is None:
        wanted = list(range(len(schema.columns)))
    else:
        if isinstance(columns, str):
            raise ValueError("columns must be a sequence of strings, not a single string")
        names = tuple(columns)
        if len(set(names)) != len(names):
            raise ValueError("projection contains duplicate column names")
        wanted = [schema.index(name) for name in names]  # KeyError on unknown name

    group_indices = _selected_group_indices(header, row_groups)
    group_rows = [group["row_count"] for group in header["row_groups"]]
    selected_count = sum(group_rows[g] for g in group_indices)

    _, data_section, _ = _read_file_handle(path, header["data_length"])
    if (zlib.crc32(data_section) & 0xFFFFFFFF) != header["data_crc32"]:
        raise ColumnarFormatError("data section checksum mismatch")

    decoded: dict[int, list] = {}
    for col_index in wanted:
        entry = header["columns"][col_index]
        col = schema.columns[col_index]
        chunks = entry["chunks"]
        values: list = []
        for g in group_indices:
            chunk = chunks[g]
            stored = data_section[chunk["offset"] : chunk["offset"] + chunk["length"]]
            if len(stored) != chunk["length"]:
                raise ColumnarFormatError(f"column {col.name!r}: chunk outside data section")
            if (zlib.crc32(stored) & 0xFFFFFFFF) != chunk["crc32"]:
                raise ColumnarFormatError(f"column {col.name!r}: checksum mismatch")
            if header["compression"] == "zlib":
                try:
                    payload = zlib.decompress(stored)
                except zlib.error as exc:
                    raise ColumnarFormatError(
                        f"cannot decompress chunk of column {col.name!r}: {exc}"
                    ) from None
            else:
                payload = stored
            if len(payload) != chunk["uncompressed_length"]:
                raise ColumnarFormatError(
                    f"column {col.name!r}: uncompressed chunk length mismatch"
                )
            chunk_values = _decode_column(col, entry["encoding"], group_rows[g], payload)
            _verify_stats(col, chunk_values, chunk)
            values.extend(chunk_values)
        decoded[col_index] = values

    if columns is None:
        all_columns = [decoded[i] for i in wanted]
        return Table._from_storage(schema, all_columns)
    if not wanted:
        # Internal zero-column scan (e.g. COUNT(*)): keep the full schema for
        # binding but decode no chunks; the selected row count is carried along.
        return Table._from_storage(schema, [], row_count=selected_count)
    # An explicit projection yields only the requested columns.
    projected_schema = Schema(tuple(schema.columns[i] for i in wanted))
    all_columns = [decoded[i] for i in wanted]
    return Table._from_storage(projected_schema, all_columns, row_count=selected_count)


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
    Version 1 and version 2 files share this structure: v2 statistics are
    aggregated over all row groups.
    """
    _, _, header, _, version = _read_header(path)
    if version == FORMAT_VERSION_V2:
        columns = []
        for entry in header["columns"]:
            null_count = sum(chunk["null_count"] for chunk in entry["chunks"])
            mins = [chunk["min"] for chunk in entry["chunks"] if chunk["min"] is not None]
            maxs = [chunk["max"] for chunk in entry["chunks"] if chunk["max"] is not None]
            columns.append(
                {
                    "name": entry["name"],
                    "type": entry["type"],
                    "nullable": entry["nullable"],
                    "row_count": header["row_count"],
                    "null_count": null_count,
                    "min": min(mins) if mins else None,
                    "max": max(maxs) if maxs else None,
                }
            )
        return {
            "format_version": FORMAT_VERSION_V2,
            "row_count": header["row_count"],
            "columns": columns,
        }
    return {
        "format_version": FORMAT_VERSION_V1,
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


def inspect_row_groups(path: str | os.PathLike) -> dict:
    """Read per-row-group metadata without touching any data bytes.

    Returns a dict with fixed key order: ``format_version``, ``row_count``,
    ``row_groups``.  Each row group is listed in file order with its
    ``row_count`` and a ``columns`` list in schema order carrying
    ``name``, ``type``, ``nullable``, ``null_count``, ``min`` and
    ``max``.  Version 1 files return an empty ``row_groups`` list.
    """
    _, _, header, _, version = _read_header(path)
    if version == FORMAT_VERSION_V1:
        return {
            "format_version": FORMAT_VERSION_V1,
            "row_count": header["row_count"],
            "row_groups": [],
        }
    groups = []
    for g, group in enumerate(header["row_groups"]):
        groups.append(
            {
                "row_count": group["row_count"],
                "columns": [
                    {
                        "name": entry["name"],
                        "type": entry["type"],
                        "nullable": entry["nullable"],
                        "null_count": entry["chunks"][g]["null_count"],
                        "min": entry["chunks"][g]["min"],
                        "max": entry["chunks"][g]["max"],
                    }
                    for entry in header["columns"]
                ],
            }
        )
    return {
        "format_version": FORMAT_VERSION_V2,
        "row_count": header["row_count"],
        "row_groups": groups,
    }
