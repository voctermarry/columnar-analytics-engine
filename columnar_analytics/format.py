"""Self-describing columnar file format.

Public API:

* :class:`ColumnSchema` / :class:`Schema` / :class:`Table` -- in-memory model
* :func:`write_file` -- deterministic, atomic write of the v1 format
* :func:`write_partitioned_file` -- deterministic, atomic write of the
  row-group-partitioned v2 format
* :func:`read_file` -- strict, validating read with optional projection
  (both format versions)
* :func:`inspect_file` -- metadata-only read (both format versions)
* :func:`inspect_row_groups` -- metadata-only per-row-group statistics
* :class:`ColumnarFormatError` -- raised for every malformed-file condition

File layout (little-endian)::

    b"CAEF"            magic
    uint8              format version (1 or 2)
    uint32             header byte length
    <header bytes>     UTF-8 JSON, schema + per-column stats/chunk pointers
    <data section>     raw concatenated column payloads, or zlib of the same
    uint32             CRC-32 of every preceding byte   (version 1 only)
    b"END1"            end marker

Version 1 stores one chunk per column and optionally zlib-compresses the
whole data section.  Version 2 (written by :func:`write_partitioned_file`)
splits the rows into consecutive row groups of a fixed size; every group
carries its row count and per-column ``null_count`` / ``min`` / ``max``
statistics, and every (group, column) block is stored -- and optionally
zlib-compressed -- independently, so a single block can be located,
checksummed, decompressed and decoded without touching the others.  The v2
trailer is the bare ``END1`` marker; integrity comes from the declared
sizes and the per-block CRC-32 checksums of the blocks actually read.

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
    "FORMAT_VERSION_PARTITIONED",
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
# Version 2: the row-group-partitioned layout written by
# ``write_partitioned_file``.  Readers dispatch on the version byte.
FORMAT_VERSION_PARTITIONED = 2

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
_V2_HEADER_KEYS = frozenset(
    ("compression", "row_count", "row_group_size", "data_length", "columns", "row_groups")
)
_V2_COLUMN_KEYS = frozenset(("name", "type", "nullable", "encoding"))
_V2_GROUP_KEYS = frozenset(("row_count", "columns"))
_V2_BLOCK_KEYS = frozenset(
    ("offset", "length", "crc32", "uncompressed_length", "null_count", "min", "max")
)
_PREFIX_LEN = 9  # magic(4) + version(1) + header length uint32
_FOOTER_LEN = 8  # crc32 uint32 + marker
_V2_TRAILER_LEN = 4  # bare end marker
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


def _validate_header_v2(header: Any) -> dict:
    """Validate the JSON header of a row-group-partitioned (v2) file."""
    if not isinstance(header, dict):
        raise ColumnarFormatError("header must be a JSON object")
    unknown = set(header) - _V2_HEADER_KEYS
    if unknown:
        raise ColumnarFormatError(f"unknown header key(s): {sorted(unknown)!r}")
    compression = header.get("compression")
    if compression not in _COMPRESSION_NAMES:
        raise ColumnarFormatError(f"unknown compression in header: {compression!r}")
    row_count = header.get("row_count")
    if not _is_nonneg_int(row_count):
        raise ColumnarFormatError("header row_count must be a non-negative integer")
    row_group_size = header.get("row_group_size")
    if (
        not isinstance(row_group_size, int)
        or isinstance(row_group_size, bool)
        or row_group_size < 1
    ):
        raise ColumnarFormatError("header row_group_size must be a positive integer")
    data_length = header.get("data_length")
    if not _is_nonneg_int(data_length):
        raise ColumnarFormatError("header data_length must be a non-negative integer")
    entries = header.get("columns")
    if not isinstance(entries, list) or not entries:
        raise ColumnarFormatError("header must list at least one column")

    columns: list[dict] = []
    seen: set[str] = set()
    for pos, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ColumnarFormatError(f"column header #{pos} must be an object")
        unknown_col = set(entry) - _V2_COLUMN_KEYS
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
        columns.append(entry)

    groups = header.get("row_groups")
    if not isinstance(groups, list):
        raise ColumnarFormatError("header row_groups must be a list")
    total_rows = 0
    next_offset = 0
    for pos, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ColumnarFormatError(f"row group #{pos} must be an object")
        unknown_group = set(group) - _V2_GROUP_KEYS
        if unknown_group:
            raise ColumnarFormatError(
                f"row group #{pos}: unknown key(s) {sorted(unknown_group)!r}"
            )
        group_rows = group.get("row_count")
        if (
            not isinstance(group_rows, int)
            or isinstance(group_rows, bool)
            or group_rows < 1
        ):
            raise ColumnarFormatError(
                f"row group #{pos}: row_count must be a positive integer"
            )
        if pos < len(groups) - 1 and group_rows != row_group_size:
            raise ColumnarFormatError(
                f"row group #{pos}: row_count does not match row_group_size"
            )
        if group_rows > row_group_size:
            raise ColumnarFormatError(
                f"row group #{pos}: row_count exceeds row_group_size"
            )
        total_rows += group_rows
        blocks = group.get("columns")
        if not isinstance(blocks, list) or len(blocks) != len(columns):
            raise ColumnarFormatError(
                f"row group #{pos}: must carry one block per column"
            )
        for col, block in zip(columns, blocks):
            name = col["name"]
            if not isinstance(block, dict):
                raise ColumnarFormatError(
                    f"row group #{pos} column {name!r}: block must be an object"
                )
            unknown_block = set(block) - _V2_BLOCK_KEYS
            if unknown_block:
                raise ColumnarFormatError(
                    f"row group #{pos} column {name!r}: unknown key(s) "
                    f"{sorted(unknown_block)!r}"
                )
            for key in ("offset", "length", "uncompressed_length", "null_count"):
                if not _is_nonneg_int(block.get(key)):
                    raise ColumnarFormatError(
                        f"row group #{pos} column {name!r}: {key} must be a "
                        "non-negative integer"
                    )
            if not _is_uint32(block.get("crc32")):
                raise ColumnarFormatError(
                    f"row group #{pos} column {name!r}: crc32 must be a uint32"
                )
            if compression == "none" and block["length"] != block["uncompressed_length"]:
                raise ColumnarFormatError(
                    f"row group #{pos} column {name!r}: uncompressed block "
                    "length mismatch"
                )
            if block["offset"] != next_offset:
                raise ColumnarFormatError(
                    f"row group #{pos} column {name!r}: blocks are not contiguous"
                )
            next_offset += block["length"]
            null_count = block["null_count"]
            if null_count > group_rows:
                raise ColumnarFormatError(
                    f"row group #{pos} column {name!r}: null_count exceeds group row_count"
                )
            if not col["nullable"] and null_count:
                raise ColumnarFormatError(
                    f"row group #{pos} column {name!r}: NULL count set for "
                    "non-nullable column"
                )
            if "min" not in block or "max" not in block:
                raise ColumnarFormatError(
                    f"row group #{pos} column {name!r}: min and max keys are required"
                )
            _validate_stats(
                {"name": name, "min": block["min"], "max": block["max"]},
                col["type"],
                group_rows,
                null_count,
            )
    if total_rows != row_count:
        raise ColumnarFormatError("row group row counts do not add up to row_count")
    if next_offset != data_length:
        raise ColumnarFormatError("declared data length does not match column blocks")
    return header


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _validate_write_options(
    table: Table, compression: Any, dictionary_encoding: Any
) -> frozenset:
    """Shared write-option validation; returns the dictionary column set.

    Everything is checked before any file is created, so invalid arguments
    never create or alter the destination.
    """
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
    return frozenset(dict_names)


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
    header_bytes = _dump_json(header)
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise ValueError("serialized header exceeds the size limit")

    prefix = _MAGIC + bytes([FORMAT_VERSION]) + struct.pack("<I", len(header_bytes))
    body = prefix + header_bytes + data_section
    crc = zlib.crc32(body) & 0xFFFFFFFF
    blob = body + struct.pack("<I", crc) + _FOOTER_MARKER

    _atomic_write(path, blob)


def write_partitioned_file(
    path: str | os.PathLike,
    table: Table,
    row_group_size: int,
    *,
    compression: str = "none",
    dictionary_encoding: Sequence[str] = (),
) -> None:
    """Write ``table`` as a row-group-partitioned (v2) columnar file.

    Rows are split, in their original order, into consecutive groups of
    ``row_group_size`` rows (the last group holds the remainder; an empty
    table yields no groups).  Every group records its row count and, per
    column, ``null_count`` / ``min`` / ``max`` statistics; every
    (group, column) block is checksummed and -- with ``compression="zlib"``
    -- compressed independently, so readers can verify, decompress and
    decode exactly the blocks they need.

    The same inputs and options always produce identical bytes, and the
    destination is replaced atomically.  All validation happens before any
    file is created: a non-positive (or non-integer) ``row_group_size`` and
    invalid ``compression`` / ``dictionary_encoding`` arguments raise
    :class:`ValueError` without creating or altering the target.
    """
    if not isinstance(table, Table):
        raise TypeError("table must be a Table instance")
    if (
        not isinstance(row_group_size, int)
        or isinstance(row_group_size, bool)
        or row_group_size < 1
    ):
        raise ValueError("row_group_size must be a positive integer")
    dict_set = _validate_write_options(table, compression, dictionary_encoding)

    row_count = table.row_count
    group_bounds: list[tuple[int, int]] = []
    start = 0
    while start < row_count:
        end = min(start + row_group_size, row_count)
        group_bounds.append((start, end))
        start = end

    data_section = bytearray()
    group_headers: list[dict] = []
    for group_start, group_end in group_bounds:
        block_info: list[dict] = []
        for i, col in enumerate(table.schema.columns):
            values = table._columns[i][group_start:group_end]
            payload, null_count, minimum, maximum = _encode_payload(
                col, values, use_dictionary=col.name in dict_set
            )
            if compression == "zlib":
                stored = zlib.compress(payload, level=6)
            else:
                stored = payload
            block_info.append(
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
        group_headers.append(
            {"row_count": group_end - group_start, "columns": block_info}
        )

    header = {
        "compression": compression,
        "row_count": row_count,
        "row_group_size": row_group_size,
        "data_length": len(data_section),
        "columns": [
            {
                "name": col.name,
                "type": col.type,
                "nullable": col.nullable,
                "encoding": "dictionary" if col.name in dict_set else "plain",
            }
            for col in table.schema.columns
        ],
        "row_groups": group_headers,
    }
    header_bytes = _dump_json(header)
    if len(header_bytes) > _MAX_HEADER_BYTES:
        raise ValueError("serialized header exceeds the size limit")

    blob = (
        _MAGIC
        + bytes([FORMAT_VERSION_PARTITIONED])
        + struct.pack("<I", len(header_bytes))
        + header_bytes
        + bytes(data_section)
        + _FOOTER_MARKER
    )
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
    format version.  Both v1 and v2 headers are accepted.
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
        if version not in (FORMAT_VERSION, FORMAT_VERSION_PARTITIONED):
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
    if version == FORMAT_VERSION_PARTITIONED:
        header = _validate_header_v2(header)
        trailer_len = _V2_TRAILER_LEN
    else:
        header = _validate_header(header)
        trailer_len = _FOOTER_LEN

    expected_size = _PREFIX_LEN + header_length + header["data_length"] + trailer_len
    if actual_size != expected_size:
        raise ColumnarFormatError(
            f"file size {actual_size} does not match declared size {expected_size}"
        )
    return prefix, header_bytes, header, actual_size, version


def read_file(
    path: str | os.PathLike, *, columns: Sequence[str] | None = None
) -> Table:
    """Read a table, optionally projecting ``columns`` in the given order.

    Both format versions are accepted.  For v2 (row-group-partitioned)
    files only the blocks of the requested columns are decompressed and
    decoded; unreferenced column blocks are never touched beyond their
    declared sizes.
    """
    if columns is not None and not isinstance(columns, str):
        # Materialise once so one-shot iterables survive both the decode
        # set computation and the final projection.
        columns = tuple(columns)
    prefix, header_bytes, header, _, version = _read_header(path)
    if version == FORMAT_VERSION_PARTITIONED:
        table = _read_partitioned_table(
            path,
            columns=None if columns is None else set(columns),
            row_groups=None,
            _header=(header, _PREFIX_LEN + len(header_bytes)),
        )
    else:
        table = _read_v1(path, prefix, header_bytes, header)
    if columns is not None:
        table = table.project(columns)
    return table


def _read_v1(path: str | os.PathLike, prefix: bytes, header_bytes: bytes, header: dict) -> Table:
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
    return Table._from_storage(schema, decoded_columns)


class _UnreadColumn:
    """Placeholder for a v2 column whose blocks were not decoded.

    The query engine's bound plans only ever index referenced (decoded)
    columns, so placeholders are never actually read; any access yields
    ``None`` rather than failing loudly.
    """

    __slots__ = ("_length",)

    def __init__(self, length: int):
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index):
        return None


def _schema_from_header_v2(header: dict) -> Schema:
    return Schema(
        [
            ColumnSchema(entry["name"], entry["type"], entry["nullable"])
            for entry in header["columns"]
        ]
    )


def _row_groups_from_header_v2(header: dict) -> list[dict]:
    """Per-group statistics of a validated v2 header, in file order."""
    names = [entry["name"] for entry in header["columns"]]
    return [
        {
            "row_count": group["row_count"],
            "columns": [
                {
                    "name": name,
                    "null_count": block["null_count"],
                    "min": block["min"],
                    "max": block["max"],
                }
                for name, block in zip(names, group["columns"])
            ],
        }
        for group in header["row_groups"]
    ]


def _read_partitioned_table(
    path: str | os.PathLike,
    columns: set[str] | None = None,
    row_groups: Sequence[int] | None = None,
    _header: tuple[dict, int] | None = None,
) -> Table:
    """Read selected columns and row groups of a v2 file.

    ``columns`` restricts which column blocks are decompressed and decoded
    (``None`` decodes all); ``row_groups`` restricts which groups contribute
    rows, in file order (``None`` takes every group).  The returned table
    carries the full schema; columns that were not decoded are inert
    placeholders.  Only the blocks actually read have their checksums,
    decompressed lengths and statistics verified.
    """
    if _header is None:
        _, header_bytes, header, _, version = _read_header(path)
        if version != FORMAT_VERSION_PARTITIONED:
            raise ColumnarFormatError("not a partitioned (v2) file")
        data_start = _PREFIX_LEN + len(header_bytes)
    else:
        header, data_start = _header
    schema = _schema_from_header_v2(header)
    names = schema.names
    if columns is None:
        decode_indices = set(range(len(names)))
    else:
        decode_indices = {i for i, name in enumerate(names) if name in columns}
    groups = header["row_groups"]
    group_indices = range(len(groups)) if row_groups is None else row_groups
    encodings = [entry["encoding"] for entry in header["columns"]]
    compression = header["compression"]

    decoded: dict[int, list] = {i: [] for i in decode_indices}
    total_rows = 0
    with open(path, "rb") as handle:
        handle.seek(-_V2_TRAILER_LEN, os.SEEK_END)
        if handle.read(_V2_TRAILER_LEN) != _FOOTER_MARKER:
            raise ColumnarFormatError("bad end marker")
        for group_index in group_indices:
            group = groups[group_index]
            group_rows = group["row_count"]
            total_rows += group_rows
            blocks = group["columns"]
            for i in decode_indices:
                block = blocks[i]
                handle.seek(data_start + block["offset"])
                stored = _read_exact(
                    handle, block["length"], f"block of column {names[i]!r}"
                )
                if (zlib.crc32(stored) & 0xFFFFFFFF) != block["crc32"]:
                    raise ColumnarFormatError(
                        f"column {names[i]!r}: block checksum mismatch"
                    )
                payload = _inflate_block(compression, stored, block)
                values = _decode_column(
                    schema.columns[i], encodings[i], group_rows, payload
                )
                _verify_stats(schema.columns[i], values, block)
                decoded[i].extend(values)

    out_columns: list = []
    for i in range(len(names)):
        if i in decode_indices:
            out_columns.append(tuple(decoded[i]))
        else:
            out_columns.append(_UnreadColumn(total_rows))
    table = Table.__new__(Table)
    table.schema = schema
    table._row_count = total_rows
    table._columns = tuple(out_columns)
    return table


def _inflate_block(compression: str, stored: bytes, block: dict) -> bytes:
    if compression == "zlib":
        try:
            payload = zlib.decompress(stored)
        except zlib.error as exc:
            raise ColumnarFormatError(f"cannot decompress column block: {exc}") from None
    else:
        payload = stored
    if len(payload) != block["uncompressed_length"]:
        raise ColumnarFormatError("uncompressed block length mismatch")
    return payload


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
    For v2 (row-group-partitioned) files the column statistics are the
    aggregates of the per-group statistics.
    """
    _, _, header, _, version = _read_header(path)
    if version == FORMAT_VERSION_PARTITIONED:
        groups = _row_groups_from_header_v2(header)
        columns = []
        for i, entry in enumerate(header["columns"]):
            null_count = sum(group["columns"][i]["null_count"] for group in groups)
            present_min = [
                group["columns"][i]["min"]
                for group in groups
                if group["columns"][i]["min"] is not None
            ]
            present_max = [
                group["columns"][i]["max"]
                for group in groups
                if group["columns"][i]["max"] is not None
            ]
            columns.append(
                {
                    "name": entry["name"],
                    "type": entry["type"],
                    "nullable": entry["nullable"],
                    "row_count": header["row_count"],
                    "null_count": null_count,
                    "min": min(present_min) if present_min else None,
                    "max": max(present_max) if present_max else None,
                }
            )
        return {
            "format_version": FORMAT_VERSION_PARTITIONED,
            "row_count": header["row_count"],
            "columns": columns,
        }
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


def inspect_row_groups(path: str | os.PathLike) -> list[dict]:
    """Read only the file metadata and return per-row-group statistics.

    For a v2 (row-group-partitioned) file the result lists one entry per
    row group, in file order; each entry carries the group's ``row_count``
    and its per-column statistics (``name``, ``null_count``, ``min``,
    ``max``) in schema order.  A v1 file has no row groups and yields an
    empty list.  No data bytes are read, decompressed or decoded.
    """
    _, _, header, _, version = _read_header(path)
    if version != FORMAT_VERSION_PARTITIONED:
        return []
    return _row_groups_from_header_v2(header)
