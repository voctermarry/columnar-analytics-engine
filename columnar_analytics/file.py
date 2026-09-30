"""Self-describing columnar file format: write, read and inspect.

File layout (all multi-byte integers are little-endian)::

    magic (4 bytes, b"CAEF")
    format version (uint16)
    magic (4 bytes, b"CAEF")
    format version (uint16)
    header frame:
        payload length (uint32) / stored length (uint32) / codec byte / bytes
        payload is JSON UTF-8 (frame always zlib compressed)
    per-column blocks; each data block is followed by a second frame holding
    the dictionary (uncompressed) for dictionary-encoded utf8 columns:
        payload length (uint32) / stored length (uint32) / codec byte / bytes
    footer frame (same framing, always zlib compressed; JSON payload with
        per-column SHA-256 checksums over the decompressed block payloads)
    SHA-256 checksum over every preceding byte (32 raw bytes)

A codec byte of 0 stores the payload raw; 1 marks zlib compression.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import zlib
from dataclasses import dataclass
from typing import Any

from .table import BOOL, FLOAT64, INT64, UTF8, Field, Schema, Table

MAGIC = b"CAEF"
FORMAT_VERSION = 1
COMPRESSIONS = frozenset({"none", "zlib"})
_CHECKSUM_ALGORITHM = "sha256"
_CHECKSUM_BYTES = 32

# Struct formats.
_U16 = struct.Struct("<H")
_U32 = struct.Struct("<I")
_D_BOOL = struct.Struct("<?")
_Q_INT = struct.Struct("<q")
_D_FLOAT = struct.Struct("<d")
_I_DICT_INDEX = struct.Struct("<i")  # -1 encodes NULL

_CODEC_RAW = 0
_CODEC_ZLIB = 1
_CODECS = frozenset({_CODEC_RAW, _CODEC_ZLIB})


class ColumnarFormatError(Exception):
    """Raised when a file cannot be parsed or fails format validation."""


@dataclass(frozen=True)
class WriteOptions:
    """Writer options.

    compression: "none" or "zlib", applied to every column block.
    dict_encoding: iterable of utf8 column names that use dictionary encoding.
    """

    compression: str = "none"
    dict_encoding: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.compression not in COMPRESSIONS:
            raise ValueError(f"unknown compression: {self.compression!r}")
        if isinstance(self.dict_encoding, str):
            # Guard against the easy mistake of passing one column name as a
            # bare string; it would otherwise be iterated character-wise.
            raise ValueError("dict_encoding must be an iterable of column names")
        try:
            names = tuple(self.dict_encoding)
        except TypeError:
            raise ValueError("dict_encoding must be an iterable of column names")
        for name in names:
            if not isinstance(name, str):
                raise ValueError("dict_encoding entries must be column names")
        object.__setattr__(self, "dict_encoding", names)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _column_stats(values: list, field: Field) -> dict[str, Any]:
    null_count = 0
    present = False
    minimum: Any = None
    maximum: Any = None
    if field.type == BOOL:
        for value in values:
            if value is None:
                null_count += 1
            elif not present:
                minimum = maximum = value
                present = True
            else:
                if value < minimum:
                    minimum = value
                if value > maximum:
                    maximum = value
    elif field.type == INT64:
        for value in values:
            if value is None:
                null_count += 1
            elif not present:
                minimum = maximum = value
                present = True
            else:
                if value < minimum:
                    minimum = value
                if value > maximum:
                    maximum = value
    elif field.type == FLOAT64:
        for value in values:
            if value is None:
                null_count += 1
                continue
            number = float(value)
            if not present:
                minimum = maximum = number
                present = True
            else:
                if number < minimum:
                    minimum = number
                if number > maximum:
                    maximum = number
    else:  # UTF8
        for value in values:
            if value is None:
                null_count += 1
            elif not present:
                minimum = maximum = value
                present = True
            else:
                if value < minimum:
                    minimum = value
                if value > maximum:
                    maximum = value
    return {
        "row_count": len(values),
        "null_count": null_count,
        "min": minimum if present else None,
        "max": maximum if present else None,
    }


# ---------------------------------------------------------------------------
# Payload encoding / decoding
# ---------------------------------------------------------------------------


def _encode_plain(values: list, field: Field) -> bytes:
    parts: list[bytes] = []
    pack = None
    if field.type == BOOL:
        pack = _D_BOOL.pack
    elif field.type == INT64:
        pack = _Q_INT.pack
    elif field.type == FLOAT64:
        pack = _D_FLOAT.pack

    if field.type == UTF8:
        for value in values:
            if value is None:
                parts.append(_U32.pack(0xFFFFFFFF))
            else:
                encoded = value.encode("utf-8")
                parts.append(_U32.pack(len(encoded)))
                parts.append(encoded)
        return b"".join(parts)

    assert pack is not None
    null_bitmap = bytearray((len(values) + 7) // 8)
    for i, value in enumerate(values):
        if value is not None:
            null_bitmap[i >> 3] |= 1 << (i & 7)

    body = bytearray()
    body.extend(_U32.pack(len(values)))
    body.extend(null_bitmap)
    for value in values:
        if value is not None:
            body.extend(pack(value))
        else:
            # Fixed-width placeholder so every row occupies one slot; the
            # null bitmap is authoritative for these bytes.
            body.extend(pack(False if field.type == BOOL else 0))
    return bytes(body)


def _decode_plain(payload: bytes, field: Field) -> list:
    stream = memoryview(payload)
    pos = 0

    if field.type == UTF8:
        result: list = []
        total = None
        while pos < len(stream):
            if pos + 4 > len(stream):
                raise ColumnarFormatError("truncated utf8 length prefix")
            (length,) = _U32.unpack(stream[pos : pos + 4])
            pos += 4
            if length == 0xFFFFFFFF:
                result.append(None)
                continue
            if pos + length > len(stream):
                raise ColumnarFormatError("truncated utf8 value bytes")
            try:
                result.append(bytes(stream[pos : pos + length]).decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ColumnarFormatError(f"invalid utf8 value: {exc}") from exc
            pos += length
        return result

    if pos + 4 > len(stream):
        raise ColumnarFormatError("truncated column payload")
    (count,) = _U32.unpack(stream[pos : pos + 4])
    pos += 4
    bitmap_len = (count + 7) // 8
    if pos + bitmap_len > len(stream):
        raise ColumnarFormatError("truncated null bitmap")
    null_bitmap = stream[pos : pos + bitmap_len]
    pos += bitmap_len

    if field.type == BOOL:
        width, unpack, caster = 1, _D_BOOL.unpack, bool
    elif field.type == INT64:
        width, unpack, caster = 8, _Q_INT.unpack, int
    else:
        width, unpack, caster = 8, _D_FLOAT.unpack, float

    expected_body = count * width
    if pos + expected_body != len(stream):
        raise ColumnarFormatError(
            f"column payload length mismatch for {field.name!r}"
        )

    result = []
    for i in range(count):
        is_null = not (null_bitmap[i >> 3] >> (i & 7)) & 1
        raw = bytes(stream[pos : pos + width])
        pos += width
        if is_null:
            result.append(None)
        else:
            result.append(caster(unpack(raw)[0]))
    return result


def _encode_dict(values: list) -> tuple[bytes, list[str]]:
    dictionary: list[str] = []
    indices: dict[str, int] = {}
    encoded_indices: list[int] = []
    for value in values:
        if value is None:
            encoded_indices.append(-1)
            continue
        index = indices.get(value)
        if index is None:
            index = len(dictionary)
            indices[value] = index
            dictionary.append(value)
        encoded_indices.append(index)

    parts = [_U32.pack(len(values))]
    for index in encoded_indices:
        parts.append(_I_DICT_INDEX.pack(index))
    return b"".join(parts), dictionary


def _decode_dict(payload: bytes, dictionary: list[str]) -> list:
    stream = memoryview(payload)
    if len(stream) < 4:
        raise ColumnarFormatError("truncated dictionary-encoded payload")
    (count,) = _U32.unpack(stream[:4])
    body = stream[4:]
    if len(body) != count * 4:
        raise ColumnarFormatError("dictionary payload length mismatch")
    result: list = []
    for i in range(count):
        (index,) = _I_DICT_INDEX.unpack(body[i * 4 : i * 4 + 4])
        if index == -1:
            result.append(None)
        elif 0 <= index < len(dictionary):
            result.append(dictionary[index])
        else:
            raise ColumnarFormatError(f"dictionary index out of range: {index}")
    return result


def _encode_dictionary(dictionary: list[str]) -> bytes:
    parts = [_U32.pack(len(dictionary))]
    for value in dictionary:
        encoded = value.encode("utf-8")
        parts.append(_U32.pack(len(encoded)))
        parts.append(encoded)
    return b"".join(parts)


def _decode_dictionary(payload: bytes) -> list[str]:
    stream = memoryview(payload)
    if len(stream) < 4:
        raise ColumnarFormatError("truncated dictionary")
    (count,) = _U32.unpack(stream[:4])
    pos = 4
    result: list[str] = []
    for _ in range(count):
        if pos + 4 > len(stream):
            raise ColumnarFormatError("truncated dictionary entry length")
        (length,) = _U32.unpack(stream[pos : pos + 4])
        pos += 4
        if pos + length > len(stream):
            raise ColumnarFormatError("truncated dictionary entry bytes")
        try:
            result.append(bytes(stream[pos : pos + length]).decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ColumnarFormatError(f"invalid utf8 dictionary value: {exc}") from exc
        pos += length
    if pos != len(stream):
        raise ColumnarFormatError("trailing bytes after dictionary")
    return result


# ---------------------------------------------------------------------------
# Low level framed IO
# ---------------------------------------------------------------------------


def _json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )


def _json_loads(raw: bytes, what: str) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ColumnarFormatError(f"invalid {what} JSON: {exc}") from exc


def _frame(payload: bytes, compression: str) -> bytes:
    codec = _CODEC_ZLIB if compression == "zlib" else _CODEC_RAW
    stored = zlib.compress(payload) if codec == _CODEC_ZLIB else payload
    return _U32.pack(len(payload)) + _U32.pack(len(stored)) + bytes([codec]) + stored


def _unframe(
    raw: bytes, pos: int, expected_compression: str | None, what: str
) -> tuple[bytes, int]:
    if pos + 9 > len(raw):
        raise ColumnarFormatError(f"truncated {what} frame header")
    (decompressed_len,) = _U32.unpack(raw[pos : pos + 4])
    (stored_len,) = _U32.unpack(raw[pos + 4 : pos + 8])
    codec = raw[pos + 8]
    pos += 9
    if codec not in _CODECS:
        raise ColumnarFormatError(f"unknown compression codec byte in {what}: {codec}")
    if pos + stored_len > len(raw):
        raise ColumnarFormatError(f"truncated {what} frame body")
    stored = raw[pos : pos + stored_len]
    pos += stored_len

    if codec == _CODEC_ZLIB:
        try:
            payload = zlib.decompress(stored)
        except zlib.error as exc:
            raise ColumnarFormatError(f"invalid compressed {what}: {exc}") from exc
    else:
        payload = stored

    if len(payload) != decompressed_len:
        raise ColumnarFormatError(f"{what} decompressed length mismatch")
    if expected_compression is not None:
        actually_compressed = "zlib" if codec == _CODEC_ZLIB else "none"
        if actually_compressed != expected_compression:
            raise ColumnarFormatError(
                f"{what} compression mismatch: header says {expected_compression!r}"
            )
    return payload, pos


def _checked_bool(value: Any, what: str) -> bool:
    if not isinstance(value, bool):
        raise ColumnarFormatError(f"illegal metadata: {what} must be a bool")
    return value


def _checked_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ColumnarFormatError(f"illegal metadata: {what} must be an int")
    return value


def _checked_str(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise ColumnarFormatError(f"illegal metadata: {what} must be a string")
    return value


# ---------------------------------------------------------------------------
# Header / footer validation
# ---------------------------------------------------------------------------


def _parse_header(header: Any) -> tuple[Schema, int, list[dict[str, Any]]]:
    if not isinstance(header, dict):
        raise ColumnarFormatError("illegal metadata: header must be an object")
    row_count = _checked_int(header.get("row_count"), "header.row_count")
    if row_count < 0:
        raise ColumnarFormatError("illegal metadata: negative row_count")

    raw_columns = header.get("columns")
    if not isinstance(raw_columns, list) or not raw_columns:
        raise ColumnarFormatError("illegal metadata: header.columns must be a non-empty list")

    fields: list[Field] = []
    seen: set[str] = set()
    column_meta: list[dict[str, Any]] = []
    for i, entry in enumerate(raw_columns):
        if not isinstance(entry, dict):
            raise ColumnarFormatError(f"illegal metadata: column {i} must be an object")
        name = _checked_str(entry.get("name"), f"column {i} name")
        if not name:
            raise ColumnarFormatError("illegal metadata: empty column name")
        if name in seen:
            raise ColumnarFormatError(f"illegal metadata: duplicate column {name!r}")
        seen.add(name)

        col_type = _checked_str(entry.get("type"), f"column {name!r} type")
        if col_type not in (BOOL, INT64, FLOAT64, UTF8):
            raise ColumnarFormatError(f"illegal metadata: unknown type {col_type!r}")
        nullable = _checked_bool(entry.get("nullable"), f"column {name!r} nullable")
        compression = _checked_str(
            entry.get("compression"), f"column {name!r} compression"
        )
        if compression not in COMPRESSIONS:
            raise ColumnarFormatError(
                f"illegal metadata: unknown compression {compression!r}"
            )
        encoding = _checked_str(entry.get("encoding"), f"column {name!r} encoding")
        if encoding not in ("plain", "dict"):
            raise ColumnarFormatError(
                f"illegal metadata: unknown encoding {encoding!r}"
            )
        if encoding == "dict" and col_type != UTF8:
            raise ColumnarFormatError(
                "illegal metadata: dictionary encoding requires utf8 type"
            )

        stats = entry.get("stats")
        if not isinstance(stats, dict):
            raise ColumnarFormatError(f"illegal metadata: stats for {name!r} missing")
        stats_row_count = _checked_int(stats.get("row_count"), f"stats.row_count of {name!r}")
        null_count = _checked_int(stats.get("null_count"), f"stats.null_count of {name!r}")
        if stats_row_count != row_count:
            raise ColumnarFormatError(
                f"illegal metadata: stats row_count mismatch for {name!r}"
            )
        if not (0 <= null_count <= row_count):
            raise ColumnarFormatError(
                f"illegal metadata: null_count out of range for {name!r}"
            )
        if not nullable and null_count:
            raise ColumnarFormatError(
                f"illegal metadata: non-nullable column {name!r} reports NULLs"
            )
        minimum = stats.get("min")
        maximum = stats.get("max")
        _check_extreme_type(minimum, col_type, nullable, name, "min")
        _check_extreme_type(maximum, col_type, nullable, name, "max")
        if (minimum is None) != (maximum is None):
            raise ColumnarFormatError(
                f"illegal metadata: min/max must both be null for {name!r}"
            )
        if minimum is not None and minimum > maximum:
            raise ColumnarFormatError(
                f"illegal metadata: min greater than max for {name!r}"
            )
        if col_type == FLOAT64:
            for boundary, label in ((minimum, "min"), (maximum, "max")):
                if boundary is not None and (math.isnan(boundary) or math.isinf(boundary)):
                    raise ColumnarFormatError(
                        f"illegal metadata: non-finite {label} for {name!r}"
                    )

        fields.append(Field(name, col_type, nullable))
        column_meta.append(
            {
                "name": name,
                "compression": compression,
                "encoding": encoding,
                "stats": stats,
            }
        )

    return Schema(fields), row_count, column_meta


def _check_extreme_type(
    value: Any, col_type: str, nullable: bool, name: str, label: str
) -> None:
    if value is None:
        return
    if col_type == BOOL:
        if not isinstance(value, bool):
            raise ColumnarFormatError(f"illegal metadata: {label} type for {name!r}")
    elif col_type == INT64:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ColumnarFormatError(f"illegal metadata: {label} type for {name!r}")
        if not (-(2**63) <= value <= 2**63 - 1):
            raise ColumnarFormatError(f"illegal metadata: {label} out of int64 range for {name!r}")
    elif col_type == FLOAT64:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ColumnarFormatError(f"illegal metadata: {label} type for {name!r}")
    else:
        if not isinstance(value, str):
            raise ColumnarFormatError(f"illegal metadata: {label} type for {name!r}")


def _parse_footer(footer: Any, schema: Schema) -> list[str]:
    if not isinstance(footer, dict):
        raise ColumnarFormatError("illegal metadata: footer must be an object")
    checksum_algorithm = footer.get("checksum_algorithm")
    if checksum_algorithm != _CHECKSUM_ALGORITHM:
        raise ColumnarFormatError(
            f"illegal metadata: unsupported checksum algorithm {checksum_algorithm!r}"
        )
    blocks = footer.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != len(schema):
        raise ColumnarFormatError("illegal metadata: footer.blocks must match columns")
    checksums: list[str] = []
    for i, entry in enumerate(blocks):
        if not isinstance(entry, dict):
            raise ColumnarFormatError(f"illegal metadata: footer block {i} must be an object")
        name = _checked_str(entry.get("name"), f"footer block {i} name")
        if name != schema.fields[i].name:
            raise ColumnarFormatError(
                f"illegal metadata: footer block order mismatch at {i}"
            )
        checksum = _checked_str(
            entry.get("checksum"), f"footer block {i} checksum"
        )
        try:
            bytes.fromhex(checksum)
        except ValueError as exc:
            raise ColumnarFormatError(
                f"illegal metadata: bad checksum for {name!r}"
            ) from exc
        if len(checksum) != 64:
            raise ColumnarFormatError(
                f"illegal metadata: bad checksum length for {name!r}"
            )
        checksums.append(checksum)
    return checksums


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_table(path: str | os.PathLike, table: Table, options: WriteOptions | None = None) -> None:
    """Write *table* to *path* atomically.

    Validation errors leave any existing file at *path* untouched.
    """
    if options is None:
        options = WriteOptions()
    if not isinstance(options, WriteOptions):
        raise ValueError("options must be a WriteOptions instance")
    if not isinstance(table, Table):
        raise ValueError("table must be a Table instance")

    schema = table.schema
    dict_names = set(options.dict_encoding)
    for name in options.dict_encoding:
        try:
            field = schema.field(name)
        except KeyError:
            raise ValueError(
                f"dictionary encoding specified for unknown column: {name!r}"
            ) from None
        if field.type != UTF8:
            raise ValueError(
                f"dictionary encoding requires a utf8 column: {name!r} is {field.type}"
            )

    header_columns: list[dict[str, Any]] = []
    encoded_blocks: list[tuple[str, str, bytes, bytes | None]] = []
    for field in schema.fields:
        values = table.column(field.name)
        stats = _column_stats(values, field)
        use_dict = field.name in dict_names
        if use_dict:
            payload, dictionary = _encode_dict(values)
            dictionary_payload = _encode_dictionary(dictionary)
            encoding = "dict"
        else:
            payload = _encode_plain(values, field)
            dictionary_payload = None
            encoding = "plain"
        framed = _frame(payload, options.compression)
        if dictionary_payload is not None:
            framed += _frame(dictionary_payload, "none")
        encoded_blocks.append((field.name, encoding, payload, framed))
        header_columns.append(
            {
                "name": field.name,
                "type": field.type,
                "nullable": field.nullable,
                "compression": options.compression,
                "encoding": encoding,
                "stats": stats,
            }
        )

    header_obj = {
        "format_version": FORMAT_VERSION,
        "row_count": table.row_count,
        "columns": header_columns,
    }
    header_payload = _json_dumps(header_obj)

    footer_blocks = [
        {
            "name": name,
            "checksum": hashlib.sha256(payload).hexdigest(),
        }
        for name, _encoding, payload, _framed in encoded_blocks
    ]
    footer_obj = {"checksum_algorithm": _CHECKSUM_ALGORITHM, "blocks": footer_blocks}
    footer_payload = _json_dumps(footer_obj)

    parts: list[bytes] = [
        MAGIC,
        _U16.pack(FORMAT_VERSION),
        _frame(header_payload, "zlib"),
    ]
    for _name, _encoding, _payload, framed in encoded_blocks:
        parts.append(framed)
    parts.append(_frame(footer_payload, "zlib"))

    body = b"".join(parts)
    checksum = hashlib.sha256(body).digest()
    assert len(checksum) == _CHECKSUM_BYTES
    blob = body + checksum

    _atomic_replace(path, blob)


def _atomic_replace(path: str | os.PathLike, blob: bytes) -> None:
    target = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(target))
    fd, tmp_name = tempfile.mkstemp(prefix=".columnar-", dir=directory)
    try:
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            os.unlink(tmp_name)
            raise
        os.replace(tmp_name, target)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _read_bytes(path: str | os.PathLike) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _parse_file(raw: bytes, *, decode_values: bool = True) -> tuple[Schema, int, list[dict[str, Any]], list[list] | None]:
    if len(raw) < 4 + 2:
        raise ColumnarFormatError("file is truncated")
    if raw[:4] != MAGIC:
        raise ColumnarFormatError("bad magic: not a columnar analytics file")
    pos = 4
    (version,) = _U16.unpack(raw[pos : pos + 2])
    pos += 2
    if version != FORMAT_VERSION:
        raise ColumnarFormatError(f"unsupported format version: {version}")

    header_payload, pos = _unframe(raw, pos, "zlib", "header")
    header = _json_loads(header_payload, "header")
    if isinstance(header, dict) and header.get("format_version") != FORMAT_VERSION:
        raise ColumnarFormatError(
            f"unsupported format version in header: {header.get('format_version')!r}"
        )
    schema, row_count, column_meta = _parse_header(header)

    block_payloads: list[bytes] = []
    for meta in column_meta:
        payload, pos = _unframe(raw, pos, meta["compression"], f"block {meta['name']!r}")
        block_payloads.append(payload)
        if meta["encoding"] == "dict":
            dictionary_payload, pos = _unframe(
                raw, pos, "none", f"dictionary {meta['name']!r}"
            )
            meta["dictionary_payload"] = dictionary_payload

    footer_payload, pos = _unframe(raw, pos, "zlib", "footer")
    footer = _json_loads(footer_payload, "footer")
    checksums = _parse_footer(footer, schema)

    if pos + _CHECKSUM_BYTES != len(raw):
        raise ColumnarFormatError("trailing bytes after file checksum")
    stored_checksum = raw[pos : pos + _CHECKSUM_BYTES]
    actual_checksum = hashlib.sha256(raw[:pos]).digest()
    if stored_checksum != actual_checksum:
        raise ColumnarFormatError("file checksum mismatch")

    columns: list[list] | None = [] if decode_values else None
    for meta, payload, expected_checksum in zip(column_meta, block_payloads, checksums):
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected_checksum:
            raise ColumnarFormatError(
                f"column block checksum mismatch: {meta['name']!r}"
            )
        if not decode_values:
            continue
        field = schema.field(meta["name"])
        if meta["encoding"] == "dict":
            dictionary = _decode_dictionary(meta["dictionary_payload"])
            values = _decode_dict(payload, dictionary)
        else:
            values = _decode_plain(payload, field)
        if len(values) != row_count:
            raise ColumnarFormatError(
                f"decoded row count mismatch for {meta['name']!r}"
            )
        _verify_values(values, field)
        _verify_stats(values, field, meta["stats"])
        columns.append(values)  # type: ignore[union-attr]

    return schema, row_count, column_meta, columns


def _verify_values(values: list, field: Field) -> None:
    for value in values:
        if value is None:
            if not field.nullable:
                raise ColumnarFormatError(
                    f"NULL in non-nullable column {field.name!r}"
                )
            continue
        if field.type == BOOL:
            if not isinstance(value, bool):
                raise ColumnarFormatError(f"decoded type mismatch in {field.name!r}")
        elif field.type == INT64:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ColumnarFormatError(f"decoded type mismatch in {field.name!r}")
        elif field.type == FLOAT64:
            if not isinstance(value, float) or math.isnan(value) or math.isinf(value):
                raise ColumnarFormatError(f"decoded illegal float in {field.name!r}")
        elif not isinstance(value, str):
            raise ColumnarFormatError(f"decoded type mismatch in {field.name!r}")


def _verify_stats(values: list, field: Field, stats: dict[str, Any]) -> None:
    computed = _column_stats(values, field)
    if computed["row_count"] != stats["row_count"]:
        raise ColumnarFormatError(f"stats row_count mismatch for {field.name!r}")
    if computed["null_count"] != stats["null_count"]:
        raise ColumnarFormatError(f"stats null_count mismatch for {field.name!r}")
    if computed["min"] != stats["min"] or computed["max"] != stats["max"]:
        raise ColumnarFormatError(f"stats min/max mismatch for {field.name!r}")


def read_table(path: str | os.PathLike) -> Table:
    """Read a table previously written with :func:`write_table`."""
    raw = _read_bytes(path)
    schema, _row_count, _meta, columns = _parse_file(raw, decode_values=True)
    data = {field.name: values for field, values in zip(schema.fields, columns)}
    return Table(schema, data)


def inspect_file(path: str | os.PathLike) -> dict[str, Any]:
    """Return format version, row count and per-column metadata without column data."""
    raw = _read_bytes(path)
    schema, row_count, column_meta, _columns = _parse_file(raw, decode_values=False)
    return {
        "format_version": FORMAT_VERSION,
        "row_count": row_count,
        "columns": [
            {
                "name": field.name,
                "type": field.type,
                "nullable": field.nullable,
                "compression": meta["compression"],
                "encoding": meta["encoding"],
                "row_count": meta["stats"]["row_count"],
                "null_count": meta["stats"]["null_count"],
                "min": meta["stats"]["min"],
                "max": meta["stats"]["max"],
            }
            for field, meta in zip(schema.fields, column_meta)
        ],
    }
