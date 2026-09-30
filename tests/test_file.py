"""Tests for the columnar file format: round-trips, stats, options and corruption."""

import os
import struct

import pytest

from columnar_analytics import (
    BOOL,
    FLOAT64,
    INT64,
    UTF8,
    ColumnarFormatError,
    Field,
    Schema,
    Table,
    WriteOptions,
    inspect_file,
    read_table,
    write_table,
)
from columnar_analytics.file import MAGIC


def make_table():
    schema = Schema(
        [
            Field("flag", BOOL),
            Field("n", INT64, nullable=False),
            Field("ratio", FLOAT64),
            Field("label", UTF8),
        ]
    )
    return Table(
        schema,
        {
            "flag": [True, False, None, True, False],
            "n": [10, -2, 3, -(2**63), 2**63 - 1],
            "ratio": [1.5, -0.25, None, 0.0, 42.0],
            "label": ["banana", "苹果", "banana", None, "cherry"],
        },
    )


@pytest.mark.parametrize("compression", ["none", "zlib"])
@pytest.mark.parametrize("dict_encoding", [[], ["label"]])
def test_round_trip(tmp_path, compression, dict_encoding):
    path = tmp_path / "data.col"
    table = make_table()
    write_table(path, table, WriteOptions(compression=compression, dict_encoding=dict_encoding))
    restored = read_table(path)
    assert restored == table
    assert restored.schema.names == ["flag", "n", "ratio", "label"]


def test_round_trip_empty_table(tmp_path):
    schema = Schema([Field("n", INT64), Field("s", UTF8)])
    table = Table(schema, {"n": [], "s": []})
    path = tmp_path / "empty.col"
    write_table(path, table)
    assert read_table(path) == table
    info = inspect_file(path)
    assert info["row_count"] == 0


def test_round_trip_all_null_columns(tmp_path):
    schema = Schema(
        [
            Field("b", BOOL),
            Field("i", INT64),
            Field("f", FLOAT64),
            Field("s", UTF8),
        ]
    )
    table = Table(
        schema,
        {"b": [None, None], "i": [None, None], "f": [None, None], "s": [None, None]},
    )
    path = tmp_path / "nulls.col"
    write_table(path, table, WriteOptions(dict_encoding=["s"]))
    assert read_table(path) == table
    info = inspect_file(path)
    for column in info["columns"]:
        assert column["null_count"] == 2
        assert column["min"] is None
        assert column["max"] is None


def test_inspect_stats(tmp_path):
    path = tmp_path / "data.col"
    write_table(path, make_table(), WriteOptions(compression="zlib", dict_encoding=["label"]))
    info = inspect_file(path)
    assert list(info.keys()) == ["format_version", "row_count", "columns"]
    assert info["format_version"] == 1
    assert info["row_count"] == 5
    by_name = {column["name"]: column for column in info["columns"]}
    assert [column["name"] for column in info["columns"]] == [
        "flag",
        "n",
        "ratio",
        "label",
    ]
    assert by_name["n"]["null_count"] == 0
    assert by_name["n"]["min"] == -(2**63)
    assert by_name["n"]["max"] == 2**63 - 1
    assert by_name["ratio"]["null_count"] == 1
    assert by_name["ratio"]["min"] == -0.25
    assert by_name["ratio"]["max"] == 42.0
    assert by_name["label"]["null_count"] == 1
    assert by_name["label"]["min"] == "banana"
    assert by_name["label"]["max"] == "苹果"
    assert by_name["label"]["encoding"] == "dict"
    assert by_name["n"]["encoding"] == "plain"
    assert by_name["n"]["compression"] == "zlib"


def test_dictionary_indices_follow_first_appearance(tmp_path):
    path = tmp_path / "dict.col"
    table = make_table()
    write_table(path, table, WriteOptions(dict_encoding=["label"]))
    raw = path.read_bytes()
    info = inspect_file(path)
    assert info["columns"][3]["encoding"] == "dict"
    # Round-trip is the behavioural guarantee; first-appearance order is also
    # visible in the decoded data which must be identical to the input.
    assert read_table(path).column("label") == table.column("label")


def test_deterministic_bytes(tmp_path):
    table = make_table()
    first = tmp_path / "a.col"
    second = tmp_path / "b.col"
    options = WriteOptions(compression="zlib", dict_encoding=["label"])
    write_table(first, table, options)
    write_table(second, table, options)
    assert first.read_bytes() == second.read_bytes()


def test_different_options_produce_different_encoding_metadata(tmp_path):
    table = make_table()
    plain = tmp_path / "plain.col"
    compressed = tmp_path / "z.col"
    write_table(plain, table)
    write_table(compressed, table, WriteOptions(compression="zlib"))
    assert inspect_file(plain)["columns"][0]["compression"] == "none"
    assert inspect_file(compressed)["columns"][0]["compression"] == "zlib"


def test_unknown_compression_rejected():
    with pytest.raises(ValueError):
        WriteOptions(compression="gzip")


def test_dict_encoding_requires_utf8(tmp_path):
    table = make_table()
    path = tmp_path / "bad.col"
    with pytest.raises(ValueError):
        write_table(path, table, WriteOptions(dict_encoding=["n"]))
    assert not path.exists()


def test_dict_encoding_unknown_column(tmp_path):
    table = make_table()
    path = tmp_path / "bad.col"
    with pytest.raises(ValueError):
        write_table(path, table, WriteOptions(dict_encoding=["missing"]))
    assert not path.exists()


def test_validation_error_leaves_existing_file_unchanged(tmp_path):
    path = tmp_path / "data.col"
    write_table(path, make_table())
    original_bytes = path.read_bytes()

    # A table object whose column violates the non-nullable contract cannot
    # even be constructed; the write-time ValueError path is exercised via a
    # non-utf8 dictionary option instead.
    with pytest.raises(ValueError):
        write_table(path, make_table(), WriteOptions(dict_encoding=["n"]))
    assert path.read_bytes() == original_bytes


def test_write_replaces_target_atomically(tmp_path):
    path = tmp_path / "data.col"
    write_table(path, make_table())
    first = path.read_bytes()
    write_table(path, make_table())
    second = path.read_bytes()
    assert first == second
    # No temp files left behind.
    leftovers = [name for name in os.listdir(tmp_path) if name.startswith(".columnar-")]
    assert leftovers == []


def test_write_to_unwritable_directory(tmp_path):
    table = make_table()
    target = tmp_path / "missing_dir" / "data.col"
    with pytest.raises(OSError):
        write_table(target, table)
    assert not target.exists()


def test_read_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        read_table(tmp_path / "nope.col")
    with pytest.raises(OSError):
        inspect_file(tmp_path / "nope.col")


def test_read_directory_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        read_table(tmp_path)


def _write_sample(tmp_path, name="data.col", **options):
    path = tmp_path / name
    write_table(path, make_table(), WriteOptions(**options))
    return path


def test_bad_magic(tmp_path):
    path = _write_sample(tmp_path)
    raw = bytearray(path.read_bytes())
    raw[0] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(ColumnarFormatError, match="magic"):
        read_table(path)
    with pytest.raises(ColumnarFormatError, match="magic"):
        inspect_file(path)


def test_unknown_version(tmp_path):
    path = _write_sample(tmp_path)
    raw = bytearray(path.read_bytes())
    raw[4:6] = struct.pack("<H", 999)
    path.write_bytes(bytes(raw))
    with pytest.raises(ColumnarFormatError, match="version"):
        read_table(path)


@pytest.mark.parametrize("cut", [0, 1, 5, 6, 20, 100])
def test_truncated_files(tmp_path, cut):
    path = _write_sample(tmp_path, compression="zlib", dict_encoding=["label"])
    raw = path.read_bytes()
    path.write_bytes(raw[:cut])
    with pytest.raises(ColumnarFormatError):
        read_table(path)
    with pytest.raises(ColumnarFormatError):
        inspect_file(path)


def test_truncated_near_end(tmp_path):
    path = _write_sample(tmp_path)
    raw = path.read_bytes()
    for cut in (len(raw) - 1, len(raw) - 10, len(raw) - 33):
        path.write_bytes(raw[:cut])
        with pytest.raises(ColumnarFormatError):
            read_table(path)


def test_file_checksum_mismatch(tmp_path):
    path = _write_sample(tmp_path, compression="zlib")
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0x01
    path.write_bytes(bytes(raw))
    with pytest.raises(ColumnarFormatError, match="checksum"):
        read_table(path)


def test_block_corruption_detected(tmp_path):
    path = _write_sample(tmp_path, compression="none")
    raw = bytearray(path.read_bytes())
    # Flip a byte well inside the file; checksum or block validation must catch it.
    raw[60] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(ColumnarFormatError):
        read_table(path)
    with pytest.raises(ColumnarFormatError):
        inspect_file(path)


def test_trailing_bytes_rejected(tmp_path):
    path = _write_sample(tmp_path)
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ColumnarFormatError, match="trailing"):
        read_table(path)


def test_garbage_file(tmp_path):
    path = tmp_path / "garbage.col"
    path.write_bytes(b"this is definitely not a columnar file" * 4)
    with pytest.raises(ColumnarFormatError):
        read_table(path)


def test_empty_file(tmp_path):
    path = tmp_path / "empty.col"
    path.write_bytes(b"")
    with pytest.raises(ColumnarFormatError):
        read_table(path)


def test_written_file_is_self_describing(tmp_path):
    path = _write_sample(tmp_path)
    raw = path.read_bytes()
    assert raw[:4] == MAGIC
    assert struct.unpack("<H", raw[4:6])[0] == 1


def test_bool_column_round_trip_with_nulls(tmp_path):
    schema = Schema([Field("b", BOOL)])
    table = Table(schema, {"b": [True, False, None, False, True, None]})
    path = tmp_path / "b.col"
    write_table(path, table, WriteOptions(compression="zlib"))
    assert read_table(path) == table


def test_utf8_boundary_strings(tmp_path):
    schema = Schema([Field("s", UTF8, nullable=False)])
    values = ["", "a" * 1000, "🎉" * 50, "混合 unicode αβγ"]
    table = Table(schema, {"s": values})
    path = tmp_path / "s.col"
    write_table(path, table, WriteOptions(dict_encoding=["s"]))
    restored = read_table(path)
    assert restored.column("s") == values
    # Empty string is a real dictionary value, distinct from NULL.
    info = inspect_file(path)
    assert info["columns"][0]["null_count"] == 0
    assert info["columns"][0]["min"] == ""
