"""Tests for the columnar file layer."""

from __future__ import annotations

import io
import json
import os
import struct
import subprocess
import sys
import zlib

import pytest

import columnar_analytics
from columnar_analytics import (
    FORMAT_VERSION,
    ColumnSchema,
    ColumnarFormatError,
    Schema,
    Table,
    inspect_file,
    read_file,
    write_file,
)
from columnar_analytics.cli import main

SCHEMA = Schema(
    [
        ColumnSchema("flag", "bool"),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("f", "float64"),
        ColumnSchema("s", "utf8", nullable=True),
        ColumnSchema("all_null", "int64", nullable=True),
    ]
)

ROWS = {
    "flag": [True, False, True, False],
    "n": [1, None, -2**63, 2**63 - 1],
    "f": [1.5, -0.0, 3.25, 1e300],
    "s": ["b", "a", None, "héllo→世界"],
    "all_null": [None, None, None, None],
}


def make_table():
    return Table(SCHEMA, ROWS)


@pytest.mark.parametrize("compression", ["none", "zlib"])
@pytest.mark.parametrize("dictionary", [[], ["s"]])
def test_roundtrip(tmp_path, compression, dictionary):
    path = tmp_path / "t.caef"
    write_file(path, make_table(), compression=compression, dictionary_encoding=dictionary)
    table = read_file(path)
    assert table.schema == SCHEMA
    assert table.row_count == 4
    assert table.column_names == SCHEMA.names
    for name in SCHEMA.names:
        assert table.column(name) == ROWS[name]


def test_columns_mapping_preserves_order(tmp_path):
    path = tmp_path / "t.caef"
    shuffled = {name: ROWS[name] for name in reversed(SCHEMA.names)}
    write_file(path, Table(SCHEMA, shuffled))
    assert read_file(path).column_names == SCHEMA.names


def test_columns_as_sequence(tmp_path):
    path = tmp_path / "t.caef"
    seq = [ROWS[name] for name in SCHEMA.names]
    write_file(path, Table(SCHEMA, seq))
    assert read_file(path).column("s") == ROWS["s"]


def test_empty_table_roundtrip(tmp_path):
    schema = Schema([ColumnSchema("a", "int64"), ColumnSchema("b", "utf8")])
    table = Table(schema, {"a": [], "b": []})
    path = tmp_path / "empty.caef"
    write_file(path, table)
    restored = read_file(path)
    assert restored.row_count == 0
    assert restored.column("a") == []
    assert restored.column("b") == []


def test_projection_reorders_and_subsets(tmp_path):
    path = tmp_path / "t.caef"
    write_file(path, make_table())
    table = read_file(path, columns=["s", "flag"])
    assert table.column_names == ("s", "flag")
    assert table.column("s") == ROWS["s"]
    assert table.column("flag") == ROWS["flag"]

    projected = make_table().project(["n"])
    assert projected.column_names == ("n",)
    assert projected.row_count == 4


def test_projection_unknown_column(tmp_path):
    path = tmp_path / "t.caef"
    write_file(path, make_table())
    with pytest.raises(KeyError):
        read_file(path, columns=["nope"])
    with pytest.raises(KeyError):
        make_table().project(["nope"])


def test_projection_duplicate_column(tmp_path):
    path = tmp_path / "t.caef"
    write_file(path, make_table())
    with pytest.raises(ValueError):
        read_file(path, columns=["n", "n"])
    with pytest.raises(ValueError):
        make_table().project(["n", "n"])


# ---------------------------------------------------------------------------
# Schema / value validation
# ---------------------------------------------------------------------------


def test_invalid_schema():
    with pytest.raises(ValueError):
        ColumnSchema("", "int64")
    with pytest.raises(ValueError):
        ColumnSchema("x", "decimal")
    with pytest.raises(ValueError):
        Schema([])
    with pytest.raises(ValueError):
        Schema([ColumnSchema("a", "int64"), ColumnSchema("a", "float64")])


def test_missing_column():
    with pytest.raises(ValueError):
        Table(SCHEMA, {name: ROWS[name] for name in SCHEMA.names if name != "n"})


def test_extra_column():
    data = dict(ROWS)
    data["extra"] = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        Table(SCHEMA, data)


def test_wrong_column_count_sequence():
    with pytest.raises(ValueError):
        Table(SCHEMA, [[1], [2]])


def test_length_mismatch():
    data = dict(ROWS)
    data["n"] = [1, None]
    with pytest.raises(ValueError):
        Table(SCHEMA, data)


@pytest.mark.parametrize(
    "col_type,bad_value",
    [
        ("bool", "true"),
        ("bool", 1),
        ("int64", 1.0),
        ("int64", True),
        ("float64", 1),
        ("float64", "1.0"),
        ("utf8", b"x"),
        ("utf8", 1),
    ],
)
def test_type_mismatch(col_type, bad_value):
    schema = Schema([ColumnSchema("c", col_type)])
    with pytest.raises(ValueError):
        Table(schema, {"c": [bad_value]})


def test_int64_range():
    schema = Schema([ColumnSchema("c", "int64")])
    with pytest.raises(ValueError):
        Table(schema, {"c": [2**63]})
    with pytest.raises(ValueError):
        Table(schema, {"c": [-(2**63) - 1]})


def test_null_in_non_nullable():
    with pytest.raises(ValueError):
        Table(Schema([ColumnSchema("c", "utf8")]), {"c": [None]})
    with pytest.raises(ValueError):
        Table(Schema([ColumnSchema("c", "bool")]), {"c": [None]})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_rejected(bad):
    schema = Schema([ColumnSchema("c", "float64")])
    with pytest.raises(ValueError):
        Table(schema, {"c": [bad]})


def test_unknown_compression(tmp_path):
    with pytest.raises(ValueError):
        write_file(tmp_path / "x", make_table(), compression="gzip")


@pytest.mark.parametrize("target", ["flag", "n", "f"])
def test_dictionary_requires_utf8(tmp_path, target):
    with pytest.raises(ValueError):
        write_file(
            tmp_path / "x",
            make_table(),
            dictionary_encoding=[target],
        )


def test_dictionary_unknown_column(tmp_path):
    with pytest.raises(ValueError):
        write_file(tmp_path / "x", make_table(), dictionary_encoding=["nope"])


def test_dictionary_duplicate_entry(tmp_path):
    with pytest.raises(ValueError):
        write_file(tmp_path / "x", make_table(), dictionary_encoding=["s", "s"])


def test_invalid_arguments_leave_no_file(tmp_path):
    target = tmp_path / "nope.caef"
    with pytest.raises(ValueError):
        write_file(target, make_table(), compression="gzip")
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_failed_write_preserves_existing(tmp_path):
    path = tmp_path / "keep.caef"
    write_file(path, make_table())
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_file(path, make_table(), compression="bogus")
    assert path.read_bytes() == original


# ---------------------------------------------------------------------------
# Determinism, compression and dictionary encoding
# ---------------------------------------------------------------------------


def test_deterministic_bytes(tmp_path):
    p1, p2 = tmp_path / "a", tmp_path / "b"
    for opts in (
        {"compression": "none"},
        {"compression": "zlib"},
        {"compression": "none", "dictionary_encoding": ["s"]},
        {"compression": "zlib", "dictionary_encoding": ["s"]},
    ):
        write_file(p1, make_table(), **opts)
        write_file(p2, make_table(), **opts)
        assert p1.read_bytes() == p2.read_bytes()


def test_dictionary_ids_follow_first_occurrence(tmp_path):
    schema = Schema([ColumnSchema("s", "utf8")])
    table = Table(schema, {"s": ["z", "a", "z", "", "a", ""]})
    path = tmp_path / "d.caef"
    write_file(path, table, dictionary_encoding=["s"])
    blob = path.read_bytes()
    # uint32 dict size 3, then offsets+blob listing values in first-occurrence
    # order: z, a, "".
    assert (
        struct.pack("<I", 3)
        + struct.pack("<IIII", 0, 1, 2, 2)
        + b"za"
    ) in blob
    restored = read_file(path)
    assert restored.column("s") == ["z", "a", "z", "", "a", ""]


def test_zlib_and_none_contain_same_data(tmp_path):
    p_none = tmp_path / "n"
    p_zlib = tmp_path / "z"
    write_file(p_none, make_table())
    write_file(p_zlib, make_table(), compression="zlib")
    none_blob = p_none.read_bytes()
    zlib_blob = p_zlib.read_bytes()
    assert none_blob != zlib_blob  # at least the wrapper differs
    assert read_file(p_zlib).columns == read_file(p_none).columns


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_inspect_metadata(tmp_path):
    path = tmp_path / "t.caef"
    write_file(path, make_table(), compression="zlib", dictionary_encoding=["s"])
    meta = inspect_file(path)
    assert list(meta) == ["format_version", "row_count", "columns"]
    assert meta["format_version"] == FORMAT_VERSION
    assert meta["row_count"] == 4
    names = [c["name"] for c in meta["columns"]]
    assert names == ["flag", "n", "f", "s", "all_null"]
    by_name = {c["name"]: c for c in meta["columns"]}
    assert by_name["flag"] == {
        "name": "flag",
        "type": "bool",
        "nullable": False,
        "row_count": 4,
        "null_count": 0,
        "min": False,
        "max": True,
    }
    assert by_name["n"]["null_count"] == 1
    assert by_name["n"]["min"] == -(2**63)
    assert by_name["n"]["max"] == 2**63 - 1
    assert by_name["all_null"]["null_count"] == 4
    assert by_name["all_null"]["min"] is None
    assert by_name["all_null"]["max"] is None
    assert by_name["s"]["min"] == "a"
    assert by_name["s"]["max"] == "héllo→世界"


def test_inspect_does_not_touch_data_section(tmp_path, monkeypatch):
    path = tmp_path / "t.caef"
    write_file(path, make_table(), compression="zlib")
    blob = bytearray(path.read_bytes())
    # Corrupt the last byte of the compressed data section (before the 8-byte footer).
    blob[-9] ^= 0xFF
    path.write_bytes(blob)
    # inspect must succeed (header + size only); full read must fail on the CRC.
    meta = inspect_file(path)
    assert meta["row_count"] == 4
    with pytest.raises(ColumnarFormatError):
        read_file(path)


# ---------------------------------------------------------------------------
# Corruption handling
# ---------------------------------------------------------------------------


@pytest.fixture()
def written_path(tmp_path):
    path = tmp_path / "t.caef"
    write_file(path, make_table(), compression="zlib", dictionary_encoding=["s"])
    return path


def _corrupt(path, offset, new_byte):
    blob = bytearray(path.read_bytes())
    blob[offset] = new_byte
    path.write_bytes(blob)


def test_bad_magic(written_path):
    _corrupt(written_path, 0, ord("X"))
    with pytest.raises(ColumnarFormatError):
        read_file(written_path)
    with pytest.raises(ColumnarFormatError):
        inspect_file(written_path)


def test_unknown_version(written_path):
    _corrupt(written_path, 4, FORMAT_VERSION + 99)
    with pytest.raises(ColumnarFormatError):
        read_file(written_path)


def test_truncation(written_path):
    blob = written_path.read_bytes()
    for cut in (0, 4, 9, len(blob) // 2, len(blob) - 1, len(blob) - 8):
        written_path.write_bytes(blob[:cut])
        with pytest.raises(ColumnarFormatError):
            read_file(written_path)
        with pytest.raises(ColumnarFormatError):
            inspect_file(written_path)


def test_extra_trailing_bytes(written_path):
    written_path.write_bytes(written_path.read_bytes() + b"\x00")
    with pytest.raises(ColumnarFormatError):
        read_file(written_path)
    with pytest.raises(ColumnarFormatError):
        inspect_file(written_path)


def test_footer_crc_corruption(written_path):
    blob = bytearray(written_path.read_bytes())
    # Flip a byte inside the compressed data section.
    blob[40] ^= 0x01
    written_path.write_bytes(blob)
    with pytest.raises(ColumnarFormatError):
        read_file(written_path)


def test_footer_crc_field_corruption(written_path):
    blob = bytearray(written_path.read_bytes())
    blob[-8] ^= 0x01
    written_path.write_bytes(blob)
    with pytest.raises(ColumnarFormatError):
        read_file(written_path)


def test_bad_end_marker(written_path):
    blob = bytearray(written_path.read_bytes())
    blob[-1] ^= 0x01
    written_path.write_bytes(blob)
    with pytest.raises(ColumnarFormatError):
        read_file(written_path)


def test_header_corruption(written_path):
    blob = bytearray(written_path.read_bytes())
    header_len = struct.unpack("<I", blob[5:9])[0]
    blob[10 + header_len // 2] ^= 0x01
    written_path.write_bytes(blob)
    with pytest.raises(ColumnarFormatError):
        read_file(written_path)


def test_tampered_header_unknown_compression(written_path):
    blob = bytearray(written_path.read_bytes())
    header_len = struct.unpack("<I", blob[5:9])[0]
    header = json.loads(bytes(blob[9 : 9 + header_len]).decode())
    header["compression"] = "gzip"
    new_header = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    new_blob = blob[:5] + struct.pack("<I", len(new_header)) + new_header + blob[9 + header_len :]
    written_path.write_bytes(new_blob)
    with pytest.raises(ColumnarFormatError):
        inspect_file(written_path)


def test_os_error_passthrough(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        read_file(missing)
    with pytest.raises(FileNotFoundError):
        inspect_file(missing)


def _rebuild_with_column_payload(path, col_name, mutate):
    """Rewrite a file with one column's payload mutated and all CRCs fixed."""
    blob = bytearray(path.read_bytes())
    header_len = struct.unpack("<I", blob[5:9])[0]
    header = json.loads(bytes(blob[9 : 9 + header_len]).decode())
    assert header["compression"] == "none"
    entries = {c["name"]: c for c in header["columns"]}
    data_start = 9 + header_len
    entry = entries[col_name]
    start = data_start + entry["offset"]
    end = start + entry["length"]
    payload = bytearray(bytes(blob[start:end]))
    mutate(payload)
    blob[start:end] = payload
    entry["crc32"] = zlib.crc32(payload) & 0xFFFFFFFF
    raw = bytes(blob[data_start:-8])
    header["data_crc32"] = zlib.crc32(raw) & 0xFFFFFFFF
    new_header = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    prefix = blob[:5] + struct.pack("<I", len(new_header))
    covered = prefix + new_header + raw
    path.write_bytes(
        covered + struct.pack("<I", zlib.crc32(covered) & 0xFFFFFFFF) + b"END1"
    )


def test_validity_index_contradiction_rejected(tmp_path):
    path = tmp_path / "d.caef"
    schema = Schema([ColumnSchema("s", "utf8", nullable=True)])
    Table(schema, {"s": ["a", "b"]})  # smoke: construction works
    write_file(path, Table(schema, {"s": ["a", "b"]}), dictionary_encoding=["s"])

    def flip_validity_to_null(payload):
        # payload: validity bitmap (1 byte) then dict_size/offsets/blob/indices
        payload[0] &= 0b11111110  # mark row 0 NULL

    _rebuild_with_column_payload(path, "s", flip_validity_to_null)
    with pytest.raises(ColumnarFormatError):
        read_file(path)
    # inspect_file never touches payload bytes, so it stays metadata-only.
    assert inspect_file(path)["row_count"] == 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_version(capsys):
    assert main(["version"]) == 0
    out = capsys.readouterr().out
    assert out.strip() == columnar_analytics.__version__ == "0.1.0"


def test_cli_no_command_prints_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "usage:" in out
    assert "version" in out
    assert "inspect" in out


def test_cli_inspect_json(tmp_path, capsys):
    path = tmp_path / "t.caef"
    write_file(path, make_table())
    assert main(["inspect", str(path)]) == 0
    out = capsys.readouterr().out
    meta = json.loads(out)
    assert list(meta) == ["format_version", "row_count", "columns"]
    assert meta["columns"][0]["name"] == "flag"


def test_cli_inspect_format_error(tmp_path, capsys):
    path = tmp_path / "bad"
    path.write_bytes(b"not a columnar file at all")
    code = main(["inspect", str(path)])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_inspect_missing_path(tmp_path, capsys):
    code = main(["inspect", str(tmp_path / "nope")])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_as_module_subprocess(tmp_path):
    path = tmp_path / "t.caef"
    write_file(path, make_table())
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "columnar_analytics.cli", "version"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "0.1.0"
