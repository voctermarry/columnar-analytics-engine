"""Tests for the query-result export entry points and the export CLI commands."""

from __future__ import annotations

import json

import pytest

from columnar_analytics import (
    ColumnSchema,
    ColumnarFormatError,
    QuerySyntaxError,
    QueryValidationError,
    Schema,
    Table,
    export_query_file,
    export_query_files,
    write_file,
)
from columnar_analytics.cli import main


SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("f", "float64", nullable=True),
        ColumnSchema("s", "utf8", nullable=True),
        ColumnSchema("flag", "bool"),
        ColumnSchema("to the, utf8", "utf8", nullable=True),
    ]
)

DATA = {
    "id": [1, 2, 3, 4],
    "n": [10, None, 30, 40],
    "f": [1.5, 2.0, None, -0.25],
    "s": ["a", "", None, "x,y"],
    "flag": [True, False, True, False],
    "to the, utf8": ["い", "ろ", None, "は"],
}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "t.caef"
    write_file(p, Table(SCHEMA, DATA))
    return p


@pytest.fixture()
def target(tmp_path):
    return tmp_path / "out.csv"


# ---------------------------------------------------------------------------
# CSV rendering
# ---------------------------------------------------------------------------


def test_csv_basic_layout(path, target):
    rows = export_query_file(path, "SELECT * FROM input ORDER BY id", target)
    assert rows == 4
    blob = target.read_bytes()
    assert not blob.startswith(b"\xef\xbb\xbf")  # no BOM
    assert b"\r" not in blob
    assert blob.decode("utf-8") == (
        'id,n,f,s,flag,"to the, utf8"\n'
        "1,10,1.5,a,true,い\n"
        '2,,2.0,"",false,ろ\n'
        "3,30,,,true,\n"
        '4,40,-0.25,"x,y",false,は\n'
    )


def test_csv_quoting_of_special_characters(tmp_path):
    p = tmp_path / "t.caef"
    write_file(
        p,
        Table(
            Schema([ColumnSchema("s", "utf8", nullable=True)]),
            {"s": ['a"b', "x\r\ny", "z\nw", "q\rr", "plain"]},
        ),
    )
    target = tmp_path / "out.csv"
    export_query_file(p, "SELECT s FROM input", target)
    blob = target.read_bytes()
    assert blob.decode("utf-8") == 's\n"a""b"\n"x\r\ny"\n"z\nw"\n"q\rr"\nplain\n'
    # Six record-separator LFs plus the two LFs embedded in quoted fields.
    assert blob.count(b"\n") == 8


def test_csv_null_and_empty_string_are_distinct(path, target):
    export_query_file(path, "SELECT s FROM input ORDER BY id", target)
    lines = target.read_text("utf-8").split("\n")
    assert lines[1] == "a"
    assert lines[2] == '""'  # empty string is a quoted empty field
    assert lines[3] == ""  # NULL is an empty unquoted field


def test_csv_projection_order_and_aggregates(path, target):
    rows = export_query_file(
        path,
        "SELECT flag, COUNT(*), AVG(f) FROM input GROUP BY flag ORDER BY flag",
        target,
    )
    assert rows == 2
    assert target.read_text("utf-8") == "flag,COUNT(*),AVG(f)\nfalse,2,0.875\ntrue,2,1.5\n"


def test_csv_zero_rows_writes_header_only(path, target):
    rows = export_query_file(path, "SELECT id, s FROM input WHERE id > 100", target)
    assert rows == 0
    assert target.read_bytes() == b"id,s\n"


def test_csv_repeat_export_byte_identical(path, target):
    export_query_file(path, "SELECT * FROM input", target)
    first = target.read_bytes()
    export_query_file(path, "SELECT * FROM input", target)
    assert target.read_bytes() == first


# ---------------------------------------------------------------------------
# JSONL rendering
# ---------------------------------------------------------------------------


def test_jsonl_basic(path, target):
    rows = export_query_file(
        path, "SELECT id, n, f, s, flag FROM input ORDER BY id", target, format="jsonl"
    )
    assert rows == 4
    assert target.read_text("utf-8") == (
        '{"id":1,"n":10,"f":1.5,"s":"a","flag":true}\n'
        '{"id":2,"n":null,"f":2.0,"s":"","flag":false}\n'
        '{"id":3,"n":30,"f":null,"s":null,"flag":true}\n'
        '{"id":4,"n":40,"f":-0.25,"s":"x,y","flag":false}\n'
    )


def test_jsonl_non_ascii_not_escaped_and_key_order(path, target):
    export_query_file(
        path, 'SELECT "to the, utf8", id FROM input ORDER BY id', target, format="jsonl"
    )
    first = target.read_text("utf-8").split("\n")[0]
    assert first == '{"to the, utf8":"い","id":1}'


def test_jsonl_zero_rows_is_zero_bytes(path, target):
    rows = export_query_file(path, "SELECT id FROM input WHERE id > 100", target, format="jsonl")
    assert rows == 0
    assert target.read_bytes() == b""


def test_jsonl_repeat_byte_identical(path, target):
    export_query_file(path, "SELECT * FROM input", target, format="jsonl")
    first = target.read_bytes()
    export_query_file(path, "SELECT * FROM input", target, format="jsonl")
    assert target.read_bytes() == first


def test_format_default_is_csv(path, target):
    export_query_file(path, "SELECT id FROM input", target)
    assert target.read_text("utf-8").split("\n")[0] == "id"


# ---------------------------------------------------------------------------
# Failures: no target created, existing target untouched
# ---------------------------------------------------------------------------


def test_unknown_format_raises_value_error(path, target):
    with pytest.raises(ValueError):
        export_query_file(path, "SELECT id FROM input", target, format="xml")
    assert not target.exists()


def test_target_equal_to_source_raises_value_error(path):
    with pytest.raises(ValueError):
        export_query_file(path, "SELECT id FROM input", path)
    with pytest.raises(ValueError):
        export_query_file(path, "SELECT id FROM input", str(path))


def test_failure_leaves_existing_target_untouched(path, target):
    target.write_bytes(b"keep me")
    with pytest.raises(QueryValidationError):
        export_query_file(path, "SELECT nope FROM input", target)
    assert target.read_bytes() == b"keep me"
    with pytest.raises(QuerySyntaxError):
        export_query_file(path, "SELCT id FROM input", target)
    assert target.read_bytes() == b"keep me"


def test_missing_source_is_os_error(tmp_path, target):
    with pytest.raises(OSError):
        export_query_file(tmp_path / "nope.caef", "SELECT id FROM input", target)
    assert not target.exists()


def test_malformed_source_is_format_error(tmp_path, target):
    bad = tmp_path / "bad.caef"
    bad.write_bytes(b"not a columnar file")
    with pytest.raises(ColumnarFormatError):
        export_query_file(bad, "SELECT id FROM input", target)
    assert not target.exists()


def test_unwritable_target_directory_is_os_error(path, tmp_path):
    with pytest.raises(OSError):
        export_query_file(path, "SELECT id FROM input", tmp_path / "no-dir" / "out.csv")


# ---------------------------------------------------------------------------
# Two-file exports
# ---------------------------------------------------------------------------


@pytest.fixture()
def sources(tmp_path):
    left = tmp_path / "l.caef"
    right = tmp_path / "r.caef"
    write_file(
        left,
        Table(
            Schema([ColumnSchema("k", "int64"), ColumnSchema("v", "utf8", nullable=True)]),
            {"k": [1, 2, 3], "v": ["a", "b", None]},
        ),
    )
    write_file(
        right,
        Table(
            Schema([ColumnSchema("k", "int64"), ColumnSchema("w", "float64")]),
            {"k": [1, 3], "w": [0.5, 1.5]},
        ),
    )
    return {"l": left, "r": right}


def test_export_files_join_csv(sources, tmp_path):
    target = tmp_path / "out.csv"
    rows = export_query_files(
        sources,
        "SELECT l.k, l.v, r.w FROM l LEFT JOIN r ON l.k = r.k ORDER BY l.k",
        target,
    )
    assert rows == 3
    assert target.read_text("utf-8") == "l.k,l.v,r.w\n1,a,0.5\n2,b,\n3,,1.5\n"


def test_export_files_jsonl(sources, tmp_path):
    target = tmp_path / "out.jsonl"
    rows = export_query_files(
        sources,
        "SELECT l.k, r.w FROM l INNER JOIN r ON l.k = r.k ORDER BY l.k",
        target,
        format="jsonl",
    )
    assert rows == 2
    assert target.read_text("utf-8") == '{"l.k":1,"r.w":0.5}\n{"l.k":3,"r.w":1.5}\n'


def test_export_files_unreferenced_source_not_checked(sources, tmp_path):
    # Only referenced sources take part in the collision check.
    target = tmp_path / "out.csv"
    rows = export_query_files(sources, "SELECT k, v FROM l ORDER BY k", target)
    assert rows == 3


def test_export_files_target_collision_with_referenced_source(sources):
    with pytest.raises(ValueError):
        export_query_files(sources, "SELECT k FROM l", sources["l"])
    with pytest.raises(ValueError):
        export_query_files(
            sources, "SELECT l.k FROM l INNER JOIN r ON l.k = r.k", sources["r"]
        )


def test_export_files_bad_sources_raise_value_error(tmp_path):
    with pytest.raises(ValueError):
        export_query_files({}, "SELECT k FROM l", tmp_path / "out.csv")
    with pytest.raises(ValueError):
        export_query_files({"l": 3}, "SELECT k FROM l", tmp_path / "out.csv")


def test_export_files_query_errors(sources, tmp_path):
    target = tmp_path / "out.csv"
    with pytest.raises(QuerySyntaxError):
        export_query_files(sources, "SELECT k", target)
    with pytest.raises(QueryValidationError):
        export_query_files(sources, "SELECT nope FROM l", target)
    assert not target.exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_export_success_quiet(path, target, capsys):
    code = main(["export", str(path), "SELECT id, flag FROM input ORDER BY id", str(target)])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert target.read_text("utf-8") == "id,flag\n1,true\n2,false\n3,true\n4,false\n"


def test_cli_export_format_jsonl(path, tmp_path, capsys):
    target = tmp_path / "out.jsonl"
    code = main(
        [
            "export",
            str(path),
            "SELECT id FROM input ORDER BY id",
            str(target),
            "--format",
            "jsonl",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert target.read_text("utf-8") == '{"id":1}\n{"id":2}\n{"id":3}\n{"id":4}\n'


def test_cli_export_unknown_format_exit_2(path, target, capsys):
    code = main(["export", str(path), "SELECT id FROM input", str(target), "--format", "xml"])
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() != ""
    assert not target.exists()


def test_cli_export_query_error_exit_2(path, target, capsys):
    assert main(["export", str(path), "SELCT id FROM input", str(target)]) == 2
    assert main(["export", str(path), "SELECT nope FROM input", str(target)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert not target.exists()


def test_cli_export_os_error_exit_1(tmp_path, capsys):
    target = tmp_path / "out.csv"
    code = main(["export", str(tmp_path / "nope.caef"), "SELECT id FROM input", str(target)])
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() != ""
    assert not target.exists()


def test_cli_export_files_success(sources, tmp_path, capsys):
    target = tmp_path / "out.csv"
    code = main(
        [
            "export-files",
            json.dumps({k: str(v) for k, v in sources.items()}),
            "SELECT l.k, r.w FROM l INNER JOIN r ON l.k = r.k ORDER BY l.k",
            str(target),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert target.read_text("utf-8") == "l.k,r.w\n1,0.5\n3,1.5\n"


def test_cli_export_files_invalid_sources_json_exit_2(tmp_path, capsys):
    code = main(["export-files", "{not json", "SELECT k FROM l", str(tmp_path / "o.csv")])
    assert code == 2
    assert capsys.readouterr().out == ""


def test_cli_export_files_value_error_exit_2(sources, tmp_path, capsys):
    code = main(
        [
            "export-files",
            json.dumps({k: str(v) for k, v in sources.items()}),
            "SELECT k FROM l",
            str(sources["l"]),
        ]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() != ""
