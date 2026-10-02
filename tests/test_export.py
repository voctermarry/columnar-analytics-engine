"""Tests for the CSV / JSONL query-export layer and the export CLI commands."""

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
        ColumnSchema("名前", "utf8", nullable=True),
    ]
)

DATA = {
    "id": [1, 2, 3],
    "n": [10, None, 30],
    "f": [1.5, None, -0.0],
    "s": ["plain", "", 'a,b"x"\ny'],
    "flag": [True, False, True],
    "名前": ["い", None, "ろ"],
}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "t.caef"
    write_file(p, Table(SCHEMA, DATA))
    return p


def read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def test_csv_default_format_header_order_and_lf(path, tmp_path):
    dst = tmp_path / "out.csv"
    rows = export_query_file(path, "SELECT id, n, f, s, flag FROM input ORDER BY id", dst)
    assert rows == 3
    blob = read_bytes(dst)
    assert not blob.startswith(b"\xef\xbb\xbf")  # no UTF-8 BOM
    assert blob == (
        b"id,n,f,s,flag\n"
        b"1,10,1.5,plain,true\n"
        b"2,,,\"\",false\n"
        b'3,30,-0.0,"a,b""x""\ny",true\n'
    )
    assert b"\r" not in blob
    assert blob.endswith(b"\n")


def test_csv_null_is_empty_field_empty_string_is_quoted(path, tmp_path):
    dst = tmp_path / "out.csv"
    export_query_file(
        path, "SELECT s, n FROM input WHERE id = 1 OR id = 2 ORDER BY id", dst
    )
    lines = read_bytes(dst).decode("utf-8").split("\n")
    assert lines == ["s,n", "plain,10", '"",', ""]


def test_csv_quoting_rules_and_quote_doubling(path, tmp_path):
    dst = tmp_path / "out.csv"
    export_query_file(path, "SELECT s FROM input WHERE id = 3", dst)
    text = read_bytes(dst).decode("utf-8")
    assert text == 's\n"a,b""x""\ny"\n'


def test_csv_carriage_return_is_quoted_and_preserved(tmp_path):
    schema = Schema([ColumnSchema("v", "utf8")])
    p = tmp_path / "cr.caef"
    write_file(p, Table(schema, {"v": ["a\rb"]}))
    dst = tmp_path / "out.csv"
    export_query_file(p, "SELECT v FROM input", dst)
    assert read_bytes(dst) == b'v\n"a\rb"\n'


def test_csv_bool_and_number_text(path, tmp_path):
    dst = tmp_path / "out.csv"
    export_query_file(path, "SELECT flag, f, id FROM input WHERE id = 1", dst)
    assert read_bytes(dst) == b"flag,f,id\ntrue,1.5,1\n"


def test_csv_non_ascii_is_not_escaped(path, tmp_path):
    dst = tmp_path / "out.csv"
    export_query_file(path, "SELECT 名前 FROM input WHERE id = 1", dst)
    assert read_bytes(dst) == "名前\nい\n".encode("utf-8")


def test_csv_zero_rows_is_header_only(path, tmp_path):
    dst = tmp_path / "out.csv"
    rows = export_query_file(path, "SELECT id, s FROM input WHERE id > 1000", dst)
    assert rows == 0
    assert read_bytes(dst) == b"id,s\n"


def test_csv_header_escapes_special_characters(tmp_path):
    schema = Schema(
        [ColumnSchema("a,b", "int64"), ColumnSchema('q"q', "utf8", nullable=True)]
    )
    p = tmp_path / "weird.caef"
    write_file(p, Table(schema, {"a,b": [1], 'q"q': ["x"]}))
    dst = tmp_path / "out.csv"
    export_query_file(p, "SELECT * FROM input", dst)
    assert read_bytes(dst) == b'"a,b","q""q"\n1,x\n'


def test_csv_aggregate_and_scalar_alias_labels(path, tmp_path):
    agg = tmp_path / "agg.csv"
    export_query_file(
        path,
        "SELECT flag, COUNT(*) FROM input GROUP BY flag ORDER BY flag",
        agg,
    )
    lines = read_bytes(agg).decode("utf-8").split("\n")
    assert lines[0] == "flag,COUNT(*)"
    assert lines[1] == "false,1"
    assert lines[2] == "true,2"
    assert lines[3] == ""

    expr = tmp_path / "expr.csv"
    export_query_file(
        path, "SELECT id + 1 AS next FROM input WHERE id = 1", expr
    )
    assert read_bytes(expr) == b"next\n2\n"


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def test_jsonl_compact_objects_in_column_order(path, tmp_path):
    dst = tmp_path / "out.jsonl"
    rows = export_query_file(
        path,
        "SELECT id, n, f, s, flag FROM input ORDER BY id",
        dst,
        format="jsonl",
    )
    assert rows == 3
    blob = read_bytes(dst)
    assert blob == (
        b'{"id":1,"n":10,"f":1.5,"s":"plain","flag":true}\n'
        b'{"id":2,"n":null,"f":null,"s":"","flag":false}\n'
        b'{"id":3,"n":30,"f":-0.0,"s":"a,b\\"x\\"\\ny","flag":true}\n'
    )
    for line in blob.decode("utf-8").splitlines():
        json.loads(line)  # every line is a standalone JSON object


def test_jsonl_non_ascii_unescaped_and_null(path, tmp_path):
    dst = tmp_path / "out.jsonl"
    export_query_file(
        path, "SELECT 名前 FROM input ORDER BY id", dst, format="jsonl"
    )
    blob = read_bytes(dst)
    assert blob == '{"名前":"い"}\n{"名前":null}\n{"名前":"ろ"}\n'.encode("utf-8")
    assert b"\\u" not in blob


def test_jsonl_zero_rows_is_zero_bytes(path, tmp_path):
    dst = tmp_path / "out.jsonl"
    rows = export_query_file(
        path, "SELECT id FROM input WHERE id > 1000", dst, format="jsonl"
    )
    assert rows == 0
    assert read_bytes(dst) == b""


def test_jsonl_join_qualified_keys(tmp_path):
    left = tmp_path / "l.caef"
    right = tmp_path / "r.caef"
    write_file(
        left,
        Table(
            Schema([ColumnSchema("id", "int64"), ColumnSchema("v", "utf8")]),
            {"id": [1], "v": ["x"]},
        ),
    )
    write_file(
        right,
        Table(
            Schema([ColumnSchema("id", "int64"), ColumnSchema("tag", "utf8")]),
            {"id": [1], "tag": ["y"]},
        ),
    )
    dst = tmp_path / "out.jsonl"
    export_query_files(
        {"l": left, "r": right},
        "SELECT l.id, r.tag FROM l INNER JOIN r ON l.id = r.id",
        dst,
        format="jsonl",
    )
    blob = read_bytes(dst)
    assert blob == b'{"l.id":1,"r.tag":"y"}\n'
    assert list(json.loads(blob.decode("utf-8").strip()).keys()) == ["l.id", "r.tag"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_repeated_exports_are_byte_identical(path, tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    export_query_file(path, "SELECT id, s FROM input", a)
    export_query_file(path, "SELECT id, s FROM input", b)
    assert read_bytes(a) == read_bytes(b)

    aj = tmp_path / "a.jsonl"
    bj = tmp_path / "bj.jsonl"
    export_query_file(path, "SELECT id, s FROM input", aj, format="jsonl")
    export_query_file(path, "SELECT id, s FROM input", bj, format="jsonl")
    assert read_bytes(aj) == read_bytes(bj)


def test_export_without_order_by_keeps_deterministic_query_order(path, tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    sql = "SELECT s FROM input WHERE id >= 2"
    export_query_file(path, sql, first)
    export_query_file(path, sql, second)
    assert read_bytes(first) == read_bytes(second)
    assert read_bytes(first) == b's\n""\n"a,b""x""\ny"\n'


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_unknown_format_raises_value_error(path, tmp_path):
    with pytest.raises(ValueError):
        export_query_file(path, "SELECT id FROM input", tmp_path / "x", format="xml")
    with pytest.raises(ValueError):
        export_query_file(path, "SELECT id FROM input", tmp_path / "x", format=None)


def test_destination_aliasing_source_raises_value_error(path, tmp_path):
    with pytest.raises(ValueError):
        export_query_file(path, "SELECT id FROM input", path)
    # The overlap check precedes query execution: a syntactically invalid
    # statement still raises ValueError for the alias.
    with pytest.raises(ValueError):
        export_query_file(path, "NOT SQL", path)
    assert path.exists()


def test_destination_aliasing_source_via_relative_or_symlink(path, tmp_path):
    import os

    alias = tmp_path / "alias.caef"
    os.symlink(path, alias)
    with pytest.raises(ValueError):
        export_query_file(path, "SELECT id FROM input", alias)

    # A destination spelling the same file through "." / ".." still aliases.
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        rel = os.path.join(".", os.path.basename(path))
        with pytest.raises(ValueError):
            export_query_file(rel, "SELECT id FROM input", rel)
    finally:
        os.chdir(cwd)


def test_destination_aliasing_referenced_join_source(tmp_path):
    left = tmp_path / "l.caef"
    right = tmp_path / "r.caef"
    one_row = Table(Schema([ColumnSchema("id", "int64")]), {"id": [1]})
    write_file(left, one_row)
    write_file(right, one_row)
    sources = {"l": left, "r": right}
    sql = "SELECT l.id FROM l INNER JOIN r ON l.id = r.id"
    with pytest.raises(ValueError):
        export_query_files(sources, sql, left)
    with pytest.raises(ValueError):
        export_query_files(sources, sql, right)
    # An unreferenced source may be overwritten by the export.
    only_left = tmp_path / "only.csv"
    export_query_files({"l": left, "r": right}, "SELECT id FROM l", only_left)
    assert read_bytes(only_left) == b"id\n1\n"


def test_sources_validation_still_value_error(tmp_path):
    with pytest.raises(ValueError):
        export_query_files({}, "SELECT id FROM l", tmp_path / "x")
    with pytest.raises(ValueError):
        export_query_files({"l": 3}, "SELECT id FROM l", tmp_path / "x")


# ---------------------------------------------------------------------------
# Atomicity: failures never create or alter the destination
# ---------------------------------------------------------------------------


def test_syntax_error_creates_no_destination(path, tmp_path):
    dst = tmp_path / "never.csv"
    with pytest.raises(QuerySyntaxError):
        export_query_file(path, "SELCT id FRM input", dst)
    assert not dst.exists()


def test_query_error_leaves_existing_destination_untouched(path, tmp_path):
    dst = tmp_path / "existing.csv"
    dst.write_bytes(b"KEEP ME\n")
    with pytest.raises(QueryValidationError):
        export_query_file(path, "SELECT nope FROM input", dst)
    assert read_bytes(dst) == b"KEEP ME\n"


def test_malformed_source_leaves_destination_untouched(tmp_path):
    src = tmp_path / "bad.caef"
    src.write_bytes(b"not a columnar file")
    dst = tmp_path / "out.csv"
    dst.write_bytes(b"KEEP ME\n")
    with pytest.raises(ColumnarFormatError):
        export_query_file(src, "SELECT * FROM input", dst)
    assert read_bytes(dst) == b"KEEP ME\n"


def test_os_error_leaves_destination_untouched(path, tmp_path):
    dst = tmp_path / "missing-dir" / "out.csv"
    with pytest.raises(OSError):
        export_query_file(path, "SELECT id FROM input", dst)
    assert not (tmp_path / "missing-dir").exists()


def test_successful_export_replaces_existing_file(path, tmp_path):
    dst = tmp_path / "out.csv"
    dst.write_bytes(b"OLD CONTENT")
    export_query_file(path, "SELECT id FROM input WHERE id = 1", dst)
    assert read_bytes(dst) == b"id\n1\n"


def test_query_files_syntax_error_does_not_touch_sources_or_dest(path, tmp_path):
    right = tmp_path / "r.caef"
    write_file(right, Table(Schema([ColumnSchema("id", "int64")]), {"id": [1]}))
    dst = tmp_path / "out.csv"
    with pytest.raises(QuerySyntaxError):
        export_query_files(
            {"l": path, "r": right}, "SELECT l.id FROM l INNER JOIN r", dst
        )
    assert not dst.exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_export_success_is_silent_exit_0(path, tmp_path, capsys):
    dst = tmp_path / "out.csv"
    code = main(["export", str(path), "SELECT id FROM input WHERE id = 1", str(dst)])
    assert code == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert read_bytes(dst) == b"id\n1\n"


def test_cli_export_format_jsonl(path, tmp_path, capsys):
    dst = tmp_path / "out.jsonl"
    code = main(
        [
            "export",
            str(path),
            "SELECT id FROM input ORDER BY id",
            str(dst),
            "--format",
            "jsonl",
        ]
    )
    assert code == 0
    assert capsys.readouterr().out == ""
    assert read_bytes(dst) == b'{"id":1}\n{"id":2}\n{"id":3}\n'


@pytest.mark.parametrize(
    "sql",
    [
        "SELCT id FROM input",
        "SELECT nope FROM input",
    ],
)
def test_cli_export_query_errors_exit_2(path, tmp_path, capsys, sql):
    dst = tmp_path / "out.csv"
    code = main(["export", str(path), sql, str(dst)])
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip()
    assert not dst.exists()


def test_cli_export_unknown_format_exit_2(path, tmp_path, capsys):
    dst = tmp_path / "out.csv"
    code = main(
        ["export", str(path), "SELECT id FROM input", str(dst), "--format", "pdf"]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "format" in captured.err
    assert not dst.exists()


def test_cli_export_destination_overlap_exit_2(path, capsys):
    code = main(["export", str(path), "SELECT id FROM input", str(path)])
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_export_os_error_exit_1(path, tmp_path, capsys):
    dst = tmp_path / "no-such-dir" / "out.csv"
    code = main(["export", str(path), "SELECT id FROM input", str(dst)])
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_export_files_success_and_errors(tmp_path, capsys):
    left = tmp_path / "l.caef"
    right = tmp_path / "r.caef"
    write_file(left, Table(Schema([ColumnSchema("id", "int64")]), {"id": [1, 2]}))
    write_file(right, Table(Schema([ColumnSchema("id", "int64")]), {"id": [1]}))

    dst = tmp_path / "out.csv"
    code = main(
        [
            "export-files",
            json.dumps({"l": str(left), "r": str(right)}),
            "SELECT l.id FROM l INNER JOIN r ON l.id = r.id",
            str(dst),
        ]
    )
    assert code == 0
    assert capsys.readouterr().out == ""
    assert read_bytes(dst) == b"l.id\n1\n"

    bad_json = tmp_path / "bad.csv"
    code = main(["export-files", "{not json", "SELECT l.id FROM l", str(bad_json)])
    assert code == 2
    assert capsys.readouterr().out == ""
    assert not bad_json.exists()

    bad_sources = tmp_path / "bad2.csv"
    code = main(
        [
            "export-files",
            json.dumps({"l": 3}),
            "SELECT id FROM l",
            str(bad_sources),
        ]
    )
    assert code == 2
    assert not bad_sources.exists()

    overlap = tmp_path / "overlap.csv"
    code = main(
        [
            "export-files",
            json.dumps({"l": str(left), "r": str(right)}),
            "SELECT l.id FROM l INNER JOIN r ON l.id = r.id",
            str(right),
        ]
    )
    assert code == 2
    assert not overlap.exists()
