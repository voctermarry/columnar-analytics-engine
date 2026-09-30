"""Tests for the single-file SQL query layer."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from columnar_analytics import (
    ColumnSchema,
    ColumnarFormatError,
    QuerySyntaxError,
    QueryValidationError,
    Schema,
    Table,
    query_file,
    write_file,
)
from columnar_analytics.cli import main

SCHEMA = Schema(
    [
        ColumnSchema("flag", "bool"),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("f", "float64"),
        ColumnSchema("s", "utf8", nullable=True),
        ColumnSchema("select", "int64"),
    ]
)

ROWS = {
    "flag": [True, False, True, False, True],
    "n": [1, None, -7, 2, 10],
    "f": [1.5, 2.5, -0.25, 2.5, 100.0],
    "s": ["b", "a", None, "a", "世界"],
    "select": [0, 1, 2, 3, 4],
}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "t.caef"
    write_file(p, Table(SCHEMA, ROWS), compression="zlib", dictionary_encoding=["s"])
    return p


def rows_of(table):
    return [
        [table._columns[j][i] for j in range(len(table.schema.columns))]
        for i in range(table.row_count)
    ]


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def test_select_star(path):
    result = query_file(path, "SELECT * FROM input")
    assert result.schema == SCHEMA
    assert result.row_count == 5
    assert result.column_names == SCHEMA.names
    for name in SCHEMA.names:
        assert result.column(name) == ROWS[name]


def test_select_star_keeps_order_with_plain_file(tmp_path):
    p = tmp_path / "plain.caef"
    write_file(p, Table(SCHEMA, ROWS))
    result = query_file(p, "select * from input")
    assert result.column_names == ("flag", "n", "f", "s", "select")


def test_projection_subset_and_reorder(path):
    result = query_file(path, "select s, n, flag from input")
    assert result.column_names == ("s", "n", "flag")
    assert result.column("s") == ROWS["s"]
    assert result.column("n") == ROWS["n"]
    assert result.column("flag") == ROWS["flag"]


def test_projection_preserves_column_metadata(path):
    result = query_file(path, "select n from input")
    col = result.schema.columns[0]
    assert col.name == "n"
    assert col.type == "int64"
    assert col.nullable is True


def test_quoted_keyword_column_name(path):
    result = query_file(path, 'select "select" from input')
    assert result.column_names == ("select",)
    assert result.column("select") == [0, 1, 2, 3, 4]
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select select from input")


def test_unicode_column_identifier(tmp_path):
    p = tmp_path / "u.caef"
    schema = Schema([ColumnSchema("名前", "utf8"), ColumnSchema("x", "int64")])
    write_file(p, Table(schema, {"名前": ["a", "b"], "x": [1, 2]}))
    result = query_file(p, "select 名前 from input where x = 2")
    assert result.column_names == ("名前",)
    assert result.column("名前") == ["b"]
    quoted = query_file(p, 'select "名前" from input')
    assert quoted.column("名前") == ["a", "b"]
    # identifiers are exact-match: a different Unicode name is unknown
    with pytest.raises(QueryValidationError):
        query_file(p, "select 名 from input")


def test_duplicate_projection_column(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select n, n from input")
    with pytest.raises(QueryValidationError):
        query_file(path, 'select n, "N" from input')  # names are case/exact sensitive


def test_unknown_projection_column(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select nope from input")


def test_alias_not_supported(path):
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select n AS x from input")


# ---------------------------------------------------------------------------
# FROM clause
# ---------------------------------------------------------------------------


def test_from_requires_input(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select * from other")
    with pytest.raises(QueryValidationError):
        query_file(path, 'select * from "input"')
    # keyword is case-insensitive
    assert query_file(path, "select * from INPUT").row_count == 5


# ---------------------------------------------------------------------------
# WHERE: comparisons and literals
# ---------------------------------------------------------------------------


def test_comparison_operators(path):
    assert query_file(path, "select n from input where n = 1").column("n") == [1]
    assert query_file(path, "select n from input where n != 1").column("n") == [-7, 2, 10]
    assert query_file(path, "select n from input where n < 2").column("n") == [1, -7]
    assert query_file(path, "select n from input where n <= 2").column("n") == [1, -7, 2]
    assert query_file(path, "select n from input where n > 2").column("n") == [10]
    assert query_file(path, "select n from input where n >= 2").column("n") == [2, 10]


def test_signed_literals(path):
    assert query_file(path, "select n from input where n = -7").column("n") == [-7]
    assert query_file(path, "select n from input where f = -0.25").column("n") == [-7]
    assert query_file(path, "select n from input where n >= +1").column("n") == [1, 2, 10]


def test_int_float_cross_comparison(path):
    assert query_file(path, "select n from input where n < 2.5").column("n") == [1, -7, 2]
    assert query_file(path, "select n from input where f > 2").column("n") == [None, 2, 10]
    assert query_file(path, "select n from input where 2 = f").column("n") == []
    assert query_file(path, "select n from input where 100.0 = f").column("n") == [10]


def test_string_literal(path):
    assert query_file(path, "select n from input where s = 'a'").column("n") == [None, 2]
    assert query_file(path, "select n from input where s != 'a'").column("n") == [1, 10]
    assert query_file(path, "select n from input where s < 'c'").column("n") == [1, None, 2]
    # doubled single quote is an escaped quote
    result = query_file(path, "select n from input where s = 'it''s'")
    assert result.column("n") == []


def test_bool_literal_equality_only(path):
    assert query_file(path, "select n from input where flag = TRUE").column("n") == [1, -7, 10]
    assert query_file(path, "select n from input where flag = false").column("n") == [None, 2]
    with pytest.raises(QueryValidationError):
        query_file(path, "select * from input where flag < true")
    with pytest.raises(QueryValidationError):
        query_file(path, "select * from input where flag >= false")


def test_incomplete_type_combinations(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select * from input where n = '1'")
    with pytest.raises(QueryValidationError):
        query_file(path, "select * from input where s = 1")
    with pytest.raises(QueryValidationError):
        query_file(path, "select * from input where f = true")
    with pytest.raises(QueryValidationError):
        query_file(path, "select * from input where s < 1")
    with pytest.raises(QueryValidationError):
        query_file(path, "select * from input where flag = 1")


def test_is_null(path):
    assert query_file(path, "select n from input where n is null").column("n") == [None]
    assert query_file(path, "select n from input where s is null").column("n") == [-7]
    assert query_file(path, "select n from input where n is not null").column("n") == [1, -7, 2, 10]
    assert query_file(path, "select n from input where s is not null").column("n") == [1, None, 2, 10]


def test_where_unknown_column(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select * from input where missing = 1")


# ---------------------------------------------------------------------------
# Three-valued logic
# ---------------------------------------------------------------------------


def test_null_comparison_propagates_unknown(path):
    # n is NULL in row 2: every ordinary comparison drops that row
    result = query_file(path, "select n from input where n >= 0")
    assert result.column("n") == [1, 2, 10]
    result = query_file(path, "select n from input where n != 1")
    assert result.column("n") == [-7, 2, 10]


def test_three_valued_truth_table(path):
    # UNKNOWN AND TRUE -> UNKNOWN (NULL-n row has s='a'); T AND UNKNOWN -> UNKNOWN (n=10)
    result = query_file(path, "select n from input where n > 0 and s = 'a'")
    assert result.column("n") == [2]
    result = query_file(path, "select n from input where n > 0 and s is null")
    assert result.column("n") == []
    # NULL OR TRUE -> TRUE
    result = query_file(path, "select n from input where n is null or flag = true")
    assert result.column("n") == [1, None, -7, 10]
    # UNKNOWN OR FALSE -> UNKNOWN; NOT(UNKNOWN) -> UNKNOWN
    result = query_file(path, "select n from input where not (n > 5 or s = 'z')")
    # row0 F or F -> T; NULL-n row U or F -> U; n=-7 F or U -> U; n=2 F or F -> T
    assert result.column("n") == [1, 2]
    result = query_file(path, "select n from input where not (n < 100)")
    assert result.column("n") == []  # NULL row stays UNKNOWN
    result = query_file(path, "select n from input where not (n >= 0)")
    assert result.column("n") == [-7]


def test_and_or_with_nulls_directly():
    from columnar_analytics.query import _eval_predicate, _Logical, _BoundColumn, _Comparison, _Literal

    col = _BoundColumn(0, "int64", True, "n")
    lit = _Literal(1, "int64")
    # NULL = 1 is UNKNOWN
    cmp_unknown = _Comparison(col, "=", lit)
    true_node = _Comparison(_Literal(2, "int64"), "=", _Literal(2, "int64"))
    false_node = _Comparison(_Literal(2, "int64"), "=", _Literal(3, "int64"))
    assert _eval_predicate(cmp_unknown, (None,)) is None
    assert _eval_predicate(_Logical("and", cmp_unknown, false_node), (None,)) is False
    assert _eval_predicate(_Logical("and", cmp_unknown, true_node), (None,)) is None
    assert _eval_predicate(_Logical("or", cmp_unknown, true_node), (None,)) is True
    assert _eval_predicate(_Logical("or", cmp_unknown, false_node), (None,)) is None


# ---------------------------------------------------------------------------
# Precedence, parentheses, keywords
# ---------------------------------------------------------------------------


def test_precedence_not_comparison_and_or(path):
    # OR loosest: (flag = FALSE AND n > 0) OR (n = 1)
    result = query_file(
        path, "select n from input where flag = false and n > 0 or n = 1"
    )
    assert result.column("n") == [1, 2]
    # NOT binds over the following predicate: (NOT flag = TRUE) AND n > 0
    result = query_file(path, "select n from input where not flag = true and n > 0")
    assert result.column("n") == [2]


def test_parentheses_change_grouping(path):
    result = query_file(
        path, "select n from input where flag = false and (n > 0 or n = 1)"
    )
    assert result.column("n") == [2]
    result = query_file(path, "select n from input where ((n = 10))")
    assert result.column("n") == [10]


def test_keyword_case_insensitive(path):
    result = query_file(path, "SeLeCt n FrOm input wHeRe n Is NoT nUlL AnD n = 10")
    assert result.column("n") == [10]


def test_no_where_keeps_all_rows(path):
    assert query_file(path, "select n from input").row_count == 5


def test_row_order_preserved(path):
    result = query_file(path, "select n from input where n is not null")
    assert result.column("n") == [1, -7, 2, 10]


# ---------------------------------------------------------------------------
# Empty results and empty tables
# ---------------------------------------------------------------------------


def test_empty_result_keeps_columns(path):
    result = query_file(path, "select s, n from input where n > 1000")
    assert result.column_names == ("s", "n")
    cols = result.schema.columns
    assert cols[0].type == "utf8" and cols[0].nullable is True
    assert cols[1].type == "int64" and cols[1].nullable is True
    assert result.row_count == 0
    assert result.column("s") == []
    assert rows_of(result) == []


def test_empty_table(tmp_path):
    p = tmp_path / "empty.caef"
    schema = Schema([ColumnSchema("a", "int64"), ColumnSchema("b", "utf8", nullable=True)])
    write_file(p, Table(schema, {"a": [], "b": []}))
    result = query_file(p, "select b, a from input where a > 3")
    assert result.column_names == ("b", "a")
    assert result.row_count == 0
    assert result.column("b") == []
    result = query_file(p, "select * from input")
    assert result.column_names == ("a", "b")
    assert result.row_count == 0


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   ",
        "SELECT",
        "SELECT *",
        "SELECT * FROM",
        "SELECT * FROM input WHERE",
        "SELECT * FROM input WHERE n",
        "SELECT * FROM input WHERE n =",
        "SELECT * FROM input WHERE = 1",
        "SELECT * FROM input WHERE n > 1 AND",
        "SELECT * FROM input WHERE (n = 1",
        "SELECT * FROM input WHERE n = 1)",
        "SELECT * FROM input WHERE NOT",
        "SELECT * FROM input WHERE n == 1",
        "SELECT * FROM input WHERE n ~ 1",
        "SELECT * FROM input WHERE n = 1;",
        "SELECT * FROM input ORDER BY n",
        "SELECT * FROM input WHERE n = 1 GROUP BY n",
        "SELECT * FROM input WHERE n = 1 LIMIT 1",
        "SELECT * FROM input WHERE n = NULL",
        "SELECT * FROM input WHERE n = 1e999",
        "SELECT * FROM input WHERE n = 9223372036854775808",
        "SELECT * FROM input WHERE n = 'abc",
        'SELECT * FROM input WHERE n = "abc',
        "SELECT *, n FROM input",
        "SELECT n FROM input JOIN x",
        "DELETE FROM input",
        "select n from input where n = 1 /* x */",
        "select n from input where n = 1 -- x",
        "SELECT n FROM input WHERE n LIKE 'a'",
        "SELECT n FROM input WHERE n IN (1, 2)",
    ],
)
def test_syntax_errors(path, sql):
    with pytest.raises(QuerySyntaxError):
        query_file(path, sql)


def test_non_string_sql(path):
    with pytest.raises(QuerySyntaxError):
        query_file(path, b"select * from input")  # type: ignore[arg-type]


def test_int64_literal_bounds(path):
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select * from input where n = -9223372036854775809")
    result = query_file(path, "select * from input where n = -9223372036854775808")
    assert result.row_count == 0


# ---------------------------------------------------------------------------
# Error propagation from the file layer
# ---------------------------------------------------------------------------


def test_format_error_propagates(tmp_path):
    p = tmp_path / "bad"
    p.write_bytes(b"not a columnar file")
    with pytest.raises(ColumnarFormatError):
        query_file(p, "select * from input")


def test_os_error_propagates(tmp_path):
    with pytest.raises(FileNotFoundError):
        query_file(tmp_path / "missing", "select * from input")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_query_json_shape(path, capsys):
    code = main(["query", str(path), "select n, s from input where n = 2"])
    assert code == 0
    out = capsys.readouterr().out
    assert out.endswith("\n")
    assert "\n" not in out[:-1]  # single line
    payload = json.loads(out)
    assert list(payload) == ["columns", "rows"]
    assert payload["columns"] == [
        {"name": "n", "type": "int64", "nullable": True},
        {"name": "s", "type": "utf8", "nullable": True},
    ]
    assert payload["rows"] == [[2, "a"]]


def test_cli_query_unicode_and_null(path, capsys):
    code = main(["query", str(path), "select n, s from input where n >= 10 or s is null"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["rows"] == [[-7, None], [10, "世界"]]


def test_cli_query_empty_result_keeps_columns(path, capsys):
    code = main(["query", str(path), "select n from input where n > 1000"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [{"name": "n", "type": "int64", "nullable": True}]
    assert payload["rows"] == []


def test_cli_query_deterministic_bytes(path, capsys):
    sql = "select flag, n, f, s from input where f >= -0.25"
    outputs = []
    for _ in range(3):
        assert main(["query", str(path), sql]) == 0
        outputs.append(capsys.readouterr().out.encode("utf-8"))
    assert outputs[0] == outputs[1] == outputs[2]
    payload = json.loads(outputs[0].decode("utf-8"))
    assert list(payload["columns"][0]) == ["name", "type", "nullable"]


@pytest.mark.parametrize(
    "sql",
    [
        "select from input",
        "select * from other",
        "select nope from input",
        "select n, n from input",
        "select * from input where n = 'x'",
        "select * from input where flag < true",
    ],
)
def test_cli_query_syntax_or_validation_error(path, capsys, sql):
    code = main(["query", str(path), sql])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_query_format_error(tmp_path, capsys):
    p = tmp_path / "bad"
    p.write_bytes(b"junk")
    code = main(["query", str(p), "select * from input"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_query_os_error(tmp_path, capsys):
    code = main(["query", str(tmp_path / "missing"), "select * from input"])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_query_subprocess(path):
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "columnar_analytics.cli",
            "query",
            str(path),
            "select n from input where s = '世界'",
        ],
        capture_output=True,
        env=env,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout.decode("utf-8"))
    assert payload == {"columns": [{"name": "n", "type": "int64", "nullable": True}], "rows": [[10]]}


def test_public_exports():
    import columnar_analytics

    assert columnar_analytics.query_file is query_file
    assert issubclass(QuerySyntaxError, Exception)
    assert issubclass(QueryValidationError, Exception)
    assert QuerySyntaxError is not QueryValidationError
