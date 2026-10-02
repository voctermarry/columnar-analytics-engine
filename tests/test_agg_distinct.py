"""Tests for DISTINCT aggregates: COUNT/SUM/AVG/MIN/MAX(DISTINCT col).

Covers single-file and two-file (joined) statements, per-group and global
deduplication, NULL and float64 signed-zero handling, result types and
nullability, HAVING / ORDER BY placement, the explain plan shape, CSV /
JSONL export, the CLI entries and the error classification.  Plain
aggregate behaviour is pinned by the other suites; a few checks here pin
that plans and results without the new syntax keep their old shape.
"""

from __future__ import annotations

import json

import pytest

from columnar_analytics import (
    ColumnSchema,
    QuerySyntaxError,
    QueryValidationError,
    Schema,
    Table,
    explain_file,
    explain_files,
    export_query_file,
    export_query_files,
    query_file,
    query_files,
    write_file,
)
from columnar_analytics.cli import main


SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("f", "float64", nullable=True),
        ColumnSchema("s", "utf8", nullable=True),
        ColumnSchema("flag", "bool", nullable=True),
    ]
)

DATA = {
    "id": [1, 2, 3, 4, 5, 6, 7],
    "n": [10, None, 10, 10, None, 30, 10],
    "f": [1.5, 2.5, 0.0, -0.0, None, 1.5, 2.5],
    "s": ["a", "b", None, "a", "c", "a", None],
    "flag": [True, False, True, True, None, False, True],
}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "t.caef"
    write_file(p, Table(SCHEMA, DATA), compression="zlib", dictionary_encoding=["s"])
    return p


LEFT_SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("v", "float64", nullable=True),
    ]
)
LEFT_DATA = {"id": [1, 2, 3, 4], "k": [10, 10, 20, None], "v": [1.0, 1.0, 2.0, None]}
RIGHT_SCHEMA = Schema(
    [
        ColumnSchema("rid", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("tag", "utf8", nullable=True),
    ]
)
RIGHT_DATA = {"rid": [100, 200, 300], "k": [10, 10, 30], "tag": ["x", "x", "z"]}


@pytest.fixture()
def sources(tmp_path):
    left = tmp_path / "l.caef"
    right = tmp_path / "r.caef"
    write_file(left, Table(LEFT_SCHEMA, LEFT_DATA))
    write_file(right, Table(RIGHT_SCHEMA, RIGHT_DATA))
    return {"l": left, "r": right}


def rows(result):
    return [
        [result._columns[c][i] for c in range(len(result.schema.columns))]
        for i in range(result.row_count)
    ]


def find_operator(plan, name):
    for op in plan["operators"]:
        if op["operator"] == name:
            return op
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Global (no GROUP BY) aggregation
# ---------------------------------------------------------------------------


def test_count_distinct_int64(path):
    result = query_file(path, "SELECT COUNT(DISTINCT n) FROM input")
    assert result.column_names == ("COUNT(DISTINCT n)",)
    col = result.schema.columns[0]
    assert (col.type, col.nullable) == ("int64", False)
    assert result.column("COUNT(DISTINCT n)") == [2]


def test_count_distinct_float_signed_zero(path):
    # 0.0 and -0.0 are the same distinct value; NULLs are ignored.
    result = query_file(path, "SELECT COUNT(DISTINCT f) FROM input")
    assert result.column("COUNT(DISTINCT f)") == [3]


def test_count_distinct_utf8_and_bool(path):
    result = query_file(path, "SELECT COUNT(DISTINCT s), COUNT(DISTINCT flag) FROM input")
    assert result.column("COUNT(DISTINCT s)") == [3]
    assert result.column("COUNT(DISTINCT flag)") == [2]


def test_sum_avg_min_max_distinct_int64(path):
    result = query_file(
        path,
        "SELECT SUM(DISTINCT n), AVG(DISTINCT n), MIN(DISTINCT n), MAX(DISTINCT n) FROM input",
    )
    assert result.column("SUM(DISTINCT n)") == [40]
    assert result.column("AVG(DISTINCT n)") == [20.0]
    assert result.column("MIN(DISTINCT n)") == [10]
    assert result.column("MAX(DISTINCT n)") == [30]
    types = {c.name: (c.type, c.nullable) for c in result.schema.columns}
    assert types == {
        "SUM(DISTINCT n)": ("int64", True),
        "AVG(DISTINCT n)": ("float64", True),
        "MIN(DISTINCT n)": ("int64", True),
        "MAX(DISTINCT n)": ("int64", True),
    }


def test_sum_avg_min_max_distinct_float64(path):
    result = query_file(
        path,
        "SELECT SUM(DISTINCT f), AVG(DISTINCT f), MIN(DISTINCT f), MAX(DISTINCT f) FROM input",
    )
    # Distinct values: {1.5, 2.5, 0.0} (0.0 and -0.0 collapse).
    assert result.column("SUM(DISTINCT f)") == [4.0]
    assert result.column("AVG(DISTINCT f)") == [pytest.approx(4.0 / 3)]
    assert result.column("MIN(DISTINCT f)") == [0.0]
    assert result.column("MAX(DISTINCT f)") == [2.5]


def test_min_max_distinct_utf8_and_bool(path):
    result = query_file(
        path,
        "SELECT MIN(DISTINCT s), MAX(DISTINCT s), MIN(DISTINCT flag), MAX(DISTINCT flag) FROM input",
    )
    assert result.column("MIN(DISTINCT s)") == ["a"]
    assert result.column("MAX(DISTINCT s)") == ["c"]
    assert result.column("MIN(DISTINCT flag)") == [False]
    assert result.column("MAX(DISTINCT flag)") == [True]


def test_distinct_aggregates_empty_selection(path):
    result = query_file(
        path,
        "SELECT COUNT(DISTINCT n), SUM(DISTINCT n), AVG(DISTINCT n), "
        "MIN(DISTINCT s), MAX(DISTINCT f) FROM input WHERE id > 100",
    )
    assert rows(result) == [[0, None, None, None, None]]
    count_col = result.schema.columns[0]
    assert (count_col.type, count_col.nullable) == ("int64", False)


def test_distinct_aggregates_all_null_argument(path):
    result = query_file(
        path,
        "SELECT COUNT(DISTINCT n), SUM(DISTINCT n) FROM input WHERE n IS NULL",
    )
    assert rows(result) == [[0, None]]


def test_where_applies_before_dedup(path):
    result = query_file(
        path, "SELECT COUNT(DISTINCT n), SUM(DISTINCT n) FROM input WHERE id < 4"
    )
    assert rows(result) == [[1, 10]]


def test_plain_and_distinct_calls_coexist(path):
    result = query_file(
        path, "SELECT COUNT(n), COUNT(DISTINCT n), SUM(n), SUM(DISTINCT n) FROM input"
    )
    assert result.column_names == (
        "COUNT(n)",
        "COUNT(DISTINCT n)",
        "SUM(n)",
        "SUM(DISTINCT n)",
    )
    assert rows(result) == [[5, 2, 70, 40]]


def test_keyword_case_and_whitespace_insignificant(path):
    result = query_file(path, "SELECT count(  Distinct   n ) FROM input")
    assert result.column_names == ("COUNT(DISTINCT n)",)
    assert result.column("COUNT(DISTINCT n)") == [2]
    plan_a = explain_file(path, "SELECT COUNT(DISTINCT n) FROM input")
    plan_b = explain_file(path, "SELECT count(  distinct   n ) FROM input")
    assert plan_a == plan_b


# ---------------------------------------------------------------------------
# GROUP BY / HAVING / ORDER BY / LIMIT
# ---------------------------------------------------------------------------


def test_group_by_distinct_per_group(path):
    result = query_file(
        path,
        "SELECT s, COUNT(DISTINCT n), SUM(DISTINCT n) FROM input "
        "GROUP BY s ORDER BY s",
    )
    # s='a' sees n in {10, 30}; the NULL group sees {10}; 'b'/'c' see only NULL.
    assert rows(result) == [
        ["a", 2, 40],
        ["b", 0, None],
        ["c", 0, None],
        [None, 1, 10],
    ]


def test_group_by_distinct_empty_selection_yields_zero_rows(path):
    result = query_file(
        path,
        "SELECT s, COUNT(DISTINCT n) FROM input WHERE id > 100 GROUP BY s",
    )
    assert result.row_count == 0
    assert result.column_names == ("s", "COUNT(DISTINCT n)")


def test_having_unprojected_distinct_aggregate(path):
    result = query_file(
        path,
        "SELECT s FROM input GROUP BY s HAVING COUNT(DISTINCT n) > 1 ORDER BY s",
    )
    assert result.column("s") == ["a"]


def test_having_on_projected_distinct_aggregate(path):
    result = query_file(
        path,
        "SELECT s, COUNT(DISTINCT n) FROM input GROUP BY s "
        "HAVING COUNT(DISTINCT n) <= 1 ORDER BY s",
    )
    assert rows(result) == [["b", 0], ["c", 0], [None, 1]]


def test_order_by_selected_distinct_aggregate(path):
    result = query_file(
        path,
        "SELECT s, SUM(DISTINCT n) FROM input GROUP BY s "
        "ORDER BY SUM(DISTINCT n) DESC NULLS LAST, s LIMIT 2",
    )
    assert rows(result) == [["a", 40], [None, 10]]


def test_order_by_distinct_aggregate_must_be_selected(path):
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "SELECT s, COUNT(DISTINCT n) FROM input GROUP BY s "
            "ORDER BY COUNT(DISTINCT id)",
        )


def test_order_by_plain_and_distinct_are_different_results(path):
    # ORDER BY COUNT(DISTINCT n) does not match a selected plain COUNT(n).
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "SELECT s, COUNT(n) FROM input GROUP BY s ORDER BY COUNT(DISTINCT n)",
        )
    result = query_file(
        path,
        "SELECT s, COUNT(n) FROM input GROUP BY s ORDER BY COUNT(n) DESC, s LIMIT 1",
    )
    assert rows(result) == [["a", 3]]


def test_same_call_computed_once_across_clauses(path):
    plan = explain_file(
        path,
        "SELECT s, COUNT(DISTINCT n) FROM input GROUP BY s "
        "HAVING COUNT(DISTINCT n) > 0 ORDER BY COUNT(DISTINCT n)",
    )
    aggregates = find_operator(plan, "Aggregate")["aggregates"]
    assert aggregates == [
        {
            "function": "COUNT",
            "argument": "n",
            "output": "COUNT(DISTINCT n)",
            "distinct": True,
        }
    ]


# ---------------------------------------------------------------------------
# Explain plan shape
# ---------------------------------------------------------------------------


def test_explain_aggregate_operator_marks_only_distinct_entries(path):
    plan = explain_file(
        path,
        "SELECT s, SUM(n), COUNT(DISTINCT n) FROM input GROUP BY s "
        "HAVING MAX(DISTINCT id) > 1 ORDER BY COUNT(DISTINCT n)",
    )
    agg = find_operator(plan, "Aggregate")
    assert agg["group_keys"] == ["s"]
    # First-reference order: SELECT left to right, then HAVING.
    assert agg["aggregates"] == [
        {"function": "SUM", "argument": "n", "output": "SUM(n)"},
        {
            "function": "COUNT",
            "argument": "n",
            "output": "COUNT(DISTINCT n)",
            "distinct": True,
        },
        {
            "function": "MAX",
            "argument": "id",
            "output": "MAX(DISTINCT id)",
            "distinct": True,
        },
    ]


def test_explain_plain_aggregates_keep_historical_shape(path):
    plan = explain_file(path, "SELECT s, COUNT(n), SUM(f) FROM input GROUP BY s")
    assert find_operator(plan, "Aggregate")["aggregates"] == [
        {"function": "COUNT", "argument": "n", "output": "COUNT(n)"},
        {"function": "SUM", "argument": "f", "output": "SUM(f)"},
    ]


def test_explain_output_matches_query_result_schema(path):
    sql = (
        "SELECT s, COUNT(DISTINCT n), AVG(DISTINCT f) FROM input "
        "GROUP BY s ORDER BY s LIMIT 2"
    )
    plan = explain_file(path, sql)
    result = query_file(path, sql)
    assert plan["output"] == [
        {"name": col.name, "type": col.type, "nullable": col.nullable}
        for col in result.schema.columns
    ]


def test_explain_plan_is_json_serialisable(path):
    plan = explain_file(path, "SELECT COUNT(DISTINCT s) FROM input")
    text = json.dumps(plan, ensure_ascii=False, allow_nan=False)
    assert "COUNT(DISTINCT s)" in text


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def test_distinct_argument_syntax_errors(path):
    for sql in (
        "SELECT COUNT(DISTINCT *) FROM input",
        "SELECT SUM(DISTINCT *) FROM input",
        "SELECT COUNT(DISTINCT) FROM input",
        "SELECT SUM(DISTINCT n + 1) FROM input",
        "SELECT COUNT(DISTINCT n, f) FROM input",
        "SELECT COUNT(n DISTINCT) FROM input",
        "SELECT COUNT(DISTINCT DISTINCT n) FROM input",
    ):
        with pytest.raises(QuerySyntaxError):
            query_file(path, sql)


def test_distinct_argument_syntax_errors_before_file_access(tmp_path):
    missing = tmp_path / "nope.caef"
    for sql in (
        "SELECT COUNT(DISTINCT *) FROM input",
        "SELECT COUNT(DISTINCT) FROM input",
        "SELECT SUM(DISTINCT n + 1) FROM input",
        "SELECT COUNT(DISTINCT n, f) FROM input",
    ):
        with pytest.raises(QuerySyntaxError):
            query_file(missing, sql)
        with pytest.raises(QuerySyntaxError):
            explain_file(missing, sql)


def test_distinct_aggregate_validation_errors(path):
    for sql in (
        "SELECT COUNT(DISTINCT nope) FROM input",
        "SELECT SUM(DISTINCT s) FROM input",
        "SELECT AVG(DISTINCT flag) FROM input",
        "SELECT SUM(DISTINCT COUNT(n)) FROM input",
        "SELECT DISTINCT COUNT(DISTINCT n) FROM input",
    ):
        with pytest.raises(QueryValidationError):
            query_file(path, sql)


def test_distinct_sum_int64_overflow(path, tmp_path):
    big = tmp_path / "big.caef"
    schema = Schema([ColumnSchema("n", "int64", nullable=True)])
    write_file(big, Table(schema, {"n": [2**62, 2**62, 1]}))
    # DISTINCT collapses the duplicates before summing.
    result = query_file(big, "SELECT SUM(DISTINCT n) FROM input")
    assert result.column("SUM(DISTINCT n)") == [2**62 + 1]
    with pytest.raises(QueryValidationError):
        query_file(big, "SELECT SUM(n) FROM input")
    # Two distinct values whose total leaves the int64 range still overflow.
    overflow = tmp_path / "overflow.caef"
    write_file(overflow, Table(schema, {"n": [2**62, 2**62 + 1, None]}))
    with pytest.raises(QueryValidationError):
        query_file(overflow, "SELECT SUM(DISTINCT n) FROM input")


def test_distinct_sum_non_finite_float(path, tmp_path):
    p = tmp_path / "f.caef"
    schema = Schema([ColumnSchema("f", "float64", nullable=True)])
    write_file(p, Table(schema, {"f": [1e308, 1.1e308, 1e308]}))
    with pytest.raises(QueryValidationError):
        query_file(p, "SELECT SUM(DISTINCT f) FROM input")
    # After deduplication the remaining values may still sum to a finite value.
    result = query_file(p, "SELECT SUM(DISTINCT f) FROM input WHERE f = 1e308")
    assert result.column("SUM(DISTINCT f)") == [1e308]


# ---------------------------------------------------------------------------
# Two-file joins
# ---------------------------------------------------------------------------


def test_join_distinct_aggregates(sources):
    result = query_files(
        sources,
        "SELECT l.k, COUNT(DISTINCT l.v), MIN(DISTINCT r.tag) FROM l "
        "LEFT JOIN r ON l.k = r.k GROUP BY l.k ORDER BY l.k",
    )
    assert result.column_names == ("l.k", "COUNT(DISTINCT l.v)", "MIN(DISTINCT r.tag)")
    assert rows(result) == [[10, 1, "x"], [20, 1, None], [None, 0, None]]


def test_join_distinct_requires_qualified_columns(sources):
    with pytest.raises(QueryValidationError):
        query_files(
            sources,
            "SELECT COUNT(DISTINCT v) FROM l INNER JOIN r ON l.k = r.k",
        )


def test_join_strategies_agree_on_distinct_aggregates(sources):
    sql = (
        "SELECT l.k, COUNT(DISTINCT l.v), SUM(DISTINCT l.id), MIN(DISTINCT r.tag) "
        "FROM l LEFT JOIN r ON l.k = r.k GROUP BY l.k ORDER BY l.k"
    )
    expected = query_files(sources, sql)
    for strategy in ("hash", "sort_merge"):
        result = query_files(sources, sql, strategy)
        assert result.column_names == expected.column_names
        assert rows(result) == rows(expected)
        assert [
            (c.name, c.type, c.nullable) for c in result.schema.columns
        ] == [(c.name, c.type, c.nullable) for c in expected.schema.columns]


def test_explain_files_join_distinct_aggregates(sources):
    plan = explain_files(
        sources,
        "SELECT l.k, COUNT(DISTINCT l.v) FROM l INNER JOIN r ON l.k = r.k "
        "GROUP BY l.k HAVING MAX(DISTINCT r.rid) > 100 ORDER BY l.k",
    )
    agg = find_operator(plan, "Aggregate")
    assert agg["group_keys"] == ["l.k"]
    assert agg["aggregates"] == [
        {
            "function": "COUNT",
            "argument": "l.v",
            "output": "COUNT(DISTINCT l.v)",
            "distinct": True,
        },
        {
            "function": "MAX",
            "argument": "r.rid",
            "output": "MAX(DISTINCT r.rid)",
            "distinct": True,
        },
    ]
    assert plan["output"] == [
        {"name": "l.k", "type": "int64", "nullable": True},
        {"name": "COUNT(DISTINCT l.v)", "type": "int64", "nullable": False},
    ]


# ---------------------------------------------------------------------------
# Export and CLI
# ---------------------------------------------------------------------------


def test_export_csv_and_jsonl(path, tmp_path):
    sql = "SELECT s, COUNT(DISTINCT n) FROM input GROUP BY s ORDER BY s"
    csv_dest = tmp_path / "out.csv"
    assert export_query_file(path, sql, csv_dest) == 4
    assert csv_dest.read_bytes() == (
        b"s,COUNT(DISTINCT n)\na,2\nb,0\nc,0\n,1\n"
    )
    jsonl_dest = tmp_path / "out.jsonl"
    assert export_query_file(path, sql, jsonl_dest, "jsonl") == 4
    assert jsonl_dest.read_bytes() == (
        b'{"s":"a","COUNT(DISTINCT n)":2}\n'
        b'{"s":"b","COUNT(DISTINCT n)":0}\n'
        b'{"s":"c","COUNT(DISTINCT n)":0}\n'
        b'{"s":null,"COUNT(DISTINCT n)":1}\n'
    )


def test_export_files_distinct_aggregates(sources, tmp_path):
    sql = (
        "SELECT l.k, COUNT(DISTINCT l.v) FROM l LEFT JOIN r ON l.k = r.k "
        "GROUP BY l.k ORDER BY l.k"
    )
    dest = tmp_path / "out.csv"
    assert export_query_files(sources, sql, dest, "csv", "sort_merge") == 3
    assert dest.read_bytes() == b"l.k,COUNT(DISTINCT l.v)\n10,1\n20,1\n,0\n"


def test_cli_query_and_explain(path, capsys):
    assert main(["query", str(path), "SELECT COUNT(DISTINCT n) FROM input"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["columns"] == [
        {"name": "COUNT(DISTINCT n)", "type": "int64", "nullable": False}
    ]
    assert payload["rows"] == [[2]]

    assert main(["explain", str(path), "SELECT COUNT(DISTINCT n) FROM input"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert find_operator(plan, "Aggregate")["aggregates"] == [
        {
            "function": "COUNT",
            "argument": "n",
            "output": "COUNT(DISTINCT n)",
            "distinct": True,
        }
    ]


def test_cli_distinct_syntax_error_exit_code(path, capsys):
    assert main(["query", str(path), "SELECT COUNT(DISTINCT *) FROM input"]) == 2
    assert capsys.readouterr().out == ""


def test_cli_export(path, tmp_path):
    dest = tmp_path / "out.csv"
    assert (
        main(["export", str(path), "SELECT COUNT(DISTINCT n) FROM input", str(dest)])
        == 0
    )
    assert dest.read_bytes() == b"COUNT(DISTINCT n)\n2\n"


# ---------------------------------------------------------------------------
# Stability / non-regression
# ---------------------------------------------------------------------------


def test_repeated_execution_is_stable(path):
    sql = (
        "SELECT s, COUNT(DISTINCT n), SUM(DISTINCT f) FROM input "
        "GROUP BY s HAVING COUNT(DISTINCT n) >= 0 ORDER BY s"
    )
    first = query_file(path, sql)
    for _ in range(3):
        again = query_file(path, sql)
        assert rows(again) == rows(first)
        assert [
            (c.name, c.type, c.nullable) for c in again.schema.columns
        ] == [(c.name, c.type, c.nullable) for c in first.schema.columns]


def test_select_distinct_row_dedup_unchanged(path):
    # Row-level SELECT DISTINCT is a separate feature and keeps its rules.
    result = query_file(path, "SELECT DISTINCT n FROM input ORDER BY n")
    assert result.column("n") == [10, 30, None]
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT COUNT(DISTINCT n) FROM input")
