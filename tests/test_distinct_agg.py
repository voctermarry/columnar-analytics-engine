"""Tests for argument-level DISTINCT aggregates (COUNT/SUM/AVG/MIN/MAX)."""

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
    query_file,
    query_files,
    write_file,
)
from columnar_analytics.cli import main


SCHEMA = Schema(
    [
        ColumnSchema("g", "utf8"),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("f", "float64", nullable=True),
        ColumnSchema("s", "utf8", nullable=True),
        ColumnSchema("flag", "bool", nullable=True),
    ]
)

DATA = {
    "g": ["a", "a", "a", "b", "b", "b"],
    "n": [1, 1, 2, None, None, 5],
    "f": [0.0, -0.0, 1.5, 2.5, 2.5, None],
    "s": ["u", "u", "v", None, "w", "w"],
    "flag": [True, True, False, None, False, False],
}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "t.caef"
    write_file(p, Table(SCHEMA, DATA))
    return p


@pytest.fixture()
def sources(tmp_path):
    left = tmp_path / "l.caef"
    write_file(
        left,
        Table(
            Schema([ColumnSchema("k", "int64"), ColumnSchema("v", "int64", nullable=True)]),
            {"k": [1, 1, 2, 3], "v": [10, 10, 20, None]},
        ),
    )
    right = tmp_path / "r.caef"
    write_file(
        right,
        Table(
            Schema([ColumnSchema("k", "int64"), ColumnSchema("t", "utf8", nullable=True)]),
            {"k": [1, 2, 2], "t": ["a", "b", "b"]},
        ),
    )
    return {"l": left, "r": right}


def rows_of(table):
    return [
        [table._columns[c][r] for c in range(len(table.schema.columns))]
        for r in range(table.row_count)
    ]


# ---------------------------------------------------------------------------
# Global (ungrouped) DISTINCT aggregates
# ---------------------------------------------------------------------------


def test_count_distinct_int64(path):
    result = query_file(path, "SELECT COUNT(DISTINCT n) FROM input")
    assert result.column_names == ("COUNT(DISTINCT n)",)
    col = result.schema.columns[0]
    assert (col.type, col.nullable) == ("int64", False)
    assert result.column("COUNT(DISTINCT n)") == [3]


def test_count_distinct_utf8_and_bool(path):
    result = query_file(
        path, "SELECT COUNT(DISTINCT s), COUNT(DISTINCT flag) FROM input"
    )
    assert result.column("COUNT(DISTINCT s)") == [3]
    assert result.column("COUNT(DISTINCT flag)") == [2]


def test_count_distinct_ignores_nulls(path):
    result = query_file(path, "SELECT COUNT(DISTINCT n) FROM input WHERE g = 'b'")
    assert result.column("COUNT(DISTINCT n)") == [1]


def test_sum_avg_min_max_distinct(path):
    result = query_file(
        path,
        "SELECT SUM(DISTINCT n), AVG(DISTINCT n), MIN(DISTINCT n), MAX(DISTINCT n) "
        "FROM input",
    )
    assert result.column_names == (
        "SUM(DISTINCT n)",
        "AVG(DISTINCT n)",
        "MIN(DISTINCT n)",
        "MAX(DISTINCT n)",
    )
    assert result.column("SUM(DISTINCT n)") == [8]
    assert result.column("AVG(DISTINCT n)") == [8 / 3]
    assert result.column("MIN(DISTINCT n)") == [1]
    assert result.column("MAX(DISTINCT n)") == [5]
    for col in result.schema.columns:
        assert col.nullable is True


def test_float_distinct_treats_negative_zero_as_zero(path):
    # 0.0 and -0.0 are one distinct value: {0, 1.5, 2.5}.
    result = query_file(
        path,
        "SELECT COUNT(DISTINCT f), SUM(DISTINCT f), MIN(DISTINCT f) FROM input",
    )
    assert result.column("COUNT(DISTINCT f)") == [3]
    assert result.column("SUM(DISTINCT f)") == [4.0]
    assert result.column("MIN(DISTINCT f)") == [0.0]


def test_distinct_aggregates_empty_input(tmp_path):
    p = tmp_path / "empty.caef"
    write_file(p, Table(SCHEMA, {name: [] for name in SCHEMA.names}))
    result = query_file(
        p,
        "SELECT COUNT(DISTINCT n), SUM(DISTINCT n), AVG(DISTINCT n), "
        "MIN(DISTINCT n), MAX(DISTINCT n) FROM input",
    )
    assert rows_of(result) == [[0, None, None, None, None]]
    assert result.schema.columns[0].nullable is False


def test_distinct_aggregates_null_only_input(path):
    result = query_file(
        path,
        "SELECT COUNT(DISTINCT n), SUM(DISTINCT n), MIN(DISTINCT n) "
        "FROM input WHERE n IS NULL",
    )
    assert rows_of(result) == [[0, None, None]]


def test_sum_distinct_overflow_still_raises(tmp_path):
    p = tmp_path / "big.caef"
    schema = Schema([ColumnSchema("x", "int64")])
    write_file(p, Table(schema, {"x": [2**63 - 1, 1]}))
    with pytest.raises(QueryValidationError):
        query_file(p, "SELECT SUM(DISTINCT x) FROM input")


def test_sum_distinct_deduplicates_before_summing(tmp_path):
    p = tmp_path / "big.caef"
    schema = Schema([ColumnSchema("x", "int64")])
    write_file(p, Table(schema, {"x": [2**63 - 1, 2**63 - 1]}))
    result = query_file(p, "SELECT SUM(DISTINCT x) FROM input")
    assert result.column("SUM(DISTINCT x)") == [2**63 - 1]


# ---------------------------------------------------------------------------
# Grouped queries, HAVING and ORDER BY
# ---------------------------------------------------------------------------


def test_grouped_distinct_aggregates(path):
    result = query_file(
        path,
        "SELECT g, COUNT(DISTINCT n), SUM(DISTINCT f) FROM input "
        "GROUP BY g ORDER BY g",
    )
    assert rows_of(result) == [["a", 2, 1.5], ["b", 1, 2.5]]


def test_having_may_use_unprojected_distinct_aggregate(path):
    result = query_file(
        path, "SELECT g FROM input GROUP BY g HAVING COUNT(DISTINCT n) > 1 ORDER BY g"
    )
    assert result.column("g") == ["a"]


def test_order_by_selected_distinct_aggregate(path):
    result = query_file(
        path,
        "SELECT g, COUNT(DISTINCT n) FROM input GROUP BY g "
        "ORDER BY COUNT(DISTINCT n) DESC, g",
    )
    assert rows_of(result) == [["a", 2], ["b", 1]]


def test_order_by_unselected_distinct_aggregate_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "SELECT g, COUNT(n) FROM input GROUP BY g ORDER BY COUNT(DISTINCT n)",
        )


def test_plain_and_distinct_calls_are_separate_aggregates(path):
    result = query_file(
        path, "SELECT COUNT(n), COUNT(DISTINCT n) FROM input"
    )
    assert result.column_names == ("COUNT(n)", "COUNT(DISTINCT n)")
    assert rows_of(result) == [[4, 3]]


def test_identical_calls_are_computed_once(path):
    plan = explain_file(
        path,
        "SELECT COUNT(DISTINCT n) FROM input "
        "HAVING COUNT(DISTINCT n) > 0 ORDER BY COUNT(DISTINCT n)",
    )
    aggregate = next(op for op in plan["operators"] if op["operator"] == "Aggregate")
    assert aggregate["aggregates"] == [
        {
            "function": "COUNT",
            "argument": "n",
            "output": "COUNT(DISTINCT n)",
            "distinct": True,
        }
    ]


def test_keyword_case_and_whitespace_do_not_change_results(path):
    sql1 = "SELECT g, COUNT(DISTINCT n) FROM input GROUP BY g ORDER BY g"
    sql2 = "select g, count(  DiStInCt   n ) from input group by g order by g"
    r1, r2 = query_file(path, sql1), query_file(path, sql2)
    assert [c.name for c in r1.schema.columns] == [c.name for c in r2.schema.columns]
    assert rows_of(r1) == rows_of(r2)
    assert explain_file(path, sql1) == explain_file(path, sql2)


# ---------------------------------------------------------------------------
# Explain plan
# ---------------------------------------------------------------------------


def test_explain_aggregate_operator_marks_only_distinct_items(path):
    plan = explain_file(
        path,
        "SELECT g, COUNT(DISTINCT n), SUM(n) FROM input GROUP BY g "
        "HAVING MAX(DISTINCT n) > 0",
    )
    aggregate = next(op for op in plan["operators"] if op["operator"] == "Aggregate")
    # First-reference order: SELECT items, then the HAVING-only aggregate.
    assert aggregate["aggregates"] == [
        {
            "function": "COUNT",
            "argument": "n",
            "output": "COUNT(DISTINCT n)",
            "distinct": True,
        },
        {"function": "SUM", "argument": "n", "output": "SUM(n)"},
        {
            "function": "MAX",
            "argument": "n",
            "output": "MAX(DISTINCT n)",
            "distinct": True,
        },
    ]
    having = next(op for op in plan["operators"] if op["operator"] == "Having")
    leaf = having["condition"]["operands"][0]
    assert leaf["kind"] == "aggregate"
    assert leaf["distinct"] is True
    project = next(op for op in plan["operators"] if op["operator"] == "Project")
    agg_exprs = [e["expression"] for e in project["expressions"] if e["expression"]["kind"] == "aggregate"]
    assert agg_exprs[0]["distinct"] is True
    assert "distinct" not in agg_exprs[1]
    assert plan["output"][1] == {
        "name": "COUNT(DISTINCT n)",
        "type": "int64",
        "nullable": False,
    }


def test_explain_plain_aggregates_have_no_distinct_field(path):
    plan = explain_file(path, "SELECT COUNT(n), SUM(n), AVG(n) FROM input")
    assert '"distinct"' not in json.dumps(plan)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def test_distinct_aggregate_syntax_errors_before_file_access(tmp_path):
    missing = tmp_path / "nope.caef"
    for sql in (
        "SELECT COUNT(DISTINCT) FROM input",  # missing argument
        "SELECT COUNT(DISTINCT *) FROM input",  # star
        "SELECT SUM(DISTINCT *) FROM input",  # star
        "SELECT SUM(DISTINCT n, g) FROM input",  # multiple arguments
        "SELECT SUM(DISTINCT n + 1) FROM input",  # expression argument
        "SELECT SUM(DISTINCT COUNT(n)) FROM input",  # nested aggregate
        "SELECT SUM(DISTINCT DISTINCT n) FROM input",  # repeated DISTINCT
        "SELECT SUM(DISTINCT 'x') FROM input",  # literal argument
        "SELECT SUM(n DISTINCT) FROM input",  # misplaced DISTINCT
    ):
        with pytest.raises(QuerySyntaxError):
            query_file(missing, sql)
        with pytest.raises(QuerySyntaxError):
            explain_file(missing, sql)


def test_distinct_aggregate_validation_errors(path):
    for sql in (
        "SELECT COUNT(DISTINCT nope) FROM input",  # unknown column
        "SELECT SUM(DISTINCT s) FROM input",  # type-incompatible argument
        "SELECT AVG(DISTINCT flag) FROM input",  # type-incompatible argument
        "SELECT COUNT(DISTINCT n), COUNT(DISTINCT n) FROM input",  # duplicate output
        "SELECT DISTINCT COUNT(DISTINCT n) FROM input",  # SELECT DISTINCT + aggregate
        "SELECT g, COUNT(DISTINCT n) FROM input",  # ungrouped plain column
    ):
        with pytest.raises(QueryValidationError):
            query_file(path, sql)


def test_plain_aggregate_argument_errors_unchanged(tmp_path):
    # Queries that do not use the new syntax keep their historical classes.
    missing = tmp_path / "nope.caef"
    with pytest.raises(QueryValidationError):
        query_file(missing, "SELECT SUM(n + 1) FROM input")
    with pytest.raises(QueryValidationError):
        query_file(missing, "SELECT COUNT(SUM(n)) FROM input")


# ---------------------------------------------------------------------------
# Two-file joins
# ---------------------------------------------------------------------------


def test_join_distinct_aggregates(sources):
    sql = (
        "SELECT l.k, COUNT(DISTINCT l.v), COUNT(DISTINCT r.t) FROM l "
        "LEFT JOIN r ON l.k = r.k GROUP BY l.k ORDER BY l.k"
    )
    expected = [[1, 1, 1], [2, 1, 1], [3, 0, 0]]
    reference = None
    for strategy in (None, "hash", "sort_merge"):
        result = query_files(sources, sql, strategy)
        assert result.column_names == ("l.k", "COUNT(DISTINCT l.v)", "COUNT(DISTINCT r.t)")
        assert rows_of(result) == expected
        if reference is None:
            reference = result
        else:
            assert rows_of(result) == rows_of(reference)


def test_join_distinct_aggregate_requires_qualified_column(sources):
    with pytest.raises(QueryValidationError):
        query_files(
            sources, "SELECT COUNT(DISTINCT v) FROM l INNER JOIN r ON l.k = r.k"
        )


def test_join_explain_marks_distinct_aggregates(sources):
    plan = explain_files(
        sources,
        "SELECT COUNT(DISTINCT l.v) FROM l INNER JOIN r ON l.k = r.k",
        "sort_merge",
    )
    aggregate = next(op for op in plan["operators"] if op["operator"] == "Aggregate")
    assert aggregate["aggregates"] == [
        {
            "function": "COUNT",
            "argument": "l.v",
            "output": "COUNT(DISTINCT l.v)",
            "distinct": True,
        }
    ]


# ---------------------------------------------------------------------------
# Export and CLI
# ---------------------------------------------------------------------------


def test_export_distinct_aggregates(path, tmp_path):
    csv_dest = tmp_path / "out.csv"
    export_query_file(path, "SELECT COUNT(DISTINCT n) FROM input", csv_dest, "csv")
    assert csv_dest.read_text(encoding="utf-8").splitlines() == [
        "COUNT(DISTINCT n)",
        "3",
    ]
    jsonl_dest = tmp_path / "out.jsonl"
    export_query_file(path, "SELECT COUNT(DISTINCT n) FROM input", jsonl_dest, "jsonl")
    assert json.loads(jsonl_dest.read_text(encoding="utf-8")) == {
        "COUNT(DISTINCT n)": 3
    }


def test_cli_query_and_explain(path, capsys):
    assert main(["query", str(path), "SELECT COUNT(DISTINCT n) FROM input"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["columns"] == [
        {"name": "COUNT(DISTINCT n)", "type": "int64", "nullable": False}
    ]
    assert out["rows"] == [[3]]
    assert main(["explain", str(path), "SELECT COUNT(DISTINCT n) FROM input"]) == 0
    plan = json.loads(capsys.readouterr().out)
    aggregate = next(op for op in plan["operators"] if op["operator"] == "Aggregate")
    assert aggregate["aggregates"][0]["distinct"] is True


def test_cli_distinct_syntax_error_exit_code(path, capsys):
    rc = main(["query", str(path), "SELECT COUNT(DISTINCT) FROM input"])
    assert rc == 2
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# SELECT DISTINCT row deduplication is unaffected
# ---------------------------------------------------------------------------


def test_select_distinct_row_dedup_unchanged(path):
    result = query_file(path, "SELECT DISTINCT s FROM input")
    assert result.column("s") == ["u", "v", None, "w"]
