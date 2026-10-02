"""Tests for the HAVING post-grouping filter (single file, joins, plan, CLI)."""

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
        ColumnSchema("id", "int64"),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("f", "float64", nullable=True),
        ColumnSchema("s", "utf8", nullable=True),
        ColumnSchema("flag", "bool"),
    ]
)

DATA = {
    "id": [1, 2, 3, 4, 5, 6],
    "n": [10, None, 30, 10, None, 30],
    "f": [1.5, 2.5, None, -0.5, 10.0, None],
    "s": ["a", "b", None, "a", "c", "a"],
    "flag": [True, False, True, False, True, False],
}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "t.caef"
    write_file(p, Table(SCHEMA, DATA))
    return p


@pytest.fixture()
def sources(tmp_path, path):
    right_schema = Schema(
        [
            ColumnSchema("rid", "int64"),
            ColumnSchema("k", "int64"),
            ColumnSchema("tag", "utf8"),
        ]
    )
    right = tmp_path / "r.caef"
    write_file(right, Table(right_schema, {"rid": [10, 20], "k": [1, 2], "tag": ["x", "y"]}))
    return {"l": path, "r": right}


def rows_of(table):
    return [
        [table._columns[c][r] for c in range(len(table.schema.columns))]
        for r in range(table.row_count)
    ]


# ---------------------------------------------------------------------------
# Basic filtering
# ---------------------------------------------------------------------------


def test_having_filters_groups_on_aggregate(path):
    result = query_file(
        path, "SELECT s, COUNT(*), SUM(n) FROM input GROUP BY s HAVING COUNT(*) > 1"
    )
    assert result.column_names == ("s", "COUNT(*)", "SUM(n)")
    # Only the 'a' group has more than one row; NULL/b/c groups are dropped.
    assert rows_of(result) == [["a", 3, 50]]


def test_having_on_group_column(path):
    result = query_file(
        path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING s = 'a'"
    )
    assert rows_of(result) == [["a", 3]]


def test_having_aggregate_not_in_select(path):
    result = query_file(path, "SELECT s FROM input GROUP BY s HAVING SUM(n) > 25")
    assert result.column_names == ("s",)
    # a: SUM(n)=50; NULL group: 60; b: NULL (excluded); c: NULL (excluded).
    assert result.column("s") == ["a", None]


def test_having_composition_with_not_and_or(path):
    result = query_file(
        path,
        "SELECT s FROM input GROUP BY s "
        "HAVING NOT (COUNT(*) = 1) OR s = 'c'",
    )
    assert result.column("s") == ["a", "c"]
    result = query_file(
        path,
        "SELECT s FROM input GROUP BY s "
        "HAVING COUNT(*) > 1 AND (s = 'a' OR s = 'z')",
    )
    assert result.column("s") == ["a"]


def test_having_is_null_three_valued_logic(path):
    # b and c groups have SUM(n) = NULL -> comparison UNKNOWN -> dropped.
    result = query_file(
        path, "SELECT s FROM input GROUP BY s HAVING SUM(n) > 0"
    )
    assert result.column("s") == ["a", None]
    # IS NULL keeps exactly the groups whose aggregate is NULL.
    result = query_file(
        path, "SELECT s FROM input GROUP BY s HAVING SUM(n) IS NULL ORDER BY s"
    )
    assert result.column("s") == ["b", "c"]
    result = query_file(
        path, "SELECT s FROM input GROUP BY s HAVING SUM(n) IS NOT NULL"
    )
    assert set(result.column("s")) == {"a", None}


def test_having_true_false_literals(path):
    result = query_file(
        path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING TRUE"
    )
    assert result.row_count == 4
    result = query_file(
        path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING FALSE"
    )
    assert result.row_count == 0


def test_having_bool_column_equality(path):
    result = query_file(
        path,
        "SELECT s FROM input GROUP BY s HAVING MIN(flag) = MAX(flag)",
    )
    # a has both True and False -> excluded; b is all False, c is True and
    # the NULL-key group has a single True row -> all three survive.
    assert result.column("s") == ["b", None, "c"]


# ---------------------------------------------------------------------------
# Global aggregation (no GROUP BY)
# ---------------------------------------------------------------------------


def test_having_without_group_by_filters_global_row(path):
    assert rows_of(query_file(path, "SELECT COUNT(*) FROM input HAVING COUNT(*) > 3")) == [[6]]
    assert query_file(path, "SELECT COUNT(*) FROM input HAVING COUNT(*) > 100").row_count == 0
    # Output schema is still described on a zero-row result.
    empty = query_file(path, "SELECT COUNT(*) FROM input HAVING COUNT(*) > 100")
    assert empty.column_names == ("COUNT(*)",)


def test_having_without_group_by_empty_where_still_builds_row(path):
    result = query_file(
        path, "SELECT COUNT(*) FROM input WHERE id > 1000 HAVING COUNT(*) = 0"
    )
    assert rows_of(result) == [[0]]
    result = query_file(
        path, "SELECT COUNT(*) FROM input WHERE id > 1000 HAVING COUNT(*) > 0"
    )
    assert result.row_count == 0
    # SUM over an empty group is NULL: IS NULL keeps the global row.
    result = query_file(
        path, "SELECT COUNT(*) FROM input WHERE id > 1000 HAVING SUM(n) IS NULL"
    )
    assert rows_of(result) == [[0]]


def test_having_global_aggregate_not_selected(path):
    result = query_file(path, "SELECT COUNT(*) FROM input HAVING SUM(n) = 80")
    assert rows_of(result) == [[6]]


# ---------------------------------------------------------------------------
# Interaction with WHERE / ORDER BY / LIMIT
# ---------------------------------------------------------------------------


def test_where_runs_before_grouping_having_after(path):
    result = query_file(
        path,
        "SELECT s, COUNT(*) FROM input WHERE id >= 3 "
        "GROUP BY s HAVING COUNT(*) = 1 ORDER BY s",
    )
    # Remaining rows id>=3: groups None(1), a(2), c(1); HAVING keeps the
    # size-1 groups, ORDER BY s ASC puts NULL last.
    assert rows_of(result) == [["c", 1], [None, 1]]


def test_grouped_empty_where_zero_groups_even_with_having(path):
    result = query_file(
        path,
        "SELECT s, COUNT(*) FROM input WHERE id > 1000 GROUP BY s "
        "HAVING COUNT(*) >= 0",
    )
    assert result.row_count == 0
    assert result.column_names == ("s", "COUNT(*)")


def test_having_runs_before_order_by_and_limit(path):
    result = query_file(
        path,
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) = 1 "
        "ORDER BY s DESC LIMIT 2",
    )
    # Surviving groups are b, NULL, c (one row each); DESC with NULLs last.
    assert rows_of(result) == [["c", 1], ["b", 1]]


def test_order_by_scope_not_extended_by_having(path):
    # An aggregate used only by HAVING is still not a legal ORDER BY target.
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "SELECT s FROM input GROUP BY s HAVING SUM(n) > 0 ORDER BY SUM(n)",
        )


def test_having_with_order_by_selected_aggregate(path):
    result = query_file(
        path,
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) >= 1 "
        "ORDER BY COUNT(*) DESC, s ASC",
    )
    assert result.column("COUNT(*)") == [3, 1, 1, 1]
    assert result.column("s") == ["a", "b", "c", None]


def test_no_order_by_keeps_first_row_group_order(path):
    result = query_file(
        path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) >= 1"
    )
    assert result.column("s") == ["a", "b", None, "c"]


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING",
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) > 1 HAVING COUNT(*) > 0",
        "SELECT COUNT(*) FROM input HAVING COUNT(*) > 1 GROUP BY s",
        "SELECT COUNT(*) FROM input ORDER BY COUNT(*) HAVING COUNT(*) > 1",
        "SELECT COUNT(*) FROM input LIMIT 2 HAVING COUNT(*) > 1",
        "SELECT COUNT(*) FROM input HAVING * > 1",
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING (COUNT(*) > 1",
    ],
)
def test_having_syntax_errors(path, sql):
    with pytest.raises(QuerySyntaxError):
        query_file(path, sql)


def test_having_syntax_error_before_file_access(tmp_path):
    missing = tmp_path / "nope.caef"
    with pytest.raises(QuerySyntaxError):
        explain_file(missing, "SELECT COUNT(*) FROM input HAVING")


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_having_on_completely_nonaggregate_query(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT id FROM input HAVING id > 1")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT s FROM input HAVING s = 'a'")


def test_having_ungrouped_column_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING n > 1")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING id = 1")


def test_having_unknown_column_and_bad_aggregate_argument(path):
    with pytest.raises(QueryValidationError):
        query_file(
            path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(missing) > 0"
        )
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) = id")


def test_having_type_incompatibility(path):
    with pytest.raises(QueryValidationError):
        query_file(
            path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) > 'x'"
        )
    with pytest.raises(QueryValidationError):
        query_file(
            path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING SUM(s) > 1"
        )
    with pytest.raises(QueryValidationError):
        query_file(
            path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) < TRUE"
        )


def test_having_non_boolean_condition(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT COUNT(*) FROM input HAVING SUM(n)")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING s")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT COUNT(*) FROM input HAVING 42")


def test_having_rejects_case_arithmetic_and_nested_aggregates(path):
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "SELECT s, COUNT(*) FROM input GROUP BY s "
            "HAVING CASE WHEN s = 'a' THEN 1 ELSE 0 END > 0",
        )
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) + 1 > 2",
        )
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "SELECT s, COUNT(*) FROM input GROUP BY s HAVING SUM(COUNT(*)) > 1",
        )


def test_having_keyword_case_insensitive(path):
    result = query_file(
        path, "select s, count(*) from input group by s having count(*) > 1"
    )
    assert result.column("s") == ["a"]


# ---------------------------------------------------------------------------
# Determinism / stability
# ---------------------------------------------------------------------------


def test_having_results_deterministic(path):
    sql = "SELECT s, COUNT(*), SUM(n) FROM input GROUP BY s HAVING COUNT(*) >= 1"
    first = query_file(path, sql)
    second = query_file(path, sql)
    assert rows_of(first) == rows_of(second)
    assert [c for c in first._columns] == [c for c in second._columns]


# ---------------------------------------------------------------------------
# EXPLAIN
# ---------------------------------------------------------------------------


def find_operator(plan, name):
    for op in plan["operators"]:
        if op["operator"] == name:
            return op
    raise KeyError(name)


def operator_kinds(plan):
    return [op["operator"] for op in plan["operators"]]


def test_explain_having_position_and_tree(path):
    plan = explain_file(
        path,
        "SELECT s, COUNT(*) FROM input GROUP BY s "
        "HAVING SUM(n) > 10 AND COUNT(*) >= 2 ORDER BY s LIMIT 1",
    )
    assert operator_kinds(plan) == [
        "Scan",
        "Aggregate",
        "Having",
        "Sort",
        "Limit",
        "Project",
    ]
    condition = find_operator(plan, "Having")["condition"]
    assert condition["kind"] == "logic"
    assert condition["operator"] == "AND"
    cmp_sum, cmp_count = condition["operands"]
    assert cmp_sum["operator"] == ">"
    leaf = cmp_sum["operands"][0]
    assert leaf == {
        "kind": "aggregate",
        "function": "SUM",
        "argument": "n",
        "type": "int64",
        "nullable": True,
    }
    assert cmp_sum["operands"][1] == {"kind": "literal", "type": "int64", "value": 10}
    count_leaf = cmp_count["operands"][0]
    assert count_leaf == {
        "kind": "aggregate",
        "function": "COUNT",
        "argument": None,
        "type": "int64",
        "nullable": False,
    }


def test_explain_aggregates_deduped_first_reference_order(path):
    # COUNT(*) is selected; AVG(f) is HAVING-only; SUM(n) is HAVING-only and
    # reused inside OR; COUNT(*) repeats and must not be listed twice.
    plan = explain_file(
        path,
        "SELECT s, COUNT(*) FROM input GROUP BY s "
        "HAVING AVG(f) > 0 OR SUM(n) > 1 OR COUNT(*) > 0",
    )
    aggregate = find_operator(plan, "Aggregate")
    assert aggregate["group_keys"] == ["s"]
    assert aggregate["aggregates"] == [
        {"function": "COUNT", "argument": None, "output": "COUNT(*)"},
        {"function": "AVG", "argument": "f", "output": "AVG(f)"},
        {"function": "SUM", "argument": "n", "output": "SUM(n)"},
    ]
    # The scan must include f and n (HAVING aggregate args) plus the group
    # key s, in schema order; output still only describes SELECT.
    assert find_operator(plan, "Scan")["required_columns"] == ["n", "f", "s"]
    assert [col["name"] for col in plan["output"]] == ["s", "COUNT(*)"]


def test_explain_having_group_column_leaf(path):
    plan = explain_file(
        path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING s IS NOT NULL"
    )
    condition = find_operator(plan, "Having")["condition"]
    assert condition == {
        "kind": "is_null",
        "operator": "IS NOT NULL",
        "operands": [{"kind": "column", "name": "s"}],
    }


def test_explain_having_absent_when_clause_absent(path):
    plan = explain_file(path, "SELECT s, COUNT(*) FROM input GROUP BY s")
    assert "Having" not in operator_kinds(plan)


def test_explain_global_aggregation_having(path):
    plan = explain_file(path, "SELECT COUNT(*) FROM input HAVING MIN(f) < 0")
    aggregate = find_operator(plan, "Aggregate")
    assert aggregate["group_keys"] == []
    assert aggregate["aggregates"] == [
        {"function": "COUNT", "argument": None, "output": "COUNT(*)"},
        {"function": "MIN", "argument": "f", "output": "MIN(f)"},
    ]
    assert operator_kinds(plan) == ["Scan", "Aggregate", "Having", "Project"]


def test_explain_having_validation_errors(path):
    with pytest.raises(QueryValidationError):
        explain_file(path, "SELECT id FROM input HAVING id > 1")
    with pytest.raises(QueryValidationError):
        explain_file(
            path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING n > 1"
        )


def test_explain_having_join(sources):
    sql = (
        "SELECT l.s, COUNT(*) FROM l INNER JOIN r ON l.id = r.k "
        "GROUP BY l.s HAVING COUNT(*) = 1 AND SUM(r.rid) > 10"
    )
    plan = explain_files(sources, sql)
    kinds = operator_kinds(plan)
    assert kinds == [
        "Scan",
        "Scan",
        "Join",
        "Aggregate",
        "Having",
        "Project",
    ]
    scans = {
        op["source"]: op["required_columns"]
        for op in plan["operators"]
        if op["operator"] == "Scan"
    }
    assert scans == {"l": ["id", "s"], "r": ["rid", "k"]}
    condition = find_operator(plan, "Having")["condition"]
    assert condition["operands"][0]["operands"][0]["function"] == "COUNT"
    assert condition["operands"][1]["operands"][0]["argument"] == "r.rid"
    # Result correctness matches the plan.
    result = query_files(sources, sql)
    assert rows_of(result) == [["b", 1]]


def test_explain_having_join_unqualified_column_rejected(sources):
    with pytest.raises(QueryValidationError):
        explain_files(
            sources,
            "SELECT l.s, COUNT(*) FROM l INNER JOIN r ON l.id = r.k "
            "GROUP BY l.s HAVING COUNT(*) > 0 AND s IS NOT NULL",
        )


def test_having_left_join_counts_null_padded_rows(path, tmp_path):
    # l has six rows; right matches only ids 1 and 2, so unmatched left rows
    # get NULL right-side values: COUNT(r.k) excludes those NULLs.
    right = tmp_path / "r.caef"
    write_file(
        right,
        Table(
            Schema(
                [
                    ColumnSchema("k", "int64"),
                    ColumnSchema("v", "int64", nullable=True),
                ]
            ),
            {"k": [1, 2], "v": [10, 20]},
        ),
    )
    sql = (
        "SELECT l.s, COUNT(*), COUNT(r.k), SUM(r.v) "
        "FROM l LEFT JOIN r ON l.id = r.k GROUP BY l.s "
        "HAVING COUNT(r.k) > 0 ORDER BY l.s"
    )
    result = query_files({"l": path, "r": right}, sql)
    # 'a' covers ids 1/4/6 but only id 1 matches (v=10); 'b' is id 2 and
    # matches v=20; the NULL-key and 'c' groups stay unmatched and are cut.
    assert rows_of(result) == [["a", 3, 1, 10], ["b", 1, 1, 20]]


def test_having_only_aggregate_still_validates_arguments(tmp_path):
    p = tmp_path / "big.caef"
    write_file(p, Table(Schema([ColumnSchema("x", "int64")]), {"x": [2**63 - 1, 1]}))
    # SUM(x) is not projected, but its int64 overflow still surfaces.
    with pytest.raises(QueryValidationError):
        query_file(p, "SELECT COUNT(*) FROM input HAVING SUM(x) > 0")


def test_having_signed_numeric_literal(path):
    result = query_file(
        path,
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) >= -1",
    )
    assert result.row_count == 4
    result = query_file(
        path,
        "SELECT s, AVG(id) FROM input GROUP BY s HAVING AVG(id) > 3",
    )
    # Group means: a=11/3, b=2, c=5, NULL-key=3 -> a and c survive.
    assert set(result.column("s")) == {"a", "c"}


# ---------------------------------------------------------------------------
# Export and CLI
# ---------------------------------------------------------------------------


def test_export_honours_having(path, tmp_path):
    dest = tmp_path / "out.csv"
    written = export_query_file(
        path,
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) > 1 ORDER BY s",
        dest,
        "csv",
    )
    assert written == 1
    assert dest.read_text(encoding="utf-8") == "s,COUNT(*)\na,3\n"

    dest_jsonl = tmp_path / "out.jsonl"
    written = export_query_file(
        path,
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) > 1",
        dest_jsonl,
        "jsonl",
    )
    assert written == 1
    assert dest_jsonl.read_text(encoding="utf-8") == '{"s":"a","COUNT(*)":3}\n'


def test_cli_query_having(path, capsys):
    code = main(
        [
            "query",
            str(path),
            "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) > 1",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == [["a", 3]]


def test_cli_explain_having(path, capsys):
    code = main(
        [
            "explain",
            str(path),
            "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) > 1",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert operator_kinds(payload) == ["Scan", "Aggregate", "Having", "Project"]


def test_cli_having_validation_error_exit_2(path, capsys):
    code = main(["query", str(path), "SELECT id FROM input HAVING id > 1"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err
