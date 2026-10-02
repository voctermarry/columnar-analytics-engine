"""Tests for the HAVING post-group filter (single file and joins)."""

from __future__ import annotations

import json
import os
import subprocess
import sys

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
        ColumnSchema("g", "utf8", nullable=True),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("v", "int64", nullable=True),
        ColumnSchema("d", "float64", nullable=True),
        ColumnSchema("flag", "bool"),
    ]
)

# First-occurrence order of g is: 'b', NULL, 'a'.
DATA = {
    "g": ["b", None, "a", "b", "a", None, "b"],
    "k": [1, 1, 2, 2, 1, 2, 1],
    "v": [10, None, 30, 40, 50, 60, None],
    "d": [1.0, 2.0, None, 4.0, 5.0, None, 7.0],
    "flag": [True, False, True, False, True, False, True],
}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "grouped.caef"
    write_file(p, Table(SCHEMA, DATA))
    return p


def rows(result):
    return [
        [result._columns[c][r] for c in range(len(result.schema.columns))]
        for r in range(result.row_count)
    ]


# ---------------------------------------------------------------------------
# Basic filtering semantics
# ---------------------------------------------------------------------------


def test_having_filters_groups_by_selected_aggregate(path):
    result = query_file(
        path, "select g, count(*) from input group by g having count(*) > 2"
    )
    assert rows(result) == [["b", 3]]


def test_having_aggregate_need_not_be_selected(path):
    result = query_file(path, "select g from input group by g having sum(v) >= 60")
    # a: 30+50=80, NULL: 60, b: 10+40=50 (last v NULL)
    assert result.column("g") == [None, "a"]
    assert result.column_names == ("g",)


def test_having_grouped_column(path):
    result = query_file(
        path, "select g, count(*) from input group by g having g is not null"
    )
    assert rows(result) == [["b", 3], ["a", 2]]


def test_having_grouped_column_comparison(path):
    result = query_file(
        path, "select g from input group by g having g >= 'b'"
    )
    # The NULL group yields UNKNOWN for the comparison and is dropped.
    assert result.column("g") == ["b"]


def test_having_multiple_group_keys(path):
    result = query_file(
        path,
        "select g, k, count(*) from input group by g, k having count(*) > 1",
    )
    # Only (b,1) has two rows (positions 0 and 6).
    assert rows(result) == [["b", 1, 2]]


def test_having_keeps_first_row_group_order(path):
    result = query_file(
        path, "select g, count(*) from input group by g having count(*) >= 2"
    )
    # All three groups qualify; filtering alone does not reorder.
    assert result.column("g") == ["b", None, "a"]


def test_having_not_and_or_combine(path):
    result = query_file(
        path,
        "select g, count(*), sum(v) from input group by g "
        "having not (count(*) = 2) or g = 'b'",
    )
    # b qualifies on both sides; the NULL and a groups have count 2 and g != 'b'.
    assert result.column("g") == ["b"]


def test_having_unknown_group_drops_via_unknown(path):
    # max(v) for the NULL group is 60 (the NULL v is in row 1, but row 5 is 60);
    # instead use a group whose aggregate is genuinely NULL.
    p_table = Table(
        Schema(
            [ColumnSchema("g", "utf8"), ColumnSchema("v", "int64", nullable=True)]
        ),
        {"g": ["a", "a", "b"], "v": [1, None, None]},
    )
    import tempfile

    p = os.path.join(tempfile.mkdtemp(), "nullagg.caef")
    write_file(p, p_table)
    # A plain comparison against the NULL aggregate is UNKNOWN -> dropped.
    result = query_file(p, "select g from input group by g having max(v) > 0")
    assert result.column("g") == ["a"]
    # IS NULL keeps the UNKNOWN-aggregate group.
    result = query_file(
        p, "select g, max(v) from input group by g having max(v) is null"
    )
    assert rows(result) == [["b", None]]
    result = query_file(
        p, "select g from input group by g having max(v) is not null"
    )
    assert result.column("g") == ["a"]


def test_having_null_literal_three_valued_logic(path):
    # count(*) is never NULL; combine with an aggregate that is: the NULL
    # group's count(v) is 1, so nothing is NULL here -- use min(v) on a
    # fully-NULL group instead.
    p_table = Table(
        Schema(
            [ColumnSchema("g", "utf8"), ColumnSchema("v", "int64", nullable=True)]
        ),
        {"g": ["a", "b"], "v": [1, None]},
    )
    import tempfile

    p = os.path.join(tempfile.mkdtemp(), "tv3.caef")
    write_file(p, p_table)
    # UNKNOWN AND TRUE -> UNKNOWN (dropped); TRUE OR UNKNOWN -> TRUE (kept).
    result = query_file(
        p, "select g from input group by g having min(v) > 0 and count(*) >= 1"
    )
    assert result.column("g") == ["a"]
    result = query_file(
        p, "select g from input group by g having min(v) > 0 or g = 'b'"
    )
    assert result.column("g") == ["a", "b"]
    # NOT UNKNOWN stays UNKNOWN.
    result = query_file(
        p, "select g from input group by g having not (min(v) > 0)"
    )
    assert result.row_count == 0


# ---------------------------------------------------------------------------
# HAVING without GROUP BY (global aggregate row)
# ---------------------------------------------------------------------------


def test_having_global_aggregate_kept(path):
    result = query_file(path, "select count(*) from input having count(*) = 7")
    assert result.column("COUNT(*)") == [7]


def test_having_global_aggregate_removed(path):
    result = query_file(path, "select count(*) from input having count(*) > 99")
    assert result.row_count == 0
    assert result.column_names == ("COUNT(*)",)


def test_having_global_row_after_where_selects_none(path):
    # WHERE selects nothing: the global row (COUNT 0) is built then filtered.
    result = query_file(
        path,
        "select count(*), sum(v) from input where v > 1000 having count(*) > 0",
    )
    assert result.row_count == 0
    assert result.column_names == ("COUNT(*)", "SUM(v)")


def test_having_global_row_is_null_comparison_unknown(path):
    result = query_file(
        path,
        "select count(*) from input where v > 1000 having sum(v) is null",
    )
    assert result.column("COUNT(*)") == [0]


def test_having_grouped_empty_filter_returns_zero_groups(path):
    result = query_file(
        path,
        "select g, count(*) from input where v > 1000 group by g having count(*) >= 0",
    )
    assert result.row_count == 0
    assert result.column_names == ("g", "COUNT(*)")


# ---------------------------------------------------------------------------
# Interaction with WHERE / ORDER BY / LIMIT
# ---------------------------------------------------------------------------


def test_where_then_having(path):
    result = query_file(
        path,
        "select g, count(*) from input where v is not null "
        "group by g having count(*) = 2",
    )
    # b has v=10,40 (2); a has v=30,50 (2); the NULL group only has v=60 (1).
    assert result.column("g") == ["b", "a"]


def test_order_by_after_having(path):
    result = query_file(
        path,
        "select g, count(*) from input group by g having count(*) >= 2 "
        "order by g nulls last",
    )
    assert result.column("g") == ["a", "b", None]


def test_limit_after_having(path):
    result = query_file(
        path,
        "select g, count(*) from input group by g having count(*) >= 2 "
        "order by g nulls last limit 1",
    )
    assert rows(result) == [["a", 2]]


def test_limit_zero_after_having(path):
    result = query_file(
        path,
        "select g, count(*) from input group by g having count(*) > 0 limit 0",
    )
    assert result.row_count == 0
    assert result.column_names == ("g", "COUNT(*)")


def test_order_by_scope_not_extended_by_having(path):
    # HAVING-only aggregates are not legal ORDER BY keys.
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "select g, count(*) from input group by g having sum(v) > 1 "
            "order by sum(v)",
        )


def test_having_repeated_execution_is_stable(path):
    import contextlib
    import io

    sql = (
        "select g, count(*), sum(v) from input where g is not null "
        "group by g having sum(v) > 40 order by g"
    )
    outs = set()
    for _ in range(3):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert main(["query", str(path), sql]) == 0
        outs.add(buf.getvalue())
    assert len(outs) == 1
    payload = json.loads(next(iter(outs)))
    assert [row[0] for row in payload["rows"]] == ["a", "b"]


# ---------------------------------------------------------------------------
# Aggregate registry semantics
# ---------------------------------------------------------------------------


def test_having_aggregates_share_computation(path):
    result = query_file(
        path,
        "select g, count(*), avg(v) from input group by g "
        "having count(*) >= 2 and avg(v) > 10 order by g nulls last",
    )
    # b avg = 25.0 (10,40; last v NULL), a avg = 40.0 (30,50), the NULL
    # group avg = 60.0 (single non-null) -- all three qualify.
    assert result.column("g") == ["a", "b", None]


def test_having_min_max_types(path):
    result = query_file(
        path, "select g from input group by g having min(v) = 10 and max(d) = 7.0"
    )
    assert result.column("g") == ["b"]


def test_having_count_column_ignores_nulls(path):
    result = query_file(
        path, "select g from input group by g having count(v) = 1"
    )
    # The NULL group: v=[None,60] -> one non-null.
    assert result.column("g") == [None]


def test_having_only_aggregate_numeric_error_still_raises(tmp_path):
    p = tmp_path / "big.caef"
    write_file(
        p,
        Table(
            Schema([ColumnSchema("x", "int64")]),
            {"x": [2**63 - 1, 1]},
        ),
    )
    with pytest.raises(QueryValidationError):
        query_file(p, "select count(*) from input having sum(x) > 0")


def test_aggregation_runs_before_having_filter(tmp_path):
    # Aggregation is a distinct stage before HAVING, so a group HAVING would
    # drop still has its (overflowing) aggregate computed and must raise.
    p = tmp_path / "groups.caef"
    write_file(
        p,
        Table(
            Schema(
                [
                    ColumnSchema("g", "utf8"),
                    ColumnSchema("x", "int64"),
                ]
            ),
            {"g": ["drop", "drop", "keep"], "x": [2**63 - 1, 1, 0]},
        ),
    )
    with pytest.raises(QueryValidationError):
        query_file(
            p,
            "select g, sum(x) from input group by g having g = 'keep'",
        )


def test_having_float_aggregate_compared_to_int_literal(path):
    result = query_file(
        path, "select g from input group by g having avg(v) > 30"
    )
    # a avg 40.0, the NULL group avg 60.0; first-row group order kept.
    assert result.column("g") == [None, "a"]


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "select g, count(*) from input group by g having",
        "select count(*) from input having",
        "select g, count(*) from input group by g having count(*) > 1 having g = 'a'",
        "select g, count(*) from input order by g having count(*) > 1",
        "select g, count(*) from input group by g limit 2 having count(*) > 1",
        "select g, count(*) from input having count(*) > 1 group by g",
        "select g, count(*) from input where having count(*) > 1",
        "select g, count(*) from input group by g having count(*) > 1 "
        "order by g having g = 'a'",
        "select g, count(*) from input group by g having (count(*) > 1",
        "select g, count(*) from input group by g having count(*) >",
    ],
)
def test_having_syntax_errors(path, sql):
    with pytest.raises(QuerySyntaxError):
        query_file(path, sql)


def test_having_syntax_checked_before_file(tmp_path):
    missing = tmp_path / "nope.caef"
    with pytest.raises(QuerySyntaxError):
        query_file(missing, "select g from input group by g having")
    with pytest.raises(QuerySyntaxError):
        explain_file(missing, "select g from input group by g having count(*) >")
    with pytest.raises(QuerySyntaxError):
        explain_file(
            missing, "select g from input group by g limit 1 having count(*) > 0"
        )


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_having_in_plain_query_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select v from input having v > 1")
    with pytest.raises(QueryValidationError):
        query_file(path, "select v from input where v > 0 having v > 1")


def test_having_unknown_column(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select g from input group by g having missing > 1")
    with pytest.raises(QueryValidationError):
        query_file(path, "select g from input group by g having count(missing) > 0")


def test_having_ungrouped_column(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select g, count(*) from input group by g having v > 1")
    with pytest.raises(QueryValidationError):
        query_file(path, "select g, count(*) from input group by g having k = 1")


def test_having_ungrouped_column_without_group_by(path):
    # No GROUP BY: every plain column is ungrouped.
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(*) from input having v > 1")


def test_having_nested_aggregate_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(
            path, "select g from input group by g having sum(count(*)) > 1"
        )
    with pytest.raises(QueryValidationError):
        query_file(
            path, "select g from input group by g having count(sum(v)) > 0"
        )


def test_having_scalar_expression_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(
            path, "select g from input group by g having count(*) + 1 > 2"
        )
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "select g from input group by g "
            "having case when g = 'a' then 1 else 0 end = 1",
        )
    with pytest.raises(QueryValidationError):
        query_file(path, "select g from input group by g having -count(*) < 0")


def test_having_star_arguments(path):
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select g from input group by g having sum(*) > 1")
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select g from input group by g having avg(*) > 1")


def test_having_illegal_aggregate_argument(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select g from input group by g having sum(g) > 1")
    with pytest.raises(QueryValidationError):
        query_file(path, "select g from input group by g having avg(g) > 1")


def test_having_incompatible_types(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select g from input group by g having count(*) = 'x'")
    with pytest.raises(QueryValidationError):
        query_file(path, "select g from input group by g having g = 1")
    with pytest.raises(QueryValidationError):
        query_file(path, "select g from input group by g having max(flag) = 1")
    with pytest.raises(QueryValidationError):
        query_file(
            path, "select g from input group by g having count(*) > '1'"
        )


def test_having_bool_only_equality(path):
    with pytest.raises(QueryValidationError):
        query_file(
            path, "select g from input group by g having max(flag) > false"
        )
    result = query_file(
        path,
        "select g from input group by g having max(flag) = true",
    )
    # Groups containing at least one True: b (T,F,T), a (T,T); NULL (F,F) out.
    assert result.column("g") == ["b", "a"]


def test_having_final_condition_must_be_bool(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select g from input group by g having count(*)")
    with pytest.raises(QueryValidationError):
        query_file(path, "select g from input group by g having g")
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(*) from input having 3")


def test_having_does_not_accept_select_alias(path):
    # Aggregate labels are aggregate calls in HAVING, never aliases; a plain
    # unknown name is a validation error.
    with pytest.raises(QueryValidationError):
        query_file(
            path, "select g, count(*) as c from input group by g having c > 1"
        )


# ---------------------------------------------------------------------------
# EXPLAIN
# ---------------------------------------------------------------------------


def test_explain_having_operator_position(path):
    plan = explain_file(
        path,
        "select g, count(*) from input group by g having count(*) > 2 "
        "order by g limit 1",
    )
    assert [op["operator"] for op in plan["operators"]] == [
        "Scan",
        "Aggregate",
        "Having",
        "Sort",
        "Limit",
        "Project",
    ]


def test_explain_having_condition_tree(path):
    plan = explain_file(
        path,
        "select g from input group by g having not (count(*) = 2) or g is null",
    )
    having = next(
        op for op in plan["operators"] if op["operator"] == "Having"
    )["condition"]
    assert having["kind"] == "logic"
    assert having["operator"] == "OR"
    left, right = having["operands"]
    assert left["kind"] == "not"
    cmp_node = left["operands"][0]
    assert cmp_node["kind"] == "comparison"
    assert cmp_node["operator"] == "="
    agg_leaf = cmp_node["operands"][0]
    assert agg_leaf == {
        "kind": "aggregate",
        "function": "COUNT",
        "argument": None,
        "type": "int64",
        "nullable": False,
    }
    assert cmp_node["operands"][1] == {
        "kind": "literal",
        "type": "int64",
        "value": 2,
    }
    assert right["kind"] == "is_null"
    assert right["operands"][0] == {"kind": "column", "name": "g"}


def test_explain_having_aggregate_leaf_types(path):
    plan = explain_file(
        path,
        "select g from input group by g "
        "having sum(v) > 1 and avg(d) > 0.5 and min(g) is not null",
    )
    having = next(
        op for op in plan["operators"] if op["operator"] == "Having"
    )["condition"]
    sum_and, min_is = having["operands"]
    cmp_sum, cmp_avg = sum_and["operands"]
    assert cmp_sum["operands"][0] == {
        "kind": "aggregate",
        "function": "SUM",
        "argument": "v",
        "type": "int64",
        "nullable": True,
    }
    assert cmp_avg["operands"][0] == {
        "kind": "aggregate",
        "function": "AVG",
        "argument": "d",
        "type": "float64",
        "nullable": True,
    }
    assert min_is["operands"][0] == {
        "kind": "aggregate",
        "function": "MIN",
        "argument": "g",
        "type": "utf8",
        "nullable": True,
    }


def test_explain_aggregates_union_dedup_first_reference(path):
    plan = explain_file(
        path,
        "select g, sum(v) from input group by g "
        "having sum(v) > 1 and count(*) > 0 order by g",
    )
    agg = next(op for op in plan["operators"] if op["operator"] == "Aggregate")
    # SELECT's SUM(v) is first; HAVING adds COUNT(*) but does not duplicate SUM.
    assert agg["aggregates"] == [
        {"function": "SUM", "argument": "v", "output": "SUM(v)"},
        {"function": "COUNT", "argument": None, "output": "COUNT(*)"},
    ]


def test_explain_aggregates_include_order_by_after_having(path):
    plan = explain_file(
        path,
        "select g, count(*) from input group by g "
        "having sum(v) > 0 order by count(*) desc",
    )
    agg = next(op for op in plan["operators"] if op["operator"] == "Aggregate")
    assert [a["output"] for a in agg["aggregates"]] == ["COUNT(*)", "SUM(v)"]


def test_explain_scan_required_columns_cover_having(path):
    plan = explain_file(
        path, "select g from input group by g having sum(v) > 1 and max(d) > 0"
    )
    scan = next(op for op in plan["operators"] if op["operator"] == "Scan")
    # g is the group key; v and d are HAVING-only aggregate arguments.
    assert scan["required_columns"] == ["g", "v", "d"]


def test_explain_scan_covers_having_group_column_without_projection(path):
    plan = explain_file(
        path, "select count(*) from input group by k having k is not null"
    )
    scan = next(op for op in plan["operators"] if op["operator"] == "Scan")
    assert scan["required_columns"] == ["k"]


def test_explain_output_only_describes_select(path):
    plan = explain_file(path, "select g from input group by g having sum(v) > 1")
    assert plan["output"] == [
        {"name": "g", "type": "utf8", "nullable": True}
    ]


def test_explain_no_having_stage_without_clause(path):
    plan = explain_file(path, "select g, count(*) from input group by g")
    assert "Having" not in [op["operator"] for op in plan["operators"]]


def test_explain_having_matches_result_schema(path):
    sql = (
        "select g, count(*), sum(v) from input group by g "
        "having count(*) > 1 and sum(v) > 40 order by g nulls last"
    )
    plan = explain_file(path, sql)
    result = query_file(path, sql)
    assert plan["output"] == [
        {"name": col.name, "type": col.type, "nullable": col.nullable}
        for col in result.schema.columns
    ]


def test_explain_global_having(path):
    plan = explain_file(path, "select count(*) from input having count(*) > 0")
    kinds = [op["operator"] for op in plan["operators"]]
    assert kinds == ["Scan", "Aggregate", "Having", "Project"]
    agg = next(op for op in plan["operators"] if op["operator"] == "Aggregate")
    assert agg["group_keys"] == []


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------


LEFT_SCHEMA = Schema(
    [ColumnSchema("id", "int64"), ColumnSchema("s", "utf8", nullable=True)]
)
LEFT_DATA = {"id": [1, 2, 3, 4], "s": ["a", "b", "a", None]}
RIGHT_SCHEMA = Schema(
    [
        ColumnSchema("rid", "int64"),
        ColumnSchema("k", "int64"),
        ColumnSchema("n", "int64", nullable=True),
    ]
)
RIGHT_DATA = {"rid": [10, 20, 30], "k": [1, 2, 2], "n": [5, None, 7]}


@pytest.fixture()
def sources(tmp_path):
    left = tmp_path / "l.caef"
    right = tmp_path / "r.caef"
    write_file(left, Table(LEFT_SCHEMA, LEFT_DATA))
    write_file(right, Table(RIGHT_SCHEMA, RIGHT_DATA))
    return {"l": left, "r": right}


def test_join_having_qualified(sources):
    result = query_files(
        sources,
        "select l.s from l inner join r on l.id = r.k "
        "where l.s is not null group by l.s having sum(r.n) > 5",
    )
    # Inner join rows: (1,a,n=5), (2,b,n=NULL), (2,b,n=7).  Group a sums to
    # 5 (not strictly greater); group b sums to 7 (NULL ignored) and stays.
    assert result.column("l.s") == ["b"]


def test_join_having_qualified_group_column(sources):
    result = query_files(
        sources,
        "select l.s, count(*) from l left join r on l.id = r.k "
        "group by l.s having l.s is not null order by l.s",
    )
    # a: the matched id=1 row plus the unmatched id=3 row (2); b: two matches
    # on k=2 (2); the NULL-s unmatched left row is removed by HAVING.
    assert rows(result) == [["a", 2], ["b", 2]]


def test_join_explain_having(sources):
    plan = explain_files(
        sources,
        "select l.s from l inner join r on l.id = r.k "
        "group by l.s having count(*) > 1 and max(r.n) > 1",
    )
    assert [op["operator"] for op in plan["operators"]] == [
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
    assert scans == {"l": ["id", "s"], "r": ["k", "n"]}
    having = next(
        op for op in plan["operators"] if op["operator"] == "Having"
    )["condition"]
    and_node = having
    count_cmp, max_cmp = and_node["operands"]
    assert count_cmp["operands"][0]["function"] == "COUNT"
    assert max_cmp["operands"][0]["argument"] == "r.n"


def test_join_having_unqualified_column_rejected(sources):
    with pytest.raises(QueryValidationError):
        query_files(
            sources,
            "select l.s from l inner join r on l.id = r.k "
            "group by l.s having count(*) > 1 and s = 'a'",
        )


def test_join_having_unknown_column(sources):
    with pytest.raises(QueryValidationError):
        query_files(
            sources,
            "select l.s from l inner join r on l.id = r.k "
            "group by l.s having sum(r.missing) > 0",
        )


def test_join_having_syntax_error_before_files(tmp_path):
    missing = tmp_path / "nope.caef"
    with pytest.raises(QuerySyntaxError):
        explain_files(
            {"l": missing, "r": missing},
            "select l.s from l inner join r on l.id = r.k "
            "group by l.s having",
        )


# ---------------------------------------------------------------------------
# Export and CLI
# ---------------------------------------------------------------------------


def test_export_csv_reflects_having(path, tmp_path):
    dest = tmp_path / "out.csv"
    n = export_query_file(
        path,
        "select g, count(*) from input group by g having count(*) > 2",
        dest,
    )
    assert n == 1
    assert dest.read_text(encoding="utf-8") == "g,COUNT(*)\nb,3\n"


def test_export_jsonl_reflects_having(path, tmp_path):
    dest = tmp_path / "out.jsonl"
    n = export_query_file(
        path,
        "select g, count(*) from input group by g having count(*) > 2",
        dest,
        "jsonl",
    )
    assert n == 1
    assert dest.read_text(encoding="utf-8") == '{"g":"b","COUNT(*)":3}\n'


def test_cli_having(path, capsys):
    code = main(
        [
            "query",
            str(path),
            "select g, count(*) from input group by g having count(*) > 2",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == [["b", 3]]


def test_cli_having_syntax_error_exit_2(path, capsys):
    code = main(
        [
            "query",
            str(path),
            "select g from input group by g having",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_having_validation_error_exit_2(path, capsys):
    code = main(
        [
            "query",
            str(path),
            "select g from input group by g having v > 1",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""


def test_cli_explain_having(path, capsys):
    code = main(
        [
            "explain",
            str(path),
            "select g, count(*) from input group by g having count(*) > 2",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [op["operator"] for op in payload["operators"]] == [
        "Scan",
        "Aggregate",
        "Having",
        "Project",
    ]


def test_cli_having_subprocess(path):
    env = dict(
        os.environ,
        PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "columnar_analytics.cli",
            "query",
            str(path),
            "select g, count(*) from input group by g having count(*) > 2",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["rows"] == [["b", 3]]
