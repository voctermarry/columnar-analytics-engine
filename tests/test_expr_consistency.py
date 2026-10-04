"""Consistency of one bound expression across every stage.

The refactor makes a single bound expression the source of truth for type,
nullability, referenced columns, plan rendering and execution.  These tests
exercise one deeply nested expression -- column references, typed literals,
unary and binary arithmetic, comparisons, AND / OR / NOT, IS [NOT] NULL and
recursively nested searched CASE -- and assert that single-table and join
queries agree on:

* the static result type and output nullability (explain vs executed schema),
* the referenced source columns (required_columns in source schema order,
  including columns appearing only inside lazily evaluated CASE branches),
* the explain condition / projection tree,
* the executed values (three-valued logic, short circuit, CASE laziness),
* and the classification of failures (QuerySyntaxError for lexical /
  grammatical problems, QueryValidationError for unknown columns, type
  incompatibility and runtime arithmetic errors -- raised by execution but
  never by the metadata-only explain).

Only the public entry points are used, so the tests pin the behaviour the
refactored internals must preserve rather than any internal node shape.
"""

from __future__ import annotations

import pytest

from columnar_analytics import (
    ColumnSchema,
    QuerySyntaxError,
    QueryValidationError,
    Schema,
    Table,
    explain_file,
    explain_files,
    query_file,
    query_files,
    write_file,
    write_partitioned_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


LEFT_SCHEMA = Schema(
    [
        ColumnSchema("k", "int64"),
        ColumnSchema("a", "int64"),
        ColumnSchema("b", "int64", nullable=True),
        ColumnSchema("c", "float64", nullable=True),
        ColumnSchema("d", "int64"),
        ColumnSchema("s", "utf8", nullable=True),
    ]
)

# Rows exercise every route of the nested expression: NULL operands, both
# WHEN branches, the ELSE branch and int64/float64 mixing.
#   a=1 b=10  c= 1.5 d=0  -> outer WHEN TRUE  (-d=0<=0), inner a=1 -> b/2 = 5.0
#   a=2 b=NULL c= 2.5 d=7  -> outer WHEN FALSE, ELSE a-b -> NULL
#   a=3 b=30  c=NULL d=5  -> outer WHEN TRUE  (-d=-5<=0 despite NULL compare),
#                            inner a>2 -> c * -1.5 -> NULL
#   a=4 b=40  c=-0.5 d=1  -> outer WHEN TRUE,  inner a>2 -> c*-1.5 = 0.75
LEFT_DATA = {
    "k": [1, 2, 3, 4],
    "a": [1, 2, 3, 4],
    "b": [10, None, 30, 40],
    "c": [1.5, 2.5, None, -0.5],
    "d": [0, 7, 5, 1],
    "s": ["x", None, "y", "z"],
}

EXPECTED_NESTED = [5.0, None, None, 0.75]

RIGHT_SCHEMA = Schema(
    [
        ColumnSchema("k", "int64"),
        ColumnSchema("t", "utf8", nullable=True),
    ]
)

# k=1 matches twice (duplicate keys), k=2 matches once with a NULL t,
# k=3/k=4 have no match (LEFT JOIN pads them).
RIGHT_DATA = {"k": [1, 1, 2], "t": ["p", "q", None]}

# A deeply nested scalar expression covering every expression kind.  Columns
# appear inside every branch on purpose: dependency collection must find
# a/b/c/d even though evaluation only ever visits the chosen CASE branch.
NESTED_EXPR = (
    "CASE "
    "WHEN NOT (b IS NULL) AND (b + c > 1 OR -d <= 0) "
    "THEN CASE WHEN a = 1 THEN b / 2 WHEN a > 2 THEN c * -1.5 ELSE 0 END "
    "ELSE CASE WHEN d IS NOT NULL THEN a - b END "
    "END"
)

# Same expression with every column qualified, for join queries.
NESTED_EXPR_QUALIFIED = (
    "CASE "
    "WHEN NOT (l.b IS NULL) AND (l.b + l.c > 1 OR -l.d <= 0) "
    "THEN CASE WHEN l.a = 1 THEN l.b / 2 WHEN l.a > 2 THEN l.c * -1.5 ELSE 0 END "
    "ELSE CASE WHEN l.d IS NOT NULL THEN l.a - l.b END "
    "END"
)


@pytest.fixture()
def left_path(tmp_path):
    p = tmp_path / "l.caef"
    write_file(p, Table(LEFT_SCHEMA, LEFT_DATA))
    return p


@pytest.fixture()
def sources(tmp_path, left_path):
    right = tmp_path / "r.caef"
    write_file(right, Table(RIGHT_SCHEMA, RIGHT_DATA))
    return {"l": left_path, "r": right}


def describe(table, name):
    col = next(c for c in table.schema.columns if c.name == name)
    return col.type, col.nullable


# ---------------------------------------------------------------------------
# Executed values are identical under single-table and join expression rules
# ---------------------------------------------------------------------------


def test_nested_expression_values_single_table(left_path):
    result = query_file(
        left_path, f"SELECT {NESTED_EXPR} AS v FROM input ORDER BY a"
    )
    assert describe(result, "v") == ("float64", True)
    assert result.column("v") == EXPECTED_NESTED


def test_nested_expression_values_inner_join(sources):
    # INNER JOIN emits a=1 twice (right rows t 'p','q') and a=2 once
    # (t NULL); the expression never references r.
    result = query_files(
        sources,
        f"SELECT {NESTED_EXPR_QUALIFIED} AS v FROM l "
        "INNER JOIN r ON l.k = r.k ORDER BY l.k, r.t",
    )
    assert describe(result, "v") == ("float64", True)
    assert result.column("v") == [5.0, 5.0, None]


def test_nested_expression_values_left_join_pads_unmatched(sources):
    # The LEFT JOIN adds the unmatched l rows a=3, a=4 after the matched
    # expansion; the shared expression still applies the same rules.
    result = query_files(
        sources,
        f"SELECT l.a, {NESTED_EXPR_QUALIFIED} AS v FROM l "
        "LEFT JOIN r ON l.k = r.k ORDER BY l.a, r.t",
    )
    pairs = list(zip(result.column("l.a"), result.column("v")))
    # a=1 twice (5.0/5.0), a=2 once (NULL), a=3 (NULL), a=4 (0.75).
    assert pairs == [(1, 5.0), (1, 5.0), (2, None), (3, None), (4, 0.75)]


def test_nested_expression_order_by_alias_nulls_first(left_path, sources):
    single = query_file(
        left_path, f"SELECT {NESTED_EXPR} AS v FROM input ORDER BY v NULLS FIRST"
    )
    joined = query_files(
        sources,
        f"SELECT {NESTED_EXPR_QUALIFIED} AS v FROM l "
        "INNER JOIN r ON l.k = r.k ORDER BY v NULLS FIRST",
    )
    assert single.column("v")[0] is None
    assert joined.column("v")[0] is None
    assert single.column("v") == [None, None, 0.75, 5.0]
    assert joined.column("v") == [None, 5.0, 5.0]


# ---------------------------------------------------------------------------
# Type / nullability consistency: explain plan == executed result schema
# ---------------------------------------------------------------------------


def test_type_and_nullability_consistent_explain_vs_execution_single(left_path):
    sql = f"SELECT {NESTED_EXPR} AS v FROM input"
    plan = explain_file(left_path, sql)
    result = query_file(left_path, sql)
    assert plan["output"] == [
        {"name": c.name, "type": c.type, "nullable": c.nullable}
        for c in result.schema.columns
    ]
    assert plan["output"][0] == {"name": "v", "type": "float64", "nullable": True}


def test_type_and_nullability_consistent_explain_vs_execution_join(sources):
    sql = (
        f"SELECT {NESTED_EXPR_QUALIFIED} AS v FROM l "
        "INNER JOIN r ON l.k = r.k"
    )
    plan = explain_files(sources, sql)
    result = query_files(sources, sql)
    assert plan["output"] == [
        {"name": c.name, "type": c.type, "nullable": c.nullable}
        for c in result.schema.columns
    ]
    assert plan["output"][0] == {"name": "v", "type": "float64", "nullable": True}


def test_pure_constant_nested_case_is_not_nullable(left_path):
    # Every reachable result is a literal and ELSE is present: a nullable
    # column appearing only in conditions routes rows but never makes the
    # value NULL.
    sql = "SELECT CASE WHEN b IS NULL THEN 1 WHEN b > 0 THEN 2 ELSE 3 END AS v FROM input"
    plan = explain_file(left_path, sql)
    assert plan["output"][0] == {"name": "v", "type": "int64", "nullable": False}


@pytest.mark.parametrize(
    "expr,nullable",
    [
        # Nullable THEN result; explicit ELSE constant still nullable.
        ("CASE WHEN a = 1 THEN b ELSE 1 END AS v", True),
        # Missing ELSE implies a NULL result.
        ("CASE WHEN a = 1 THEN 1 END AS v", True),
        # Both routes non-NULL literals -> not nullable.
        ("CASE WHEN a = 1 THEN 0 ELSE 1 END AS v", False),
    ],
)
def test_case_nullability_single(left_path, expr, nullable):
    plan = explain_file(left_path, f"SELECT {expr} FROM input")
    assert plan["output"][0]["nullable"] is nullable


def test_case_nullability_join(sources):
    cases_join = [
        (
            "SELECT CASE WHEN l.a = 1 THEN l.b ELSE 1 END AS v FROM l "
            "INNER JOIN r ON l.k = r.k",
            True,
        ),
        (
            "SELECT CASE WHEN r.k IS NULL THEN 0 ELSE 1 END AS v "
            "FROM l LEFT JOIN r ON l.k = r.k",
            False,
        ),
    ]
    for sql, nullable in cases_join:
        plan = explain_files(sources, sql)
        assert plan["output"][0]["nullable"] is nullable, sql


def test_int64_float64_case_unifies_to_float64(left_path):
    plan = explain_file(
        left_path,
        "SELECT CASE WHEN a = 1 THEN b WHEN a = 2 THEN c ELSE 0 END AS v FROM input",
    )
    assert plan["output"][0]["type"] == "float64"


# ---------------------------------------------------------------------------
# Referenced columns: one traversal, source schema order, lazy branches included
# ---------------------------------------------------------------------------


def test_required_columns_cover_every_lazy_branch_single(left_path):
    # a/b/c/d appear only inside the nested CASE's branches and conditions;
    # static collection must still read all of them, in schema order.
    sql = f"SELECT {NESTED_EXPR} AS v FROM input"
    scan = explain_file(left_path, sql)["operators"][0]
    assert scan["required_columns"] == ["a", "b", "c", "d"]


def test_required_columns_union_where_projection_and_order_by(left_path):
    sql = f"SELECT {NESTED_EXPR} AS v FROM input WHERE s IS NOT NULL ORDER BY v"
    scan = explain_file(left_path, sql)["operators"][0]
    assert scan["required_columns"] == ["a", "b", "c", "d", "s"]


def test_required_columns_attributed_per_join_source(sources):
    # Expression on l plus a WHERE leaf on r: each scan lists only its own
    # referenced columns, in its local schema order (ON keys included).
    plan = explain_files(
        sources,
        f"SELECT {NESTED_EXPR_QUALIFIED} AS v FROM l "
        "INNER JOIN r ON l.k = r.k WHERE r.t IS NOT NULL",
    )
    scans = {
        op["source"]: op["required_columns"]
        for op in plan["operators"]
        if op["operator"] == "Scan"
    }
    assert scans == {"l": ["k", "a", "b", "c", "d"], "r": ["k", "t"]}


# ---------------------------------------------------------------------------
# Explain condition tree mirrors the executed predicate
# ---------------------------------------------------------------------------


def test_explain_condition_tree_shape_single_and_join(left_path, sources):
    where = "NOT (b IS NULL) AND (a = 1 OR d <= 0)"
    where_q = "NOT (l.b IS NULL) AND (l.a = 1 OR l.d <= 0)"
    single = explain_file(left_path, f"SELECT a FROM input WHERE {where}")
    joined = explain_files(
        sources,
        f"SELECT l.a FROM l INNER JOIN r ON l.k = r.k WHERE {where_q}",
    )
    cond = next(op["condition"] for op in single["operators"] if op["operator"] == "Filter")
    jcond = next(
        op["condition"] for op in joined["operators"] if op["operator"] == "Filter"
    )
    assert cond["operator"] == "AND"
    left_leaf, right_leaf = cond["operands"]
    assert left_leaf["kind"] == "not"
    assert left_leaf["operands"][0] == {
        "kind": "is_null",
        "operator": "IS NULL",
        "operands": [{"kind": "column", "name": "b"}],
    }
    assert right_leaf["kind"] == "logic" and right_leaf["operator"] == "OR"
    cmp_a = right_leaf["operands"][0]
    assert cmp_a["operands"][0] == {"kind": "column", "name": "a"}
    assert cmp_a["operands"][1] == {"kind": "literal", "type": "int64", "value": 1}
    # The join tree is the same tree with qualified column names.
    assert jcond["operands"][0]["operands"][0]["operands"][0]["name"] == "l.b"
    assert jcond["operands"][1]["operands"][0]["operands"][0]["name"] == "l.a"
    # And it selects the same left rows (only a=1 survives).
    single_rows = query_file(
        left_path, f"SELECT a FROM input WHERE {where} ORDER BY a"
    ).column("a")
    join_rows = query_files(
        sources,
        f"SELECT l.a FROM l INNER JOIN r ON l.k = r.k WHERE {where_q} "
        "ORDER BY l.a, r.t",
    ).column("l.a")
    assert set(single_rows) == set(join_rows) == {1}


def test_explain_case_tree_in_projection(left_path):
    plan = explain_file(
        left_path,
        "SELECT CASE WHEN a = 1 THEN b WHEN a = 2 THEN 1 END AS v FROM input",
    )
    tree = next(op for op in plan["operators"] if op["operator"] == "Project")[
        "expressions"
    ][0]["expression"]
    assert tree["kind"] == "case"
    assert [entry["when"]["operator"] for entry in tree["cases"]] == ["=", "="]
    assert tree["else"] is None


# ---------------------------------------------------------------------------
# Exception classification stays identical single-table vs join
# ---------------------------------------------------------------------------


def test_unknown_column_is_validation_error_everywhere(left_path, sources):
    bad_single = "SELECT CASE WHEN zzz IS NULL THEN 1 ELSE 2 END AS v FROM input"
    bad_join = (
        "SELECT CASE WHEN l.zzz IS NULL THEN 1 ELSE 2 END AS v FROM l "
        "INNER JOIN r ON l.k = r.k"
    )
    with pytest.raises(QueryValidationError):
        query_file(left_path, bad_single)
    with pytest.raises(QueryValidationError):
        explain_file(left_path, bad_single)
    with pytest.raises(QueryValidationError):
        query_files(sources, bad_join)
    with pytest.raises(QueryValidationError):
        explain_files(sources, bad_join)


def test_unqualified_column_in_join_is_validation_error(sources):
    sql = (
        f"SELECT {NESTED_EXPR.replace('l.', '')} AS v FROM l "
        "INNER JOIN r ON l.k = r.k"
    )
    with pytest.raises(QueryValidationError):
        query_files(sources, sql)
    with pytest.raises(QueryValidationError):
        explain_files(sources, sql)


def test_type_incompatibility_is_validation_error_single_and_join(left_path, sources):
    bad_single = (
        "SELECT CASE WHEN a = 1 THEN b ELSE c END AS v FROM input WHERE s = 1"
    )
    bad_join = (
        "SELECT CASE WHEN l.a = 1 THEN l.b ELSE l.c END AS v FROM l "
        "INNER JOIN r ON l.k = r.k WHERE l.s = 1"
    )
    with pytest.raises(QueryValidationError):
        query_file(left_path, bad_single)
    with pytest.raises(QueryValidationError):
        explain_file(left_path, bad_single)
    with pytest.raises(QueryValidationError):
        query_files(sources, bad_join)
    with pytest.raises(QueryValidationError):
        explain_files(sources, bad_join)


def test_case_branch_type_conflict_is_validation_error(left_path):
    with pytest.raises(QueryValidationError):
        query_file(
            left_path,
            "SELECT CASE WHEN a = 1 THEN b WHEN a = 2 THEN s END AS v FROM input",
        )


def test_numeric_nested_expression_as_boolean_rejected(left_path):
    with pytest.raises(QueryValidationError):
        query_file(left_path, f"SELECT a FROM input WHERE {NESTED_EXPR}")
    with pytest.raises(QueryValidationError):
        query_file(left_path, "SELECT a FROM input WHERE NOT (a + 1)")


def test_syntax_error_before_file_access():
    missing = "/nonexistent/t.caef"
    with pytest.raises(QuerySyntaxError):
        query_file(missing, "SELECT CASE WHEN a = 1 THEN 1 AS v FROM input")
    with pytest.raises(QuerySyntaxError):
        query_file(missing, "SELECT a + FROM input")
    with pytest.raises(QuerySyntaxError):
        query_file(missing, "SELECT (a = 1 FROM input")
    with pytest.raises(QuerySyntaxError):
        query_file(
            {"l": missing, "r": "/nonexistent/r.caef"},
            "SELECT CASE WHEN l.a = 1 THEN 1 END v FROM l "
            "INNER JOIN r ON l.k = r.k",
        )
    with pytest.raises(QuerySyntaxError):
        query_file(
            {"l": missing, "r": "/nonexistent/r.caef"},
            "SELECT l.k FROM l INNER JOIN r",
        )


# ---------------------------------------------------------------------------
# Runtime arithmetic errors: explain never evaluates, execution raises QVE;
# CASE laziness makes errors in unhit branches unreachable.
# ---------------------------------------------------------------------------


BIG_SCHEMA = Schema([ColumnSchema("x", "int64"), ColumnSchema("g", "int64")])


@pytest.fixture()
def big_path(tmp_path):
    p = tmp_path / "big.caef"
    write_file(p, Table(BIG_SCHEMA, {"x": [1, 2**63 - 1, 0], "g": [1, 2, 3]}))
    return p


def test_explain_does_not_evaluate_runtime_arithmetic(big_path):
    # 1 / (x - x) divides by zero on every row: execution fails, explain
    # succeeds metadata-only and still reports the static type/nullable.
    sql = "SELECT 1 / (x - x) AS z FROM input"
    plan = explain_file(big_path, sql)
    assert plan["output"][0]["type"] == "float64"
    with pytest.raises(QueryValidationError):
        query_file(big_path, sql)


def test_hit_case_branch_error_raises_single_and_join(tmp_path):
    p = tmp_path / "l.caef"
    write_file(
        p,
        Table(
            Schema([ColumnSchema("k", "int64"), ColumnSchema("x", "int64")]),
            {"k": [1], "x": [0]},
        ),
    )
    q = tmp_path / "r.caef"
    write_file(q, Table(Schema([ColumnSchema("k", "int64")]), {"k": [1]}))
    sources = {"l": p, "r": q}
    single_sql = "SELECT CASE WHEN x = 0 THEN 1 / x ELSE 0 END AS z FROM input"
    join_sql = (
        "SELECT CASE WHEN l.x = 0 THEN 1 / l.x ELSE 0 END AS z FROM l "
        "INNER JOIN r ON l.k = r.k"
    )
    with pytest.raises(QueryValidationError):
        query_file(p, single_sql)
    with pytest.raises(QueryValidationError):
        query_files(sources, join_sql)
    # explain is metadata-only (no evaluation) and the static result is
    # non-nullable: both the constant division and the ELSE are constants.
    assert explain_file(p, single_sql)["output"][0]["nullable"] is False
    assert explain_files(sources, join_sql)["output"][0]["nullable"] is False


def test_unhit_case_branch_error_never_raises(big_path):
    # x=1 -> x+1 = 2; x=MAX -> 0 (the x+1 overflow branch is unhit); the
    # WHERE x >= 1 also removes x=0, so the ELSE 1/x branch is unhit too.
    result = query_file(
        big_path,
        "SELECT CASE WHEN x = 1 THEN x + 1 WHEN x > 1 THEN 0 ELSE 1 / x END AS z "
        "FROM input WHERE x >= 1 ORDER BY g",
    )
    assert result.column("z") == [2, 0]


def test_int64_overflow_in_nested_arithmetic_is_validation_error(big_path):
    with pytest.raises(QueryValidationError):
        query_file(
            big_path,
            "SELECT CASE WHEN x > 1 THEN x + 1 ELSE 0 END AS z FROM input",
        )


def test_non_finite_float_in_case_is_validation_error(tmp_path):
    p = tmp_path / "f.caef"
    write_file(p, Table(Schema([ColumnSchema("f", "float64")]), {"f": [1e308]}))
    with pytest.raises(QueryValidationError):
        query_file(
            p,
            "SELECT CASE WHEN f > 0 THEN f * 10 ELSE 0.0 END AS z FROM input",
        )


# ---------------------------------------------------------------------------
# Nullability propagation across an outer join reaches the shared expression
# ---------------------------------------------------------------------------


def test_left_join_nullability_propagates_through_nested_expression(sources):
    # l.k=4 has no right match, so under LEFT JOIN r.k is padded with NULL
    # even though it is non-nullable at the source.
    sql = (
        "SELECT CASE WHEN r.k IS NULL THEN 0 ELSE r.k + 1 END AS v, "
        "r.k + 1 AS w FROM l LEFT JOIN r ON l.k = r.k ORDER BY l.a, r.t"
    )
    plan = explain_files(sources, sql)
    result = query_files(sources, sql)
    assert plan["output"] == [
        {"name": c.name, "type": c.type, "nullable": c.nullable}
        for c in result.schema.columns
    ]
    out = {c["name"]: c for c in plan["output"]}
    assert out["w"]["nullable"] is True
    # The CASE's chosen ELSE result (r.k+1) is nullable, so v is too.
    assert out["v"]["nullable"] is True
    # Unmatched row a=4 sorts last: its CASE yields 0, the raw r.k yields NULL.
    assert result.column("v")[-1] == 0
    assert result.column("w")[-1] is None


def test_case_covering_padded_null_with_literals_is_not_nullable(sources):
    # The padded r.k is tested with IS NULL and both routes yield literals,
    # so the result can never be NULL.
    plan = explain_files(
        sources,
        "SELECT CASE WHEN r.k IS NULL THEN 0 ELSE 1 END AS v "
        "FROM l LEFT JOIN r ON l.k = r.k",
    )
    assert plan["output"][0] == {"name": "v", "type": "int64", "nullable": False}


# ---------------------------------------------------------------------------
# The same rules hold against a v2 (row-group-partitioned) source
# ---------------------------------------------------------------------------


def test_nested_expression_consistent_on_partitioned_source(tmp_path, left_path):
    p2 = tmp_path / "p2.caef"
    write_partitioned_file(p2, Table(LEFT_SCHEMA, LEFT_DATA), row_group_size=2)
    sql = f"SELECT {NESTED_EXPR} AS v FROM input ORDER BY a"
    v1_result = query_file(left_path, sql)
    v2_result = query_file(p2, sql)
    assert [
        (c.type, c.nullable) for c in v2_result.schema.columns
    ] == [(c.type, c.nullable) for c in v1_result.schema.columns]
    assert v2_result.column("v") == v1_result.column("v") == EXPECTED_NESTED
    plan = explain_file(p2, sql)
    assert plan["output"][0] == {"name": "v", "type": "float64", "nullable": True}
    assert plan["operators"][0]["required_columns"] == ["a", "b", "c", "d"]
