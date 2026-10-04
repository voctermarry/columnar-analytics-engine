"""Consistency of one bound expression across every stage.

The expression refactor makes a single bound expression tree the shared
fact for binding, column-dependency collection, explain rendering and
execution.  These tests pin the property the refactor exists to guarantee:
the *same* nested expression -- mixing column references, typed literals,
unary/binary arithmetic, comparisons, AND/OR/NOT, IS [NOT] NULL and nested
searched CASE -- reports identical static type, nullability and referenced
columns and yields identical rows and exception classifications whether it
runs against a single table or an equivalent join, under both execution and
``explain`` (which never evaluates it).
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


# Single-table source -------------------------------------------------------

SINGLE_SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("f", "float64", nullable=True),
        ColumnSchema("s", "utf8", nullable=True),
        ColumnSchema("flag", "bool", nullable=True),
    ]
)
SINGLE_DATA = {
    "id": [1, 2, 3, 4, 5],
    "n": [10, None, 30, 40, None],
    "f": [1.5, 2.5, None, -0.5, 10.0],
    "s": ["a", "b", None, "a", "c"],
    "flag": [True, False, None, True, False],
}

# A deep expression exercising every supported construct at once:
#   CASE WHEN n IS NULL THEN 0
#        WHEN NOT (f > 1.0 AND id != 2) THEN -n
#        ELSE n + f END * 2 + 1
NESTED_EXPR = (
    "(CASE WHEN n IS NULL THEN 0 "
    "WHEN NOT (f > 1.0 AND id != 2) THEN -n "
    "ELSE n + f END) * 2 + 1"
)
# Static facts: int64/float64 results unify to float64; a nullable THEN/ELSE
# result makes the value nullable; the referenced columns are n, f, id.
EXPR_TYPE = "float64"
EXPR_NULLABLE = True
EXPR_COLUMNS = ("id", "n", "f")


@pytest.fixture()
def single_path(tmp_path):
    p = tmp_path / "single.caef"
    write_file(p, Table(SINGLE_SCHEMA, SINGLE_DATA))
    return p


# Equivalent 1:1 INNER JOIN source (rid matches id) -------------------------

LEFT_SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("name", "utf8", nullable=True),
    ]
)
LEFT_DATA = {
    "id": [1, 2, 3, 4, 5],
    "n": [10, None, 30, 40, None],
    "name": ["a", "b", "c", "d", "e"],
}
RIGHT_SCHEMA = Schema(
    [
        ColumnSchema("rid", "int64"),
        ColumnSchema("f", "float64", nullable=True),
        ColumnSchema("s", "utf8", nullable=True),
    ]
)
RIGHT_DATA = {
    "rid": [1, 2, 3, 4, 5],
    "f": [1.5, 2.5, None, -0.5, 10.0],
    "s": ["a", "b", None, "a", "c"],
}


@pytest.fixture()
def join_paths(tmp_path):
    left = tmp_path / "l.caef"
    right = tmp_path / "r.caef"
    write_file(left, Table(LEFT_SCHEMA, LEFT_DATA))
    write_file(right, Table(RIGHT_SCHEMA, RIGHT_DATA))
    return {"l": left, "r": right}


JOIN_EXPR = (
    "(CASE WHEN l.n IS NULL THEN 0 "
    "WHEN NOT (r.f > 1.0 AND l.id != 2) THEN -l.n "
    "ELSE l.n + r.f END) * 2 + 1"
)


def _project_descriptor(plan, output_name):
    return next(item for item in plan["output"] if item["name"] == output_name)


def _project_expression(plan, output_name):
    project = next(
        op for op in plan["operators"] if op["operator"] == "Project"
    )
    return next(
        entry["expression"]
        for entry in project["expressions"]
        if entry["output"] == output_name
    )


def _rows(result):
    return [
        [result._columns[c][i] for c in range(len(result.schema.columns))]
        for i in range(result.row_count)
    ]


# ---------------------------------------------------------------------------
# explain and execution agree on type / nullability (single table)
# ---------------------------------------------------------------------------


def test_explain_reports_nested_expression_type_and_nullability(single_path):
    plan = explain_file(
        single_path, f"SELECT id, {NESTED_EXPR} AS x FROM input"
    )
    assert _project_descriptor(plan, "x") == {
        "name": "x",
        "type": EXPR_TYPE,
        "nullable": EXPR_NULLABLE,
    }
    # Required columns keep source-schema order and contain exactly the
    # expression's dependencies plus the projected id column.
    scan = next(op for op in plan["operators"] if op["operator"] == "Scan")
    assert scan["required_columns"] == list(EXPR_COLUMNS)
    # The expression tree is rendered recursively with CASE intact.
    tree = _project_expression(plan, "x")
    assert tree["kind"] == "arithmetic" and tree["operator"] == "+"


def test_execution_schema_matches_explain_descriptor(single_path):
    result = query_file(single_path, f"SELECT id, {NESTED_EXPR} AS x FROM input")
    actual = [(c.name, c.type, c.nullable) for c in result.schema.columns]
    plan = explain_file(
        single_path, f"SELECT id, {NESTED_EXPR} AS x FROM input"
    )
    expected = [(item["name"], item["type"], item["nullable"]) for item in plan["output"]]
    assert actual == expected
    assert actual[1] == ("x", EXPR_TYPE, EXPR_NULLABLE)


def test_missing_else_makes_result_nullable(single_path):
    expr = "CASE WHEN n IS NULL THEN 1 WHEN f > 0 THEN 2 END"
    plan = explain_file(single_path, f"SELECT id, {expr} AS x FROM input")
    assert _project_descriptor(plan, "x") == {
        "name": "x",
        "type": "int64",
        "nullable": True,
    }
    # An all-literal CASE without ELSE is nullable even with no columns...
    plan2 = explain_file(
        single_path, "SELECT id, CASE WHEN id > 0 THEN 1 END AS x FROM input"
    )
    assert _project_descriptor(plan2, "x")["nullable"] is True
    # ...but a pure-constant CASE with an ELSE of non-nullable results is not.
    plan3 = explain_file(
        single_path,
        "SELECT id, CASE WHEN 1 = 1 THEN 1 ELSE 0 END AS x FROM input",
    )
    assert _project_descriptor(plan3, "x") == {
        "name": "x",
        "type": "int64",
        "nullable": False,
    }


def test_is_not_null_predicate_is_not_nullable(single_path):
    # A CASE over a nullable bool column keeps the bool type and inherits
    # result nullability only from its THEN/ELSE values, never from its WHEN
    # conditions (a NULL condition merely routes the row elsewhere).
    sql_nullable = (
        "SELECT id, CASE WHEN n IS NOT NULL THEN flag ELSE FALSE END AS x "
        "FROM input"
    )
    plan = explain_file(single_path, sql_nullable)
    assert _project_descriptor(plan, "x") == {
        "name": "x",
        "type": "bool",
        "nullable": True,
    }
    result = query_file(single_path, sql_nullable)
    assert [(c.type, c.nullable) for c in result.schema.columns][1] == ("bool", True)

    # Constant bool results with an ELSE are never nullable.
    sql_const = (
        "SELECT id, CASE WHEN n IS NOT NULL THEN TRUE ELSE FALSE END AS x "
        "FROM input"
    )
    plan2 = explain_file(single_path, sql_const)
    assert _project_descriptor(plan2, "x") == {
        "name": "x",
        "type": "bool",
        "nullable": False,
    }


# ---------------------------------------------------------------------------
# Single table and the equivalent join share the same facts and rows
# ---------------------------------------------------------------------------


def test_single_and_join_agree_on_descriptor_and_rows(single_path, join_paths):
    single_sql = f"SELECT id, {NESTED_EXPR} AS x FROM input ORDER BY id"
    join_sql = (
        f"SELECT l.id, {JOIN_EXPR} AS x FROM l "
        "INNER JOIN r ON l.id = r.rid ORDER BY l.id"
    )

    single_plan = explain_file(single_path, single_sql)
    join_plan = explain_files(join_paths, join_sql)
    assert _project_descriptor(single_plan, "x") == _project_descriptor(join_plan, "x")

    single_rows = _rows(query_file(single_path, single_sql))
    join_rows = _rows(query_files(join_paths, join_sql))
    # The join output qualifies id as "l.id"; result rows must be identical.
    assert single_rows == join_rows
    # And every row really is float64 with NULL propagated from n/f.
    for _id, value in join_rows:
        assert value is None or isinstance(value, float)


def test_single_and_join_agree_on_nested_where(single_path, join_paths):
    # A nested scalar CASE in WHERE compared against a literal, ORed with a
    # column comparison and guarded by IS NULL: the same condition shape
    # binds and filters in the single-table and join paths.
    where_single = (
        "(CASE WHEN n IS NULL THEN 0 WHEN f > 0.0 THEN n ELSE -1 END) > 5 "
        "OR s = 'a'"
    )
    where_join = (
        "(CASE WHEN l.n IS NULL THEN 0 WHEN r.f > 0.0 THEN l.n ELSE -1 END) > 5 "
        "OR r.s = 'a'"
    )
    single_sql = f"SELECT id FROM input WHERE {where_single} ORDER BY id"
    join_sql = (
        "SELECT l.id FROM l INNER JOIN r ON l.id = r.rid "
        f"WHERE {where_join} ORDER BY l.id"
    )

    single_rows = [row[0] for row in _rows(query_file(single_path, single_sql))]
    join_rows = [row[0] for row in _rows(query_files(join_paths, join_sql))]
    assert single_rows == join_rows

    # The bound filter trees carry the same operators in the same order once
    # leaf names are ignored -- the single binder and join binder run the
    # same expression rules.
    single_tree = next(
        op for op in explain_file(single_path, single_sql)["operators"]
        if op["operator"] == "Filter"
    )["condition"]
    join_tree = next(
        op for op in explain_files(join_paths, join_sql)["operators"]
        if op["operator"] == "Filter"
    )["condition"]

    def shape(node):
        if node["kind"] == "column":
            return ("column",)
        if node["kind"] == "literal":
            return ("literal", node["type"])
        if node["kind"] == "case":
            return (
                "case",
                tuple((shape(c["when"]), shape(c["then"])) for c in node["cases"]),
                shape(node["else"]) if node["else"] is not None else None,
            )
        return (
            node["kind"],
            node.get("operator"),
            *(shape(o) for o in node.get("operands", ())),
        )

    assert shape(single_tree) == shape(join_tree)


def test_join_required_columns_follow_source_schema_order(join_paths):
    sql = (
        f"SELECT l.id, {JOIN_EXPR} AS x FROM l "
        "INNER JOIN r ON l.id = r.rid"
    )
    plan = explain_files(join_paths, sql)
    scans = {op["source"]: op["required_columns"] for op in plan["operators"]
             if op["operator"] == "Scan"}
    # Join keys are always scanned; the expression adds l.n and r.f, in each
    # source's own schema order.
    assert scans["l"] == ["id", "n"]
    assert scans["r"] == ["rid", "f"]


def test_nested_expression_under_left_join_null_padding(tmp_path, join_paths):
    # A LEFT JOIN pads the right side with NULLs; the r.f-dependent
    # expression keeps reading the same bound rules, only over NULL-padded
    # values.  Execution and the output descriptor both derive nullability
    # and type from the same bound expression.
    small = tmp_path / "r_small.caef"
    write_file(
        small,
        Table(
            RIGHT_SCHEMA,
            {"rid": [1, 2], "f": [1.5, 2.5], "s": ["a", "b"]},
        ),
    )
    sources = {"l": join_paths["l"], "r_small": small}
    expr = JOIN_EXPR.replace("r.", "r_small.")
    sql = (
        f"SELECT l.id, {expr} AS x FROM l "
        "LEFT JOIN r_small ON l.id = r_small.rid ORDER BY l.id"
    )
    plan = explain_files(sources, sql)
    assert _project_descriptor(plan, "x") == {
        "name": "x",
        "type": EXPR_TYPE,
        "nullable": True,
    }
    rows = _rows(query_files(sources, sql))
    assert [row[0] for row in rows] == [1, 2, 3, 4, 5]
    # Every result is a float or NULL; NULL-padded rows never raise even
    # though the expression contains arithmetic.
    for _id, value in rows:
        assert value is None or isinstance(value, float)
    assert len(rows) == 5


# ---------------------------------------------------------------------------
# Exception classification is identical everywhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr_fragment",
    [
        "1 / 0",                      # division by zero (runtime arithmetic)
        "9223372036854775807 + 1",    # int64 overflow
        "-(-9223372036854775808)",    # unary-minus overflow
    ],
)
def test_runtime_arithmetic_error_single_and_join(single_path, join_paths, expr_fragment):
    single_sql = f"SELECT id, {expr_fragment} AS x FROM input"
    join_sql = (
        f"SELECT l.id, {expr_fragment} AS x FROM l "
        "INNER JOIN r ON l.id = r.rid"
    )
    with pytest.raises(QueryValidationError):
        query_file(single_path, single_sql)
    with pytest.raises(QueryValidationError):
        query_files(join_paths, join_sql)
    # explain binds and plans but never evaluates, so no arithmetic error.
    explain_file(single_path, single_sql)
    explain_files(join_paths, join_sql)


def test_type_incompatibility_rejected_in_single_join_and_explain(
    single_path, join_paths
):
    single_sql = "SELECT id, n + s AS x FROM input"
    join_sql = (
        "SELECT l.id, l.n + r.s AS x FROM l INNER JOIN r ON l.id = r.rid"
    )
    for action in (
        lambda: query_file(single_path, single_sql),
        lambda: explain_file(single_path, single_sql),
        lambda: query_files(join_paths, join_sql),
        lambda: explain_files(join_paths, join_sql),
    ):
        with pytest.raises(QueryValidationError):
            action()


def test_unknown_column_rejected_consistently(single_path, join_paths):
    with pytest.raises(QueryValidationError):
        query_file(single_path, "SELECT missing + 1 AS x FROM input")
    with pytest.raises(QueryValidationError):
        explain_file(single_path, "SELECT missing + 1 AS x FROM input")
    with pytest.raises(QueryValidationError):
        query_files(
            join_paths,
            "SELECT l.id, l.missing + 1 AS x FROM l INNER JOIN r ON l.id = r.rid",
        )
    with pytest.raises(QueryValidationError):
        explain_files(
            join_paths,
            "SELECT l.id, l.missing + 1 AS x FROM l INNER JOIN r ON l.id = r.rid",
        )


def test_syntax_error_precedes_file_access(tmp_path):
    missing = tmp_path / "does-not-exist.caef"
    with pytest.raises(QuerySyntaxError):
        query_file(missing, "SELECT n + FROM input")
    with pytest.raises(QuerySyntaxError):
        explain_file(missing, "SELECT n + FROM input")
    with pytest.raises(QuerySyntaxError):
        query_files({"l": missing}, "SELECT l.n + FROM l")


def test_non_numeric_select_expression_still_rejected(single_path):
    # Arithmetic over a non-numeric operand is a binding (validation) error;
    # a bare comparison never reaches the scalar-expression grammar and stays
    # the historical syntax error raised before the file is read.
    with pytest.raises(QueryValidationError):
        query_file(single_path, "SELECT id, n + s AS x FROM input")
    with pytest.raises(QueryValidationError):
        explain_file(single_path, "SELECT id, -s AS x FROM input")
    with pytest.raises(QuerySyntaxError):
        query_file(single_path, "SELECT id, n > 1 AS x FROM input")
    with pytest.raises(QuerySyntaxError):
        explain_file(single_path, "SELECT id, n > 1 AS x FROM input")


def test_aggregate_argument_expressions_still_rejected(single_path):
    for sql in (
        "SELECT SUM(n + 1) FROM input",
        "SELECT id FROM input GROUP BY id HAVING SUM(n + 1) > 0",
        "SELECT id, COUNT(*) FROM input GROUP BY n + 1",
    ):
        with pytest.raises((QuerySyntaxError, QueryValidationError)):
            query_file(single_path, sql)


# ---------------------------------------------------------------------------
# v1 and v2 storage give the same bound facts and results
# ---------------------------------------------------------------------------


def test_partitioned_source_matches_v1(tmp_path):
    v1 = tmp_path / "v1.caef"
    v2 = tmp_path / "v2.caef"
    write_file(v1, Table(SINGLE_SCHEMA, SINGLE_DATA))
    write_partitioned_file(v2, Table(SINGLE_SCHEMA, SINGLE_DATA), 2)

    sql = f"SELECT id, {NESTED_EXPR} AS x FROM input WHERE n IS NOT NULL ORDER BY id"
    rows_v1 = _rows(query_file(v1, sql))
    rows_v2 = _rows(query_file(v2, sql))
    assert rows_v1 == rows_v2

    plan_v1 = explain_file(v1, sql)
    plan_v2 = explain_file(v2, sql)
    assert _project_descriptor(plan_v1, "x") == _project_descriptor(plan_v2, "x")
    # The v2 scan additionally reports statistics pushdown metadata without
    # changing the expression/filter shape.
    assert _project_expression(plan_v1, "x") == _project_expression(plan_v2, "x")
