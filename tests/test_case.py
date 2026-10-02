"""Tests for searched CASE scalar expressions.

Covers WHEN/THEN/ELSE/END parsing, three-valued condition matching, lazy
branch evaluation, result type unification, nullability, nesting, use in
SELECT / arithmetic / WHERE / ORDER BY, aggregate-position rejection,
joins, explain output, exports and the CLI.
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
        ColumnSchema("flag", "bool"),
    ]
)

DATA = {
    "id": [1, 2, 3, 4, 5],
    "n": [10, None, 30, 40, None],
    "f": [1.5, 2.5, None, -0.5, 10.0],
    "s": ["a", "b", None, "a", "c"],
    "flag": [True, False, True, False, True],
}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "t.caef"
    write_file(p, Table(SCHEMA, DATA))
    return p


def schema_of(table):
    return [(c.name, c.type, c.nullable) for c in table.schema.columns]


def run(path, sql):
    table = query_file(path, sql)
    return [
        [table._columns[c][r] for c in range(len(table.schema.columns))]
        for r in range(table.row_count)
    ]


# ---------------------------------------------------------------------------
# Basic matching: order, only TRUE matches, ELSE and implicit NULL
# ---------------------------------------------------------------------------


def test_case_first_true_wins_in_written_order(path):
    result = query_file(
        path,
        "select case when id >= 1 then 100 when id >= 2 then 200 else 0 end as x "
        "from input order by id",
    )
    assert result.column("x") == [100, 100, 100, 100, 100]


def test_case_multiple_when_and_else(path):
    result = query_file(
        path,
        "select case when n > 100 then 1 when n = 10 then 2 when n = 30 then 3 "
        "else 9 end as x from input order by id",
    )
    # FALSE/UNKNOWN fall through: rows 2 and 5 (n NULL) reach ELSE.
    assert result.column("x") == [2, 9, 3, 9, 9]


def test_case_without_else_returns_null(path):
    result = query_file(
        path,
        "select case when n is not null then n end as x from input order by id"
    )
    assert result.column("x") == [10, None, 30, 40, None]
    assert schema_of(result) == [("x", "int64", True)]


def test_case_unknown_condition_falls_through(path):
    # n = 10 is UNKNOWN for NULL rows; the later WHEN never matches them.
    result = query_file(
        path,
        "select case when n = 10 then 1 when n = 30 then 2 else 3 end as x "
        "from input order by id",
    )
    assert result.column("x") == [1, 3, 2, 3, 3]


# ---------------------------------------------------------------------------
# Result types and nullability
# ---------------------------------------------------------------------------


def test_case_int64_result(path):
    result = query_file(
        path, "select case when flag then id else -id end as x from input order by id"
    )
    assert schema_of(result) == [("x", "int64", False)]
    assert result.column("x") == [1, -2, 3, -4, 5]


def test_case_int64_float64_mix_unifies_to_float64(path):
    result = query_file(
        path,
        "select case when flag then id else 1.5 end as x from input order by id",
    )
    assert schema_of(result) == [("x", "float64", False)]
    assert result.column("x") == [1.0, 1.5, 3.0, 1.5, 5.0]
    result = query_file(
        path,
        "select case when flag then 1.5 else id end as x from input order by id",
    )
    assert result.column("x") == [1.5, 2.0, 1.5, 4.0, 1.5]


def test_case_bool_result(path):
    result = query_file(
        path,
        "select case when n > 20 then true else false end as x from input order by id",
    )
    assert schema_of(result) == [("x", "bool", False)]
    assert result.column("x") == [False, False, True, True, False]


def test_case_utf8_result(path):
    result = query_file(
        path,
        "select case when s = 'a' then 'yes' else 'no' end as x from input order by id",
    )
    assert schema_of(result) == [("x", "utf8", False)]
    assert result.column("x") == ["yes", "no", "no", "yes", "no"]


@pytest.mark.parametrize(
    "sql",
    [
        "select case when id > 0 then 1 else 'x' end as x from input",
        "select case when id > 0 then true else 1 end as x from input",
        "select case when id > 0 then 'a' else true end as x from input",
        "select case when id > 0 then 1 when id < 0 then 1.5 else true end as x from input",
        "select case when id > 0 then 1.5 else 'x' end as x from input",
    ],
)
def test_case_incompatible_result_types(path, sql):
    with pytest.raises(QueryValidationError):
        query_file(path, sql)


def test_case_result_nullability(path):
    result = query_file(
        path,
        "select "
        "case when id > 0 then 1 else 0 end as a, "
        "case when id > 0 then n else 0 end as b, "
        "case when id > 0 then 1 end as c, "
        "case when id > 0 then 1 else n end as d, "
        "case when id > 0 then 1 else 0.5 end as e "
        "from input where id = 1",
    )
    assert schema_of(result) == [
        ("a", "int64", False),
        ("b", "int64", True),
        ("c", "int64", True),  # omitted ELSE
        ("d", "int64", True),
        ("e", "float64", False),
    ]


def test_case_nonnull_even_when_condition_uses_nullable_column(path):
    # A NULL condition only routes to ELSE; it does not make the result NULL.
    result = query_file(
        path,
        "select case when n > 0 then 1 else 0 end as x from input order by id",
    )
    assert schema_of(result) == [("x", "int64", False)]
    assert result.column("x") == [1, 0, 1, 1, 0]


def test_case_nullable_then_with_nonnull_else(path):
    result = query_file(
        path,
        "select case when flag then n else 0 end as x from input order by id",
    )
    assert schema_of(result) == [("x", "int64", True)]
    assert result.column("x") == [10, 0, 30, 0, None]


# ---------------------------------------------------------------------------
# Laziness: unhit branches never evaluate
# ---------------------------------------------------------------------------


def test_unhit_branch_division_by_zero_does_not_raise(path):
    result = query_file(
        path,
        "select case when id < 3 then 0 else 1 / 0 end as x from input "
        "where id < 3 order by id",
    )
    assert result.column("x") == [0.0, 0.0]
    # An earlier WHEN matches, so later branches are skipped entirely.
    result = query_file(
        path,
        "select case when id < 3 then 0 when true then 1 / 0 else 2 end as x "
        "from input where id < 3 order by id",
    )
    assert result.column("x") == [0.0, 0.0]


def test_unhit_branch_int64_overflow_does_not_raise(path):
    result = query_file(
        path,
        "select case when id < 3 then 1 else 9223372036854775807 + 1 end as x "
        "from input where id < 3 order by id",
    )
    assert result.column("x") == [1, 1]


def test_unhit_branch_non_finite_float_does_not_raise(tmp_path):
    p = tmp_path / "f.caef"
    write_file(
        p, Table(Schema([ColumnSchema("x", "float64")]), {"x": [1e308, 1.0]})
    )
    result = query_file(
        p,
        "select case when x < 10 then x * 10 else 1.0 end as y from input "
        "where x < 10",
    )
    assert result.column("y") == [10.0]


def test_hit_branch_still_raises(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select case when id > 0 then 1 / 0 else 1 end as x from input")
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "select case when id > 0 then 9223372036854775807 + 1 else 1 end as x "
            "from input",
        )


def test_unhit_branch_overflow_filtered_and_limited(tmp_path):
    p = tmp_path / "big.caef"
    write_file(
        p, Table(Schema([ColumnSchema("x", "int64")]), {"x": [1, 2**63 - 1]})
    )
    # WHERE removes the row that would take the overflowing branch.
    result = query_file(
        p, "select case when x > 100 then x + 1 else x end as y from input where x < 100"
    )
    assert result.column("y") == [1]
    # LIMIT cuts before projection, so the surviving overflow row is fine.
    result = query_file(
        p,
        "select case when x > 100 then x + 1 else x end as y from input "
        "order by x limit 1",
    )
    assert result.column("y") == [1]


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------


def test_nested_case(path):
    result = query_file(
        path,
        "select case when id = 1 then case when flag then 100 else -1 end "
        "else 9 end as x from input order by id",
    )
    assert result.column("x") == [100, 9, 9, 9, 9]


def test_nested_case_in_condition_and_result_and_else(path):
    sql = (
        "select case "
        "when case when flag then id else 0 end > 2 then 'hi' "
        "else case when s = 'a' then 'aa' else 'zz' end end as x "
        "from input order by id"
    )
    assert query_file(path, sql).column("x") == ["aa", "zz", "hi", "aa", "hi"]


def test_nested_case_type_unification(path):
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "select case when id > 0 then case when flag then 1 else 2 end "
            "else 'x' end as x from input",
        )


# ---------------------------------------------------------------------------
# CASE as an arithmetic operand and in WHERE
# ---------------------------------------------------------------------------


def test_case_as_arithmetic_operand(path):
    result = query_file(
        path,
        "select case when id > 2 then 10 else 20 end + 1 as x from input order by id",
    )
    assert result.column("x") == [21, 21, 11, 11, 11]
    result = query_file(
        path,
        "select 6 / case when id = 1 then 3 else 2 end as x from input order by id"
    )
    assert schema_of(result) == [("x", "float64", False)]
    assert result.column("x") == [2.0, 3.0, 3.0, 3.0, 3.0]
    result = query_file(
        path,
        "select -case when id = 1 then 1 else 2 end as x from input order by id"
    )
    assert result.column("x") == [-1, -2, -2, -2, -2]


def test_case_on_both_sides_of_where_comparison(path):
    assert run(
        path,
        "select id from input where case when id > 3 then 1 else 0 end = 1 order by id",
    ) == [[4], [5]]
    assert run(
        path,
        "select id from input where 1 = case when id > 3 then 1 else 0 end order by id",
    ) == [[4], [5]]
    assert run(
        path,
        "select id from input where case when id > 3 then 1 else 0 end = "
        "case when id > 4 then 1 else 0 end order by id",
    ) == [[1], [2], [3], [5]]


def test_bool_case_as_where_predicate(path):
    assert run(
        path,
        "select id from input where not case when flag then true else false end "
        "order by id"
    ) == [[2], [4]]
    assert run(
        path,
        "select id from input where case when flag then true else false end and id > 1 "
        "order by id"
    ) == [[3], [5]]
    assert run(
        path,
        "select id from input where (case when flag then 1 end) is null order by id"
    ) == [[2], [4]]


def test_numeric_case_directly_as_boolean_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where case when flag then 1 else 0 end")


def test_case_condition_must_be_boolean(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select case when id then 1 end as x from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select case when s then 1 end as x from input")


def test_case_string_comparison_rules(path):
    # utf8 CASE compares only to utf8.
    assert run(
        path,
        "select id from input where case when flag then 'a' else 'b' end = s order by id"
    ) == [[1], [2]]
    with pytest.raises(QueryValidationError):
        query_file(
            path, "select id from input where case when flag then 'a' else 'b' end = 1"
        )


# ---------------------------------------------------------------------------
# ORDER BY alias over a CASE expression
# ---------------------------------------------------------------------------


def test_order_by_case_alias(path):
    result = query_file(
        path,
        "select id, case when n is null then 0 else n end as x from input order by x, id"
    )
    assert result.column("id") == [2, 5, 1, 3, 4]
    result = query_file(
        path,
        "select id, case when n is null then 0 else n end as x from input "
        "order by x desc, id"
    )
    assert result.column("id") == [4, 3, 1, 2, 5]
    result = query_file(
        path,
        "select id, case when n is null then n else 0 end as x from input "
        "order by x nulls first, id"
    )
    assert result.column("id") == [2, 5, 1, 3, 4]


# ---------------------------------------------------------------------------
# Aggregate queries keep rejecting scalar CASE expressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "select sum(case when id > 0 then 1 else 0 end) from input",
        "select count(case when id > 0 then 1 else 0 end) from input",
        "select count(*) from input group by case when id > 0 then 1 else 0 end",
        "select case when id > 0 then 1 else 0 end as x from input group by id",
        "select sum(id), case when id > 0 then 1 else 0 end as x from input",
        "select count(*) + case when id > 0 then 1 else 0 end as x from input",
    ],
)
def test_aggregate_query_rejects_case(path, sql):
    with pytest.raises(QueryValidationError):
        query_file(path, sql)


def test_aggregate_query_where_case_allowed(path):
    # CASE is a scalar expression; WHERE of an aggregate query already
    # accepts those.
    result = query_file(path, "select count(*) from input where case when n > 20 then true else false end")
    assert result.column("COUNT(*)") == [2]
    result = query_file(
        path,
        "select s, sum(id) from input where case when id * 2 > 4 then true else false end "
        "group by s order by s",
    )
    assert result.column("s") == ["a", "c", None]
    assert result.column("SUM(id)") == [4, 5, 3]


# ---------------------------------------------------------------------------
# Syntax errors (before any file access) and keyword casing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "select case then 1 end as x from input",                 # missing WHEN cond
        "select case when id > 0 then 1 from input",              # missing END
        "select case when id > 0 end as x from input",            # missing THEN result
        "select case when id > 0 then 1 else 0 as x from input",  # missing END
        "select case when id > 0 then 1 else end as x from input",  # empty ELSE
        "select case when then 1 end as x from input",            # empty condition
        "select case id when 1 then 1 end as x from input",       # simple CASE
        "select when id > 0 then 1 end as x from input",          # orphan WHEN
        "select id from input end",                               # orphan END
        "select case when id > 0 then case when id > 1 then 1 end as x from input",  # unclosed nested
        "select case when id > 0 then 1 else 0 end end as x from input",  # trailing END
        "select case when id > 0 then 1 when id < 5 then 2 as x from input",  # no END after WHENs
    ],
)
def test_case_syntax_errors_before_file_access(sql):
    with pytest.raises(QuerySyntaxError):
        query_file("/nonexistent/t.caef", sql)


def test_case_syntax_errors_before_file_access_join():
    missing = {"l": "/nonexistent/l.caef", "r": "/nonexistent/r.caef"}
    with pytest.raises(QuerySyntaxError):
        query_files(
            missing,
            "select case when l.id > 0 then 1 from l inner join r on l.id = r.id",
        )


def test_case_keywords_case_insensitive(path):
    result = query_file(
        path,
        "select CaSe wHeN id > 0 ThEn 1 eLsE 0 EnD as x from input order by id",
    )
    assert result.column("x") == [1, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------


@pytest.fixture()
def join_paths(tmp_path):
    left = tmp_path / "l.caef"
    right = tmp_path / "r.caef"
    write_file(
        left,
        Table(
            Schema(
                [
                    ColumnSchema("k", "int64"),
                    ColumnSchema("a", "int64"),
                    ColumnSchema("x", "float64", nullable=True),
                ]
            ),
            {"k": [1, 2, 3], "a": [10, 20, 30], "x": [0.5, None, 2.5]},
        ),
    )
    write_file(
        right,
        Table(
            Schema([ColumnSchema("k", "int64"), ColumnSchema("b", "int64", nullable=True)]),
            {"k": [1, 1, 3], "b": [100, None, 300]},
        ),
    )
    return {"l": left, "r": right}


def test_join_case_projection_and_where(join_paths):
    result = query_files(
        join_paths,
        "select l.k, case when r.b is null then -1 else r.b end as z "
        "from l left join r on l.k = r.k order by l.k, r.b",
    )
    # Raw r.b drives the sort (NULL last); CASE only remaps the output.
    assert result.column("l.k") == [1, 1, 2, 3]
    assert result.column("z") == [100, -1, -1, 300]
    result = query_files(
        join_paths,
        "select l.k from l left join r on l.k = r.k "
        "where case when r.b is null then true else false end order by l.k",
    )
    assert result.column("l.k") == [1, 2]


def test_join_case_order_by_alias(join_paths):
    result = query_files(
        join_paths,
        "select case when r.b is null then 0 else r.b end as z, l.k "
        "from l left join r on l.k = r.k order by z",
    )
    assert result.column("z") == [0, 0, 100, 300]
    assert result.column("l.k") == [1, 2, 1, 3]


def test_join_case_requires_qualified_columns(join_paths):
    with pytest.raises(QueryValidationError):
        query_files(
            join_paths,
            "select case when a > 0 then 1 else 0 end as z "
            "from l inner join r on l.k = r.k",
        )
    with pytest.raises(QueryValidationError):
        query_files(
            join_paths,
            "select l.k from l inner join r on l.k = r.k "
            "where case when b > 0 then true else false end",
        )


def test_join_single_source_case(join_paths):
    result = query_files(
        join_paths, "select case when a > 15 then 1 else 0 end as z from l"
    )
    assert result.column("z") == [0, 1, 1]


# ---------------------------------------------------------------------------
# Explain: recursive tree, order, nullable else, required columns
# ---------------------------------------------------------------------------


def _project_expression(plan):
    project = next(op for op in plan["operators"] if op["operator"] == "Project")
    return project["expressions"][0]["expression"]


def test_explain_case_tree_shape(path):
    plan = explain_file(
        path,
        "select case when f > 1 then n else id * 2 end as x from input",
    )
    expr = _project_expression(plan)
    assert expr["kind"] == "case"
    assert list(expr.keys()) == ["kind", "cases", "else"]
    case = expr["cases"][0]
    assert list(case.keys()) == ["when", "then"]
    assert case["when"] == {
        "kind": "comparison",
        "operator": ">",
        "operands": [
            {"kind": "column", "name": "f"},
            {"kind": "literal", "type": "int64", "value": 1},
        ],
    }
    assert case["then"] == {"kind": "column", "name": "n"}
    assert expr["else"]["kind"] == "arithmetic"
    assert expr["else"]["operator"] == "*"
    assert plan["output"] == [{"name": "x", "type": "int64", "nullable": True}]


def test_explain_case_preserves_when_then_order(path):
    plan = explain_file(
        path,
        "select case when id = 1 then 10 when id = 2 then 20 else 30 end as x from input",
    )
    cases = _project_expression(plan)["cases"]
    assert [c["when"]["operands"][1]["value"] for c in cases] == [1, 2]
    assert [c["then"]["value"] for c in cases] == [10, 20]
    assert _project_expression(plan)["else"]["value"] == 30


def test_explain_omitted_else_is_null(path):
    plan = explain_file(path, "select case when id > 0 then 1 end as x from input")
    assert _project_expression(plan)["else"] is None
    assert plan["output"] == [{"name": "x", "type": "int64", "nullable": True}]


def test_explain_nested_case(path):
    plan = explain_file(
        path,
        "select case when flag then case when id > 2 then 1 else 2 end else 3 end as x "
        "from input",
    )
    outer = _project_expression(plan)
    inner = outer["cases"][0]["then"]
    assert inner["kind"] == "case"
    assert inner["cases"][0]["then"]["value"] == 1
    assert outer["else"]["value"] == 3


def test_explain_case_required_columns(path):
    # Conditions and both THEN/ELSE result columns all count, schema order.
    plan = explain_file(
        path,
        "select case when f > 1 then n when s = 'a' then id else id end as x from input",
    )
    scan = next(op for op in plan["operators"] if op["operator"] == "Scan")
    assert scan["required_columns"] == ["id", "n", "f", "s"]


def test_explain_case_in_filter_required_columns(path):
    plan = explain_file(
        path,
        "select id from input where case when flag then n else 0 end > 5"
    )
    scan = next(op for op in plan["operators"] if op["operator"] == "Scan")
    assert scan["required_columns"] == ["id", "n", "flag"]
    cond = next(op for op in plan["operators"] if op["operator"] == "Filter")["condition"]
    assert cond["operands"][0]["kind"] == "case"


def test_explain_does_not_evaluate_branches(path):
    # A division by zero in an unhit (and even the hit) branch must not
    # raise during explain: binding only assigns types.
    plan = explain_file(
        path,
        "select case when id < 3 then 0 else 1 / 0 end as x from input",
    )
    assert plan["output"] == [{"name": "x", "type": "float64", "nullable": False}]


def test_explain_case_output_matches_query(path):
    sql = (
        "select case when s = 'a' then 'A' when s is null then 'U' else s end as tag, "
        "case when n is null then 0.0 else n end as num from input order by id"
    )
    plan = explain_file(path, sql)
    result = query_file(path, sql)
    assert plan["output"] == [
        {"name": c.name, "type": c.type, "nullable": c.nullable}
        for c in result.schema.columns
    ]


def test_explain_files_case(join_paths):
    plan = explain_files(
        join_paths,
        "select case when r.b is null then 0 else r.b end as z from l "
        "left join r on l.k = r.k",
    )
    scans = {
        op["source"]: op["required_columns"]
        for op in plan["operators"]
        if op["operator"] == "Scan"
    }
    assert scans == {"l": ["k"], "r": ["k", "b"]}
    expr = _project_expression(plan)
    assert expr["kind"] == "case"
    assert plan["output"] == [{"name": "z", "type": "int64", "nullable": True}]


# ---------------------------------------------------------------------------
# Determinism and export / CLI
# ---------------------------------------------------------------------------


def test_repeated_query_explain_export_identical(path, tmp_path):
    sql = (
        "select id, case when n is null then 0 else n end as x from input "
        "order by x, id"
    )
    a = query_file(path, sql)
    b = query_file(path, sql)
    assert a.schema == b.schema
    assert a._columns == b._columns
    assert explain_file(path, sql) == explain_file(path, sql)
    d1, d2 = tmp_path / "a.csv", tmp_path / "b.csv"
    export_query_file(path, sql, d1)
    export_query_file(path, sql, d2)
    assert d1.read_bytes() == d2.read_bytes()


def test_export_case_csv_and_jsonl(path, tmp_path):
    sql = "select case when flag then s else 'z' end as tag from input order by id"
    csv_dest = tmp_path / "o.csv"
    n = export_query_file(path, sql, csv_dest, "csv")
    assert n == 5
    text = csv_dest.read_text("utf-8")
    assert text.splitlines() == ["tag", "a", "z", "", "z", "c"]  # NULL -> empty field
    jsonl_dest = tmp_path / "o.jsonl"
    export_query_file(path, sql, jsonl_dest, "jsonl")
    lines = jsonl_dest.read_text("utf-8").splitlines()
    assert json.loads(lines[0]) == {"tag": "a"}
    assert json.loads(lines[2]) == {"tag": None}


def test_export_files_case(join_paths, tmp_path):
    dest = tmp_path / "o.csv"
    sql = (
        "select case when r.b is null then 0 else r.b end as z from l "
        "left join r on l.k = r.k order by l.k, r.b"
    )
    assert export_query_files(join_paths, sql, dest) == 4
    assert dest.read_text("utf-8").splitlines() == ["z", "100", "0", "0", "300"]


def test_cli_query_case(path, capsys):
    code = main(
        [
            "query",
            str(path),
            "select case when n is null then 0 else n end as x, id from input "
            "order by x, id",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [
        {"name": "x", "type": "int64", "nullable": True},
        {"name": "id", "type": "int64", "nullable": False},
    ]
    assert payload["rows"] == [[0, 2], [0, 5], [10, 1], [30, 3], [40, 4]]


def test_cli_explain_case(path, capsys):
    code = main(
        ["explain", str(path), "select case when id > 2 then 1 else 0 end as x from input"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    payload = json.loads(out)
    assert payload["output"] == [{"name": "x", "type": "int64", "nullable": False}]


def test_cli_case_exit_codes(path, capsys):
    assert main(["query", str(path), "select case when id then 1 end as x from input"]) == 2
    assert main(["query", str(path), "select case when id > 0 then 1 from input"]) == 2
    assert main(["query", str(path), "select case when id > 0 then 1 else 'x' end as x from input"]) == 2
    assert main(["query", str(path), "select case when id > 0 then 1/0 else 2 end as x from input"]) == 2
