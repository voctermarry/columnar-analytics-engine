"""Tests for searched CASE scalar expressions."""

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
    ]
)

DATA = {
    "id": [1, 2, 3, 4, 5],
    "n": [10, None, 30, 40, None],
    "f": [1.5, 2.5, None, -0.5, 10.0],
    "s": ["a", "b", None, "a", "c"],
}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "t.caef"
    write_file(p, Table(SCHEMA, DATA))
    return p


def schema_of(table):
    return [(c.name, c.type, c.nullable) for c in table.schema.columns]


# ---------------------------------------------------------------------------
# SELECT: branch order, ELSE, NULL fallthrough
# ---------------------------------------------------------------------------


def test_first_true_branch_wins(path):
    result = query_file(
        path,
        "select id, case when n > 20 then 1 when n > 5 then 2 else 3 end as c "
        "from input",
    )
    assert result.column("c") == [2, 3, 1, 1, 3]


def test_missing_else_yields_null(path):
    result = query_file(path, "select case when n > 100 then 1 end as c from input")
    assert result.column("c") == [None] * 5
    assert schema_of(result) == [("c", "int64", True)]


def test_unknown_condition_falls_through(path):
    # n is NULL for rows 2 and 5: the comparison is UNKNOWN, not TRUE.
    result = query_file(
        path, "select id, case when n > 20 then 1 else 0 end as c from input"
    )
    assert result.column("c") == [0, 0, 1, 1, 0]


def test_else_selected_when_no_condition_true(path):
    result = query_file(
        path,
        "select case when n > 100 then 1 when n < 0 then 2 else 42 end as c "
        "from input",
    )
    assert result.column("c") == [42] * 5
    assert schema_of(result) == [("c", "int64", False)]


def test_keywords_case_insensitive(path):
    result = query_file(
        path, "SeLeCt CaSe WhEn n > 20 ThEn 1 ElSe 0 EnD aS c FrOm input"
    )
    assert result.column("c") == [0, 0, 1, 1, 0]


def test_nested_case(path):
    result = query_file(
        path,
        "select id, case when id > 3 then case when n > 35 then 100 else 50 end "
        "else 0 end as c from input",
    )
    assert result.column("c") == [0, 0, 0, 100, 50]


def test_case_in_condition_and_result(path):
    result = query_file(
        path,
        "select id from input where case when case when n > 20 then 1 else 0 "
        "end = 1 then true else false end",
    )
    assert result.column("id") == [3, 4]


# ---------------------------------------------------------------------------
# Types and nullability
# ---------------------------------------------------------------------------


def test_int64_float64_unify_to_float64(path):
    result = query_file(
        path, "select case when id > 3 then 1.5 else 2 end as c from input"
    )
    assert result.column("c") == [2.0, 2.0, 2.0, 1.5, 1.5]
    assert schema_of(result) == [("c", "float64", False)]


def test_nullable_from_branch_columns(path):
    result = query_file(
        path,
        "select case when id > 0 then n else 0 end as a, "
        "case when id > 0 then id else 0 end as b from input",
    )
    assert schema_of(result) == [("a", "int64", True), ("b", "int64", False)]


def test_nullable_from_else_branch(path):
    result = query_file(
        path, "select case when id > 0 then 1 else n end as c from input"
    )
    assert schema_of(result) == [("c", "int64", True)]


@pytest.mark.parametrize(
    "sql",
    [
        "select case when id > 0 then 1 else 'a' end as c from input",
        "select case when id > 0 then true else 1 end as c from input",
        "select case when id > 0 then 'a' else true end as c from input",
        "select case when id > 0 then 1.5 else 'a' end as c from input",
        "select case when id > 0 then 1 when id > 1 then 'a' else 2 end as c from input",
    ],
)
def test_incompatible_branch_types_rejected(path, sql):
    with pytest.raises(QueryValidationError):
        query_file(path, sql)


def test_non_boolean_condition_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select case when id + 1 then 1 else 2 end as c from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select case when s then 1 else 2 end as c from input")


def test_utf8_case_in_select_rejected(path):
    # Computed SELECT expressions stay numeric; a utf8 CASE is only usable
    # inside comparisons.
    with pytest.raises(QueryValidationError):
        query_file(path, "select case when id > 0 then 'a' else 'b' end as c from input")


# ---------------------------------------------------------------------------
# Lazy branch evaluation
# ---------------------------------------------------------------------------


def test_unselected_branch_errors_do_not_surface(path):
    result = query_file(
        path, "select case when id > 0 then 1 else 1 / 0 end as c from input"
    )
    assert result.column("c") == [1] * 5


def test_unselected_branch_overflow_and_non_finite(tmp_path):
    p = tmp_path / "big.caef"
    write_file(
        p,
        Table(
            Schema([ColumnSchema("x", "int64"), ColumnSchema("g", "float64")]),
            {"x": [2**63 - 1], "g": [1e308]},
        ),
    )
    result = query_file(
        p, "select case when x > 0 then 0 else x + 1 end as c from input"
    )
    assert result.column("c") == [0]
    result = query_file(
        p, "select case when x > 0 then 0 else g * 10 end as c from input"
    )
    assert result.column("c") == [0.0]


def test_selected_branch_errors_surface(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select case when id > 0 then 1 / 0 else 2 end as c from input")


def test_selected_branch_error_only_for_matching_rows(path):
    # Rows 1, 2 and 5 take the ELSE branch and never divide by (n - 10) = 0;
    # rows 3 and 4 take the THEN branch with a non-zero divisor.
    result = query_file(
        path,
        "select case when n > 10 then 100 / (n - 10) else -1 end as c from input",
    )
    assert result.column("c") == [-1.0, -1.0, 5.0, pytest.approx(100 / 30), -1.0]


# ---------------------------------------------------------------------------
# CASE as arithmetic operand and in WHERE
# ---------------------------------------------------------------------------


def test_case_as_arithmetic_operand(path):
    result = query_file(
        path, "select case when id > 3 then 10 else 1 end + id as c from input"
    )
    assert result.column("c") == [2, 3, 4, 14, 15]
    result = query_file(
        path, "select id * (case when n is null then 0 else 1 end) as c from input"
    )
    assert result.column("c") == [1, 0, 3, 4, 0]


def test_case_in_where_comparison(path):
    assert query_file(
        path, "select id from input where case when n > 20 then 1 else 0 end = 1"
    ).column("id") == [3, 4]
    assert query_file(
        path, "select id from input where 1 = case when n > 20 then 1 else 0 end"
    ).column("id") == [3, 4]
    assert query_file(
        path,
        "select id from input where case when id > 3 then 'a' else 'z' end = s",
    ).column("id") == [4]


def test_bool_case_as_condition(path):
    assert query_file(
        path,
        "select id from input where case when id > 3 then true else false end",
    ).column("id") == [4, 5]


def test_case_is_null(path):
    # Without ELSE, rows whose conditions are FALSE or UNKNOWN yield NULL.
    assert query_file(
        path,
        "select id from input where case when n > 20 then 1 end is null",
    ).column("id") == [1, 2, 5]


def test_numeric_case_as_bare_condition_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where case when id > 0 then 1 else 0 end")


# ---------------------------------------------------------------------------
# ORDER BY alias
# ---------------------------------------------------------------------------


def test_order_by_case_alias(path):
    result = query_file(
        path,
        "select id, case when n is null then 0 else n end as c from input "
        "order by c desc",
    )
    assert result.column("c") == [40, 30, 10, 0, 0]
    assert result.column("id") == [4, 3, 1, 2, 5]


def test_order_by_case_alias_nulls(path):
    result = query_file(
        path,
        "select id, case when n > 20 then n end as c from input "
        "order by c nulls first",
    )
    assert result.column("id") == [1, 2, 5, 3, 4]


# ---------------------------------------------------------------------------
# Aggregate queries stay expression-free
# ---------------------------------------------------------------------------


def test_aggregate_query_rejects_case(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select case when id > 0 then 1 else 2 end as c from input group by id")
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(*), case when id > 0 then 1 else 2 end as c from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select sum(case when id > 0 then 1 else 2 end) from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input group by case when id > 0 then 1 else 2 end")


def test_aggregate_query_where_case_allowed(path):
    result = query_file(
        path,
        "select count(*) from input where case when n > 20 then 1 else 0 end = 1",
    )
    assert result.column("COUNT(*)") == [2]


# ---------------------------------------------------------------------------
# Syntax errors (raised before any file is read)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "select case when id > 0 then 1 from input",            # missing END
        "select case else 1 end as c from input",               # at least one WHEN
        "select case end as c from input",                      # empty CASE
        "select case when id > 0 then end as c from input",     # empty THEN
        "select case when then 1 end as c from input",          # empty condition
        "select case when id > 0 1 end as c from input",        # missing THEN
        "select case id when 1 then 1 else 2 end as c from input",  # no simple CASE
        "select case when id > 0 then 1 else end as c from input",    # empty ELSE
        "select case when id > 0 then case when id > 1 then 1 end as c from input",  # unclosed nesting
        "select id from input where id = 1 and end",            # orphan END
        "select id from input where id = when",                 # orphan WHEN
        "select when from input",                               # keyword as column
        "select case when id > 0 then 1 end from input",        # computed CASE without AS
    ],
)
def test_case_syntax_errors_before_file_access(sql):
    with pytest.raises(QuerySyntaxError):
        query_file("/nonexistent/t.caef", sql)


def test_case_syntax_error_before_file_access_join():
    missing = {"l": "/nonexistent/l.caef", "r": "/nonexistent/r.caef"}
    with pytest.raises(QuerySyntaxError):
        query_files(
            missing,
            "select case when l.a > 0 then 1 from l inner join r on l.k = r.k",
        )


# ---------------------------------------------------------------------------
# Two-file join queries
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


def test_join_case_expression(join_paths):
    result = query_files(
        join_paths,
        "select l.k, case when r.b > 50 then l.a else -1 end as c "
        "from l left join r on l.k = r.k",
    )
    assert result.column("l.k") == [1, 1, 2, 3]
    assert result.column("c") == [10, -1, -1, 30]


def test_join_case_requires_qualified_columns(join_paths):
    with pytest.raises(QueryValidationError):
        query_files(
            join_paths,
            "select case when a > 0 then 1 else 2 end as c "
            "from l inner join r on l.k = r.k",
        )


def test_join_case_in_where(join_paths):
    result = query_files(
        join_paths,
        "select l.k from l inner join r on l.k = r.k "
        "where case when r.b is null then 0 else 1 end = 0",
    )
    assert result.column("l.k") == [1]


# ---------------------------------------------------------------------------
# EXPLAIN
# ---------------------------------------------------------------------------


def find_operator(plan, kind):
    return next(op for op in plan["operators"] if op["operator"] == kind)


def test_explain_case_project_tree(path):
    plan = explain_file(
        path, "SELECT CASE WHEN n > 1 THEN f ELSE 2 END AS c FROM input"
    )
    project = find_operator(plan, "Project")
    expr = project["expressions"][0]["expression"]
    assert expr["kind"] == "case"
    assert [set(branch) for branch in expr["whens"]] == [{"when", "then"}]
    assert expr["whens"][0]["when"] == {
        "kind": "comparison",
        "operator": ">",
        "operands": [
            {"kind": "column", "name": "n"},
            {"kind": "literal", "type": "int64", "value": 1},
        ],
    }
    assert expr["whens"][0]["then"] == {"kind": "column", "name": "f"}
    assert expr["else"] == {"kind": "literal", "type": "int64", "value": 2}
    assert plan["output"] == [{"name": "c", "type": "float64", "nullable": True}]


def test_explain_case_without_else(path):
    plan = explain_file(path, "SELECT CASE WHEN n > 1 THEN 1 END AS c FROM input")
    expr = find_operator(plan, "Project")["expressions"][0]["expression"]
    assert expr["else"] is None
    assert plan["output"] == [{"name": "c", "type": "int64", "nullable": True}]


def test_explain_case_in_filter_and_required_columns(path):
    plan = explain_file(
        path,
        "SELECT CASE WHEN n > 1 THEN id ELSE 0 END AS c FROM input "
        "WHERE CASE WHEN f < 0 THEN 1 ELSE 0 END = 1",
    )
    cond = find_operator(plan, "Filter")["condition"]
    assert cond["kind"] == "comparison"
    assert cond["operands"][0]["kind"] == "case"
    scan = find_operator(plan, "Scan")
    assert scan["required_columns"] == ["id", "n", "f"]


def test_explain_case_join_required_columns(join_paths):
    plan = explain_files(
        join_paths,
        "select case when r.b > 1 then l.a else 0 end as c "
        "from l inner join r on l.k = r.k",
    )
    scans = [op for op in plan["operators"] if op["operator"] == "Scan"]
    assert scans[0] == {"operator": "Scan", "source": "l", "required_columns": ["k", "a"]}
    assert scans[1] == {"operator": "Scan", "source": "r", "required_columns": ["k", "b"]}


def test_explain_binds_types_without_evaluating(path):
    # Division by zero in an unselected branch must not surface at explain
    # time; neither may a selected one.
    plan = explain_file(
        path, "SELECT CASE WHEN id > 0 THEN 1 / 0 ELSE 2 END AS c FROM input"
    )
    assert plan["output"] == [{"name": "c", "type": "float64", "nullable": False}]


def test_explain_case_deterministic(path):
    sql = "SELECT CASE WHEN n > 1 THEN f ELSE 2 END AS c FROM input WHERE id > 0"
    assert explain_file(path, sql) == explain_file(path, sql.lower().replace(" as ", " AS "))


# ---------------------------------------------------------------------------
# CLI / export / determinism
# ---------------------------------------------------------------------------


def test_cli_query_case(path, capsys):
    code = main(
        ["query", str(path), "select id, case when n > 20 then 1 else 0 end as c from input"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [
        {"name": "id", "type": "int64", "nullable": False},
        {"name": "c", "type": "int64", "nullable": False},
    ]
    assert payload["rows"] == [[1, 0], [2, 0], [3, 1], [4, 1], [5, 0]]


def test_cli_case_exit_codes(path, capsys):
    assert main(["query", str(path), "select case when id > 0 then 1 end as c from input"]) == 0
    assert main(["query", str(path), "select case when id > 0 then 1 from input"]) == 2
    assert main(["query", str(path), "select case when id > 0 then 1 else 'a' end as c from input"]) == 2
    assert main(["explain", str(path), "select case when id > 0 then 1 / 0 end as c from input"]) == 0


def test_export_case(path, tmp_path):
    dest = tmp_path / "out.csv"
    rows = export_query_file(
        path, "select case when n > 20 then n else 0 end as c from input", dest
    )
    assert rows == 5
    assert dest.read_text(encoding="utf-8") == "c\n0\n0\n30\n40\n0\n"


def test_repeated_execution_is_deterministic(path):
    sql = (
        "select id, case when n > 20 then n * 2 else -1 end as c from input "
        "where id > 1 order by c desc limit 3"
    )
    first = query_file(path, sql)
    second = query_file(path, sql)
    assert first.columns == second.columns
    assert schema_of(first) == schema_of(second)
