"""Tests for numeric scalar expressions in SELECT / WHERE."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from columnar_analytics import (
    ColumnSchema,
    QuerySyntaxError,
    QueryValidationError,
    Schema,
    Table,
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
# SELECT arithmetic: operators, precedence, result types
# ---------------------------------------------------------------------------


def test_select_arithmetic_operators(path):
    result = query_file(
        path,
        "select id, n + 1 as a, n - 2 as b, n * 3 as c, n / 4 as d from input",
    )
    assert result.column("a") == [11, None, 31, 41, None]
    assert result.column("b") == [8, None, 28, 38, None]
    assert result.column("c") == [30, None, 90, 120, None]
    assert result.column("d") == [2.5, None, 7.5, 10.0, None]


def test_result_column_order_and_names(path):
    result = query_file(path, "select n + 1 as z, id, f from input")
    assert [c.name for c in result.schema.columns] == ["z", "id", "f"]


def test_precedence_and_parentheses(path):
    result = query_file(
        path,
        "select id, n + 1 * 2 as x, (n + 1) * 2 as y, -n + 1 as w from input",
    )
    assert result.column("x") == [12, None, 32, 42, None]
    assert result.column("y") == [22, None, 62, 82, None]
    assert result.column("w") == [-9, None, -29, -39, None]


def test_unary_minus_and_plus(path):
    result = query_file(path, "select -id as a, +id as b, -(id + 1) as c from input")
    assert result.column("a") == [-1, -2, -3, -4, -5]
    assert result.column("b") == [1, 2, 3, 4, 5]
    assert result.column("c") == [-2, -3, -4, -5, -6]
    assert schema_of(result) == [
        ("a", "int64", False),
        ("b", "int64", False),
        ("c", "int64", False),
    ]


def test_int64_stays_int64_float64_widens(path):
    result = query_file(
        path,
        "select id + 1 as i, id + 1.5 as m, f * 2 as g, id / 2 as h from input",
    )
    assert schema_of(result) == [
        ("i", "int64", False),
        ("m", "float64", False),
        ("g", "float64", True),
        ("h", "float64", False),
    ]
    assert result.column("i") == [2, 3, 4, 5, 6]
    assert result.column("m") == [2.5, 3.5, 4.5, 5.5, 6.5]
    assert result.column("h") == [0.5, 1.0, 1.5, 2.0, 2.5]


def test_pure_constant_expression_is_not_nullable(path):
    result = query_file(path, "select 1 + 2 * 3 as c, -1.5 as d from input")
    assert result.column("c") == [7] * 5
    assert result.column("d") == [-1.5] * 5
    assert schema_of(result) == [("c", "int64", False), ("d", "float64", False)]


def test_nullable_derived_from_participating_columns(path):
    result = query_file(
        path, "select id + n as a, id + 1 as b, n + f as c from input"
    )
    assert schema_of(result) == [
        ("a", "int64", True),
        ("b", "int64", False),
        ("c", "float64", True),
    ]
    assert result.column("a") == [11, None, 33, 44, None]


def test_alias_identifier_rules(path):
    result = query_file(path, 'select id + 1 as "weird name" from input')
    assert result.column("weird name") == [2, 3, 4, 5, 6]
    result = query_file(path, "select id + 1 as 名前 from input")
    assert result.column("名前") == [2, 3, 4, 5, 6]


# ---------------------------------------------------------------------------
# Runtime errors
# ---------------------------------------------------------------------------


def test_int64_overflow(tmp_path):
    p = tmp_path / "big.caef"
    write_file(p, Table(Schema([ColumnSchema("x", "int64")]), {"x": [2**63 - 1, -(2**63)]}))
    with pytest.raises(QueryValidationError):
        query_file(p, "select x + 1 as y from input")
    with pytest.raises(QueryValidationError):
        query_file(p, "select x * 2 as y from input")
    with pytest.raises(QueryValidationError):
        query_file(p, "select x - 1 as y from input where x < 0")
    with pytest.raises(QueryValidationError):
        query_file(p, "select -x as y from input where x < 0")


def test_division_by_zero(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id / 0 as x from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id / 0.0 as x from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id / (id - 3) as x from input")


def test_non_finite_float_result(tmp_path):
    p = tmp_path / "f.caef"
    write_file(p, Table(Schema([ColumnSchema("x", "float64")]), {"x": [1e308]}))
    with pytest.raises(QueryValidationError):
        query_file(p, "select x * 10 as y from input")
    with pytest.raises(QueryValidationError):
        query_file(p, "select x + x as y from input")


def test_filtered_rows_do_not_trigger_errors(tmp_path):
    p = tmp_path / "big.caef"
    write_file(p, Table(Schema([ColumnSchema("x", "int64")]), {"x": [1, 2**63 - 1]}))
    # The overflowing row is filtered out by WHERE.
    result = query_file(p, "select x + 1 as y from input where x < 100")
    assert result.column("y") == [2]
    # ... or cut by LIMIT (sort keys come from another column).
    result = query_file(p, "select x + 1 as y from input order by x limit 1")
    assert result.column("y") == [2]
    # Division by zero on a filtered row never happens either.
    result = query_file(p, "select 1 / (x - 1) as y from input where x != 1")
    assert result.column("y") == [1.0 / (2**63 - 2)]


def test_sort_keys_computed_before_limit(tmp_path):
    p = tmp_path / "big.caef"
    write_file(p, Table(Schema([ColumnSchema("x", "int64")]), {"x": [1, 2**63 - 1]}))
    # ORDER BY evaluates the alias on every WHERE-selected row, before LIMIT.
    with pytest.raises(QueryValidationError):
        query_file(p, "select x + 1 as y from input order by y limit 1")


# ---------------------------------------------------------------------------
# WHERE with expressions
# ---------------------------------------------------------------------------


def test_where_expression_operands(path):
    assert query_file(path, "select id from input where n + 1 > 30").column("id") == [3, 4]
    assert query_file(path, "select id from input where n = id * 10").column("id") == [1, 3, 4]
    assert query_file(path, "select id from input where (n + 1) * 2 >= 22").column("id") == [1, 3, 4]
    assert query_file(path, "select id from input where -f < 1").column("id") == [1, 2, 4, 5]


def test_where_expression_null_propagation(path):
    # n is NULL for rows 2 and 5: arithmetic yields NULL, hence UNKNOWN.
    assert query_file(path, "select id from input where n + 1 > 0").column("id") == [1, 3, 4]
    assert query_file(path, "select id from input where n * 2 is null").column("id") == [2, 5]
    assert query_file(path, "select id from input where n + 1 is not null").column("id") == [1, 3, 4]


def test_where_expression_type_compatibility(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where s + 1 = 2")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where n + 1 = 'a'")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where n + 1 = true")
    # int64 expressions and float64 columns remain comparable.
    assert query_file(path, "select id from input where f = id + 0.5").column("id") == [1, 2]


def test_numeric_expression_as_boolean_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where id + 1")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where not id * 2")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where id + 1 and id > 0")


def test_where_expression_errors_surface(tmp_path):
    p = tmp_path / "z.caef"
    write_file(p, Table(Schema([ColumnSchema("x", "int64")]), {"x": [1, 0]}))
    with pytest.raises(QueryValidationError):
        query_file(p, "select x from input where 1 / x > 0")


# ---------------------------------------------------------------------------
# ORDER BY aliases
# ---------------------------------------------------------------------------


def test_order_by_alias(path):
    result = query_file(path, "select id, n + 1 as x from input order by x")
    assert result.column("x") == [11, 31, 41, None, None]
    assert result.column("id") == [1, 3, 4, 2, 5]
    result = query_file(path, "select id, n + 1 as x from input order by x desc")
    assert result.column("x") == [41, 31, 11, None, None]
    result = query_file(
        path, "select id, n + 1 as x from input order by x nulls first"
    )
    assert result.column("x") == [None, None, 11, 31, 41]


def test_order_by_alias_stable(path):
    # Rows 2 and 5 both sort to NULL and keep their original relative order.
    result = query_file(path, "select id, n * 2 as x from input order by x")
    assert result.column("id") == [1, 3, 4, 2, 5]


def test_order_by_alias_shadows_column(path):
    # The alias "id" wins over the input column of the same name.
    result = query_file(path, "select n + 1 as id from input order by id")
    assert result.column("id") == [11, 31, 41, None, None]


def test_order_by_alias_result_type(path):
    result = query_file(path, "select id / 2 as h from input order by h desc limit 1")
    assert result.column("h") == [2.5]
    assert schema_of(result) == [("h", "float64", False)]


def test_order_by_unknown_and_duplicate(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id + 1 as x from input order by nope")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id + 1 as x from input order by x, x")


def test_duplicate_alias_and_output_name(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id + 1 as x, id * 2 as x from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id, n + 1 as id from input")


# ---------------------------------------------------------------------------
# Aggregate queries stay expression-free
# ---------------------------------------------------------------------------


def test_aggregate_query_rejects_scalar_expressions(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select n + 1 as x from input group by n")
    with pytest.raises(QueryValidationError):
        query_file(path, "select sum(id), n + 1 as x from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select sum(id + 1) from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(n * 2) from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input group by id + 1")
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(*) + 1 as x from input")


def test_aggregate_query_where_expression_allowed(path):
    result = query_file(
        path, "select count(*) from input where n + 1 > 20"
    )
    assert result.column("COUNT(*)") == [2]
    result = query_file(
        path, "select s, sum(id) from input where id * 2 > 4 group by s order by s"
    )
    assert result.column("s") == ["a", "c", None]
    assert result.column("SUM(id)") == [4, 5, 3]


# ---------------------------------------------------------------------------
# Syntax errors (raised before any file is read)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "select id + 1 from input",          # computed expression without AS
        "select id + 1 as from input",       # AS without an alias
        "select id + 1 as 5 from input",     # alias must be an identifier
        "select (id + 1 as x from input",    # unbalanced parentheses
        "select id + as x from input",       # missing operand
        "select id * as x from input",       # missing operand
        "select id % 2 as x from input",     # unsupported operator
        "select id ** 2 as x from input",    # unsupported operator
        "select id from input where id + ",  # missing operand in WHERE
        "select id from input where (id = 1",  # unbalanced parentheses
        "select id as x from input",         # bare column alias unsupported
    ],
)
def test_scalar_expression_syntax_errors_before_file_access(sql):
    with pytest.raises(QuerySyntaxError):
        query_file("/nonexistent/t.caef", sql)


def test_syntax_error_before_file_access_join():
    missing = {"l": "/nonexistent/l.caef", "r": "/nonexistent/r.caef"}
    with pytest.raises(QuerySyntaxError):
        query_files(missing, "select l.id + as x from l inner join r on l.id = r.id")


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


def test_join_scalar_expressions(join_paths):
    result = query_files(
        join_paths, "select l.a + r.b as s from l inner join r on l.k = r.k"
    )
    assert result.column("s") == [110, None, 330]
    assert schema_of(result) == [("s", "int64", True)]
    result = query_files(
        join_paths,
        "select l.k, l.a * 2 as d from l left join r on l.k = r.k "
        "where r.b + 1 > 100 or r.b is null",
    )
    assert result.column("l.k") == [1, 1, 2, 3]
    assert result.column("d") == [20, 20, 40, 60]


def test_join_order_by_alias(join_paths):
    result = query_files(
        join_paths,
        "select l.a + 1 as z from l inner join r on l.k = r.k order by z desc",
    )
    assert result.column("z") == [31, 11, 11]


def test_join_expression_requires_qualified_columns(join_paths):
    with pytest.raises(QueryValidationError):
        query_files(join_paths, "select a + 1 as z from l inner join r on l.k = r.k")
    with pytest.raises(QueryValidationError):
        query_files(
            join_paths,
            "select l.a from l inner join r on l.k = r.k where b + 1 > 0",
        )


def test_join_single_source_expressions(join_paths):
    result = query_files(join_paths, "select a + 1 as z from l")
    assert result.column("z") == [11, 21, 31]


# ---------------------------------------------------------------------------
# CLI and determinism
# ---------------------------------------------------------------------------


def test_cli_query_scalar_expression(path, capsys):
    code = main(["query", str(path), "select id, n + 1 as x from input order by x desc limit 2"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [
        {"name": "id", "type": "int64", "nullable": False},
        {"name": "x", "type": "int64", "nullable": True},
    ]
    assert payload["rows"] == [[4, 41], [3, 31]]


def test_cli_query_files_scalar_expression(join_paths, capsys):
    sources = json.dumps({"l": str(join_paths["l"]), "r": str(join_paths["r"])})
    code = main(
        ["query-files", sources, "select l.a + r.b as s from l inner join r on l.k = r.k"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [{"name": "s", "type": "int64", "nullable": True}]
    assert payload["rows"] == [[110], [None], [330]]


def test_cli_scalar_expression_exit_codes(path, capsys):
    assert main(["query", str(path), "select id / 0 as x from input"]) == 2
    assert main(["query", str(path), "select id + as x from input"]) == 2
    assert main(["query", str(path), "select id + 1 as x from input group by id"]) == 2


def test_repeated_execution_is_deterministic(path):
    sql = "select id, n * 2 + 1 as x from input where id > 1 order by x desc limit 3"
    first = query_file(path, sql)
    second = query_file(path, sql)
    assert first.columns == second.columns
    assert schema_of(first) == schema_of(second)
