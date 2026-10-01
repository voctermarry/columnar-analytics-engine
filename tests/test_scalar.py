"""Tests for numeric scalar expressions in SELECT and WHERE."""

from __future__ import annotations

import pytest

from columnar_analytics import (
    ColumnSchema,
    QuerySyntaxError,
    QueryValidationError,
    Schema,
    Table,
    query_files,
    write_file,
)
from columnar_analytics.query import query_file, query_table


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
    p = tmp_path / "scalar.caef"
    write_file(p, Table(SCHEMA, DATA))
    return p


def rows(result):
    return [
        [result._columns[c][i] for c in range(len(result.schema.columns))]
        for i in range(result.row_count)
    ]


# ---------------------------------------------------------------------------
# SELECT: types, precedence and nullability
# ---------------------------------------------------------------------------


def test_arithmetic_operators_and_precedence(path):
    result = query_file(
        path,
        "select n + 1 as a, n - 1 as b, n * 2 as c, n / 2 as d, "
        "-n as e, +f as g, (n + 2) * 3 as h, 2 + 3 * 4 as i from input order by id",
    )
    assert rows(result) == [
        [11, 9, 20, 5.0, -10, 1.5, 36, 14],
        [None, None, None, None, None, 2.5, None, 14],
        [31, 29, 60, 15.0, -30, None, 96, 14],
        [41, 39, 80, 20.0, -40, -0.5, 126, 14],
        [None, None, None, None, None, 10.0, None, 14],
    ]


def test_result_types_int64_vs_float64(path):
    result = query_file(
        path,
        "select id + 1 as i, id - 1 as j, id * 2 as k, "
        "id / 2 as d, id + 1.0 as fl, f * 2 as f2, -id as neg from input limit 1",
    )
    cols = {c.name: c for c in result.schema.columns}
    assert cols["i"].type == "int64"
    assert cols["j"].type == "int64"
    assert cols["k"].type == "int64"
    assert cols["neg"].type == "int64"
    assert cols["d"].type == "float64"
    assert cols["fl"].type == "float64"
    assert cols["f2"].type == "float64"


def test_nullable_derivation(path):
    result = query_file(
        path,
        "select id + 1 as a, n + 1 as b, f * 2 as c, "
        "n + f as d, 1 + 2 as k, 3 / 2 as q, 2.5 as fl from input limit 1",
    )
    cols = {c.name: c for c in result.schema.columns}
    assert cols["a"].nullable is False  # non-nullable input column
    assert cols["b"].nullable is True  # nullable int64
    assert cols["c"].nullable is True  # nullable float64
    assert cols["d"].nullable is True
    # Pure constants are never nullable.
    assert cols["k"].nullable is False
    assert cols["q"].nullable is False
    assert cols["fl"].nullable is False


def test_output_keeps_written_order_and_bare_names(path):
    result = query_file(path, "select n + 1 as z, id, s, id * 2 as q from input limit 2")
    assert result.column_names == ("z", "id", "s", "q")
    assert rows(result) == [[11, 1, "a", 2], [None, 2, "b", 4]]


def test_null_propagates_through_arithmetic(path):
    result = query_file(path, "select n + 5 as a, n * f as b, f / 2 as c from input order by id")
    assert result.column("a") == [15, None, 35, 45, None]
    assert result.column("b") == [15.0, None, None, -20.0, None]
    assert result.column("c") == [0.75, 1.25, None, -0.25, 5.0]


def test_division_always_float(path):
    result = query_file(path, "select id / 2 as d from input order by id")
    assert result.column("d") == [0.5, 1.0, 1.5, 2.0, 2.5]
    assert result.schema.columns[0].type == "float64"


def test_unary_chaining_and_literals(path):
    result = query_file(path, "select --id as a, -+-id as b, -5 as c, +1.5 as d from input limit 1")
    assert rows(result) == [[1, 1, -5, 1.5]]


def test_int64_literal_min_value(path):
    result = query_file(path, "select -9223372036854775808 as x from input limit 1")
    assert result.column("x") == [-(2**63)]


def test_expression_on_query_table():
    table = Table(SCHEMA, DATA)
    result = query_table(table, "select id * 10 as big, -f as nf from input order by id")
    assert result.column("big") == [10, 20, 30, 40, 50]
    assert result.column("nf") == [-1.5, -2.5, None, 0.5, -10.0]


# ---------------------------------------------------------------------------
# Runtime arithmetic errors
# ---------------------------------------------------------------------------


def _file(tmp_path, values, col="x", typ="int64"):
    p = tmp_path / "a.caef"
    write_file(p, Table(Schema([ColumnSchema(col, typ, nullable=True)]), {col: values}))
    return p


def test_int64_add_overflow(tmp_path):
    p = _file(tmp_path, [2**63 - 1])
    with pytest.raises(QueryValidationError):
        query_file(p, "select x + 1 as y from input")


def test_int64_multiply_overflow(tmp_path):
    p = _file(tmp_path, [2**32])
    with pytest.raises(QueryValidationError):
        query_file(p, "select x * x as y from input")


def test_unary_overflow(tmp_path):
    with pytest.raises(QueryValidationError):
        query_file(_file(tmp_path, [1]), "select -(-9223372036854775808) as y from input")


def test_integer_division_by_zero(tmp_path):
    p = _file(tmp_path, [0])
    with pytest.raises(QueryValidationError):
        query_file(p, "select 1 / x as y from input")
    with pytest.raises(QueryValidationError):
        query_file(p, "select 1.0 / x as y from input")


def test_float_non_finite_result(tmp_path):
    p = _file(tmp_path, [1.7976931348623157e308], typ="float64")
    with pytest.raises(QueryValidationError):
        query_file(p, "select x * x as y from input")


def test_null_operand_does_not_divide_by_zero(tmp_path):
    p = _file(tmp_path, [0, None])
    result = query_file(p, "select 1 / x as y from input where x is null")
    assert result.column("y") == [None]


# ---------------------------------------------------------------------------
# Lazy evaluation: WHERE -> sort -> LIMIT -> SELECT expressions
# ---------------------------------------------------------------------------


def test_filtered_rows_do_not_trigger_select_divzero(path):
    # id = 3 would divide 1/0, but it is filtered out.
    result = query_file(path, "select 1 / (id - 3) as x from input where id = 1")
    assert result.column("x") == [-0.5]


def test_limited_rows_do_not_trigger_select_divzero(path):
    result = query_file(path, "select 1 / (id - 3) as x from input order by id limit 2")
    assert result.column("x") == [-0.5, -1.0]


def test_row_after_limit_triggers_no_error_with_constant_divzero(path):
    assert query_file(path, "select 1 / 0 as x from input limit 0").row_count == 0
    assert query_file(path, "select 1 / 0 as x from input where id > 1000").row_count == 0


def test_surviving_row_still_triggers_divzero(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select 1 / (id - 3) as x from input order by id limit 4")


def test_where_short_circuit_skips_unevaluated_division(path):
    # AND short-circuits; rows 4 and 5 satisfy the left term but would make
    # the right side UNKNOWN (NULL f), and no row reaches the id=3 division.
    result = query_file(
        path, "select id from input where id < 4 and f / (id - 3) < 0"
    )
    assert result.column("id") == [1, 2]


def test_where_division_by_zero_still_raises(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where 1 / (id - 3) < 0")


# ---------------------------------------------------------------------------
# WHERE expressions
# ---------------------------------------------------------------------------


def test_where_arithmetic_comparisons(path):
    assert query_file(path, "select id from input where (n + 1) > 30").column("id") == [3, 4]
    assert query_file(path, "select id from input where n / 2 = 5.0").column("id") == [1]
    assert query_file(path, "select id from input where 2 * id < 6").column("id") == [1, 2]
    assert query_file(path, "select id from input where id - 1 >= 3").column("id") == [4, 5]


def test_where_arithmetic_three_valued_logic(path):
    # n + 1 is UNKNOWN for NULL n rows, so they are excluded by an AND term.
    assert query_file(path, "select id from input where n + 1 > 0 or id = 2").column("id") == [1, 2, 3, 4]
    assert query_file(path, "select id from input where n + 1 > 0 and id >= 3").column("id") == [3, 4]


def test_where_parenthesised_predicates(path):
    result = query_file(path, "select id from input where ((id > 1) AND (id < 4))")
    assert result.column("id") == [2, 3]
    result = query_file(path, "select id from input where not (id + 1 > 4)")
    assert result.column("id") == [1, 2, 3]


def test_numeric_expression_as_boolean_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where id + 1")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where n")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where not (id + 1)")


def test_arithmetic_operands_must_be_numeric(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where s + 1 = 2")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where flag + 1 = 2")


def test_comparison_type_rules_apply_to_expressions(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where (id + 1) = 'x'")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where (s = 'x') + 1 > 0")


def test_is_null_does_not_accept_expression(path):
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select id from input where (id + 1) is null")


# ---------------------------------------------------------------------------
# ORDER BY aliases
# ---------------------------------------------------------------------------


def test_order_by_select_alias(path):
    result = query_file(path, "select n * 2 as dbl from input order by dbl nulls first")
    assert result.column("dbl") == [None, None, 20, 60, 80]
    result = query_file(path, "select n * 2 as dbl from input order by dbl desc nulls last")
    assert result.column("dbl") == [80, 60, 20, None, None]


def test_alias_shadows_input_column(path):
    result = query_file(path, "select id + 100 as id from input order by id")
    assert result.column("id") == [101, 102, 103, 104, 105]


def test_order_by_alias_float_type(path):
    result = query_file(path, "select id / 2.0 as h from input order by h desc")
    assert result.column("h") == [2.5, 2.0, 1.5, 1.0, 0.5]


def test_order_by_input_column_still_allowed(path):
    # s: a,b,NULL,a,c. DESC, NULLs last: c(5), b(2), a(1), a(4), NULL(3).
    result = query_file(path, "select id + 1 as z from input order by s desc")
    assert result.column("z") == [6, 3, 2, 5, 4]


def test_unknown_order_reference(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id + 1 as z from input order by nope")


# ---------------------------------------------------------------------------
# Aggregate / GROUP BY rejection
# ---------------------------------------------------------------------------


def test_scalar_rejected_in_aggregate_query(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select n + 1 as x, count(*) from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select n + 1 as x from input group by id")


def test_group_by_rejects_expression(path):
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select id from input group by n + 1")


def test_aggregate_argument_rejects_expression(path):
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select sum(n + 1) from input")
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select count(n + 1) from input")


def test_aggregate_plus_operator_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(*) + 1 as x from input")


# ---------------------------------------------------------------------------
# Alias / result-name validation
# ---------------------------------------------------------------------------


def test_duplicate_aliases_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select n + 1 as x, f + 1 as x from input")


def test_alias_duplicating_bare_column_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select n + 1 as id, id from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id, n + 1 as id from input")


def test_quoted_alias(path):
    result = query_file(path, 'select id + 1 as "x y" from input limit 1')
    assert result.column_names == ("x y",)
    result = query_file(path, 'select id + 1 as "名前" from input limit 1')
    assert result.column_names == ("名前",)


def test_computed_expression_must_be_numeric(path):
    # A parenthesised bool/utf8 value is an expression but not numeric.
    with pytest.raises(QueryValidationError):
        query_file(path, "select (flag) as x from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select (s) as x from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select true as x from input")


# ---------------------------------------------------------------------------
# Syntax errors, all recognised before file access
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "select id + 1 from input",
        "select id + 1 as from input",
        "select id + as x from input",
        "select + as x from input",
        "select - as x from input",
        "select id * as x from input",
        "select id / as x from input",
        "select id + + from input",
        "select (id + 1 as x from input",
        "select id + 1) as x from input",
        "select id % 2 as x from input",
        "select id ** 2 as x from input",
        "select id + 1 as 9 from input",
        "select * + 1 as x from input",
        "select (id + 1 as x from input",
        "select id + 1 as x from input where (id + 1",
    ],
)
def test_scalar_syntax_errors(path, sql):
    with pytest.raises(QuerySyntaxError):
        query_file(path, sql)


def test_scalar_syntax_checked_before_file(tmp_path):
    missing = tmp_path / "nope.caef"
    with pytest.raises(QuerySyntaxError):
        query_file(missing, "select id + as x from input")
    with pytest.raises(QuerySyntaxError):
        query_file(missing, "select (id + 1 as x from input")


def test_unsupported_function_in_scalar(path):
    with pytest.raises((QuerySyntaxError, QueryValidationError)):
        query_file(path, "select foo(id) as x from input")


# ---------------------------------------------------------------------------
# Two-file queries
# ---------------------------------------------------------------------------


LEFT_SCHEMA = Schema(
    [ColumnSchema("id", "int64"), ColumnSchema("k", "int64", nullable=True)]
)
RIGHT_SCHEMA = Schema(
    [
        ColumnSchema("rid", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("v", "float64", nullable=True),
        ColumnSchema("tag", "utf8", nullable=True),
    ]
)


@pytest.fixture()
def two(tmp_path):
    lp = tmp_path / "l.caef"
    rp = tmp_path / "r.caef"
    write_file(lp, Table(LEFT_SCHEMA, {"id": [1, 2, 3], "k": [10, 20, 10]}))
    write_file(
        rp,
        Table(
            RIGHT_SCHEMA,
            {"rid": [100, 200], "k": [10, 10], "v": [1.0, 2.0], "tag": ["x", "y"]},
        ),
    )
    return {"l": lp, "r": rp}


def test_join_scalar_projection_and_alias_order(two):
    result = query_files(
        two,
        "SELECT l.id + r.rid AS total FROM l INNER JOIN r ON l.k = r.k "
        "WHERE l.id * 2 > 1 ORDER BY total",
    )
    assert result.column_names == ("total",)
    assert result.column("total") == [101, 103, 201, 203]
    assert result.schema.columns[0].type == "int64"
    assert result.schema.columns[0].nullable is False


def test_left_join_scalar_nullable_and_where(two):
    result = query_files(
        two,
        "SELECT l.id, r.v + 1.0 AS bumped FROM l LEFT JOIN r ON l.k = r.k "
        "ORDER BY l.id",
    )
    col = result.schema.columns[1]
    assert col.type == "float64" and col.nullable is True
    assert result.column("bumped") == [2.0, 3.0, None, 2.0, 3.0]


def test_join_scalar_requires_qualification(two):
    with pytest.raises(QueryValidationError):
        query_files(two, "SELECT id + 1 AS x FROM l INNER JOIN r ON l.k = r.k")
    with pytest.raises(QueryValidationError):
        query_files(two, "SELECT l.id + rid AS x FROM l INNER JOIN r ON l.k = r.k")


def test_join_scalar_type_validation(two):
    with pytest.raises(QueryValidationError):
        query_files(two, "SELECT l.id + r.tag AS x FROM l INNER JOIN r ON l.k = r.k")


def test_join_order_by_alias_skips_qualification(two):
    result = query_files(two, "SELECT l.id + 10 AS id FROM l ORDER BY id")
    assert result.column("id") == [11, 12, 13]
