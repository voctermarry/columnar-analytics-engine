"""Tests for the single-file SQL query layer and the ``query`` CLI command."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from columnar_analytics import (
    ColumnSchema,
    ColumnarFormatError,
    QuerySyntaxError,
    QueryValidationError,
    Schema,
    Table,
    query_file,
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
        ColumnSchema("flag_n", "bool", nullable=True),
        ColumnSchema("名前", "utf8", nullable=True),
    ]
)

DATA = {
    "id": [1, 2, 3, 4, 5],
    "n": [10, None, 30, 40, None],
    "f": [1.5, 2.5, None, -0.5, 10.0],
    "s": ["a", "b", None, "a", "c"],
    "flag": [True, False, True, False, True],
    "flag_n": [True, False, None, True, None],
    "名前": ["い", "ろ", None, "は", "へ"],
}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "t.caef"
    write_file(p, Table(SCHEMA, DATA), compression="zlib", dictionary_encoding=["s", "名前"])
    return p


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def test_select_star(path):
    result = query_file(path, "SELECT * FROM input")
    assert result.schema == SCHEMA
    assert result.column_names == SCHEMA.names
    assert result.row_count == 5
    assert result.column("id") == DATA["id"]


def test_select_projection_order_and_subset(path):
    result = query_file(path, "select s, id, flag from input")
    assert result.column_names == ("s", "id", "flag")
    assert result.column("id") == DATA["id"]
    assert result.column("s") == DATA["s"]
    assert result.column("flag") == DATA["flag"]


def test_keywords_case_insensitive(path):
    result = query_file(path, "SeLeCt id FrOm InPuT wHeRe id = 1")
    assert result.column("id") == [1]
    # Column names are exact Unicode: keyword case rules do not apply to them.
    with pytest.raises(QueryValidationError):
        query_file(path, "select ID from input")


def test_unicode_identifier_exact_match(path):
    result = query_file(path, 'select "名前" from input')
    assert result.column("名前") == DATA["名前"]


def test_quoted_identifier_can_have_keyword_spelling(path):
    # A quoted keyword-named column resolves as an identifier; the schema has
    # no such column, so this must be a validation error.
    with pytest.raises(QueryValidationError):
        query_file(path, 'select "select" from input')


def test_quoted_identifier_doubled_quote(path, tmp_path):
    schema = Schema([ColumnSchema('a"b', "int64")])
    p = tmp_path / "q.caef"
    write_file(p, Table(schema, {'a"b': [7]}))
    result = query_file(p, 'select "a""b" from input')
    assert result.column('a"b') == [7]


def test_duplicate_projection_column(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id, n, id from input")


def test_unknown_projection_column(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select missing from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select _nope from input")  # underscore-leading bare identifier


def test_wrong_table_name(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select * from output")
    # The fixed table name matches case-insensitively when written bare ...
    result = query_file(path, "select id from InPuT")
    assert result.column("id") == [1, 2, 3, 4, 5]
    # ... and an exact (case-sensitive) match also works when double-quoted.
    result = query_file(path, 'select id from "input"')
    assert result.row_count == 5
    with pytest.raises(QueryValidationError):
        query_file(path, 'select id from "Input"')


def test_no_alias_supported(path):
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select id x from input")
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select id as x from input")


def test_no_expressions_in_projection(path):
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select id + 1 from input")


def test_star_cannot_mix_with_columns(path):
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select *, id from input")


# ---------------------------------------------------------------------------
# WHERE: comparisons and types
# ---------------------------------------------------------------------------


def test_int64_comparisons(path):
    assert query_file(path, "select id from input where id >= 3").column("id") == [3, 4, 5]
    assert query_file(path, "select id from input where id != 3").column("id") == [1, 2, 4, 5]
    assert query_file(path, "select id from input where id < 2").column("id") == [1]
    assert query_file(path, "select id from input where id <= 2").column("id") == [1, 2]
    assert query_file(path, "select id from input where id > 4").column("id") == [5]
    assert query_file(path, "select id from input where id = 4").column("id") == [4]


def test_int_float_cross_comparison(path):
    assert query_file(path, "select id from input where f = 1.5").column("id") == [1]
    assert query_file(path, "select id from input where id > 2.0").column("id") == [3, 4, 5]
    assert query_file(path, "select id from input where 2.5 > f").column("id") == [1, 4]
    assert query_file(path, "select id from input where n = 10.0").column("id") == [1]


def test_utf8_comparisons(path):
    assert query_file(path, "select id from input where s = 'a'").column("id") == [1, 4]
    assert query_file(path, "select id from input where s != 'a'").column("id") == [2, 5]
    assert query_file(path, "select id from input where s < 'c'").column("id") == [1, 2, 4]
    assert query_file(path, "select id from input where s >= 'b'").column("id") == [2, 5]
    assert query_file(path, "select id from input where 'b' <= s").column("id") == [2, 5]


def test_string_literal_escapes(path):
    assert query_file(path, "select id from input where s = 'it''s'").column("id") == []
    # Doubled quote is a literal apostrophe: craft a matching row.
    import columnar_analytics as cae

    schema = Schema([ColumnSchema("t", "utf8")])
    p = path.parent / "quoted.caef"
    cae.write_file(p, Table(schema, {"t": ["it's"]}))
    assert query_file(p, "select t from input where t = 'it''s'").column("t") == ["it's"]
    result = query_file(path, "select id from input where s != 'z'")
    assert result.column("id") == [1, 2, 4, 5]  # NULL excluded via UNKNOWN


def test_bool_equality(path):
    assert query_file(path, "select id from input where flag = true").column("id") == [1, 3, 5]
    assert query_file(path, "select id from input where FALSE = flag").column("id") == [2, 4]
    assert query_file(path, "select id from input where flag != true").column("id") == [2, 4]


@pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
def test_bool_ordering_rejected(path, op):
    with pytest.raises(QueryValidationError):
        query_file(path, f"select id from input where flag {op} true")


@pytest.mark.parametrize(
    "left,right",
    [
        ("s", "1"),
        ("1", "s"),
        ("s", "1.5"),
        ("flag", "1"),
        ("flag", "'x'"),
        ("n", "true"),
        ("f", "'x'"),
    ],
)
def test_incompatible_comparison_types(path, left, right):
    with pytest.raises(QueryValidationError):
        query_file(path, f"select id from input where {left} = {right}")


def test_unknown_column_in_where(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where missing = 1")


# ---------------------------------------------------------------------------
# WHERE: IS NULL / NOT / AND / OR and three-valued logic
# ---------------------------------------------------------------------------


def test_is_null_and_is_not_null(path):
    assert query_file(path, "select id from input where n is null").column("id") == [2, 5]
    assert query_file(path, "select id from input where n is not null").column("id") == [1, 3, 4]
    assert query_file(path, "select id from input where NOT s IS NULL").column("id") == [1, 2, 4, 5]
    assert query_file(path, "select id from input where not n is not null").column("id") == [2, 5]


def test_nullable_bool_as_logical_operand(path):
    # flag_n: True, False, NULL, True, NULL
    assert query_file(path, "select id from input where not flag_n").column("id") == [2]
    assert query_file(path, "select id from input where flag_n").column("id") == [1, 4]
    assert query_file(path, "select id from input where flag_n or id = 5").column("id") == [1, 4, 5]


def test_and_or_precedence(path):
    # AND binds tighter than OR: (id = 1) OR (id = 2 AND s = 'b')
    result = query_file(path, "select id from input where id = 1 or id = 2 and s = 'b'")
    assert result.column("id") == [1, 2]
    result = query_file(path, "select id from input where (id = 1 or id = 2) and s = 'b'")
    assert result.column("id") == [2]


def test_not_precedence_over_comparison(path):
    # NOT binds tighter than comparison, so without parentheses the parser
    # treats the negated factor as a comparison operand, which is not a
    # column/literal and must be rejected.
    with pytest.raises((QuerySyntaxError, QueryValidationError)):
        query_file(path, "select id from input where not flag = true")
    # Parentheses give the usual SQL reading: NOT (comparison).
    result = query_file(path, "select id from input where not (flag = true)")
    assert result.column("id") == [2, 4]
    # IS [NOT] NULL is an atom postfix and binds tighter than NOT.
    assert query_file(path, "select id from input where not n is null").column("id") == [1, 3, 4]


def test_comparison_precedence_over_and(path):
    result = query_file(path, "select id from input where id > 2 and id < 5")
    assert result.column("id") == [3, 4]


def test_three_valued_logic_null_propagation(path):
    # n = 10 is UNKNOWN for NULL rows; combined with OR only known-true rows win.
    assert query_file(path, "select id from input where n = 10 or id = 4").column("id") == [1, 4]
    assert query_file(path, "select id from input where n = 10 and id >= 1").column("id") == [1]
    assert query_file(path, "select id from input where not (n = 10)").column("id") == [3, 4]
    # NULL AND FALSE -> FALSE (row 2); UNKNOWN AND TRUE -> UNKNOWN (rows 2,5 handled by id)
    assert query_file(
        path, "select id from input where n = 10 or n = 99 or id = 5"
    ).column("id") == [1, 5]


def test_comparison_against_null_literal_is_never_true(path):
    # NULL literals are not part of the grammar, but column NULLs propagate.
    result = query_file(path, "select id from input where s = 'a' or n is null")
    assert result.column("id") == [1, 2, 4, 5]


def test_nested_parentheses(path):
    result = query_file(path, "select id from input where ((id = 1) OR (id = 5)) AND flag = TRUE")
    assert result.column("id") == [1, 5]


def test_missing_where_keeps_all_rows(path):
    assert query_file(path, "select id from input").column("id") == [1, 2, 3, 4, 5]


def test_row_order_preserved(path):
    result = query_file(path, "select id from input where f is not null")
    assert result.column("id") == [1, 2, 4, 5]


def test_logical_operands_must_be_boolean(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where id")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where id and flag")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where s or flag")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where not id")


# ---------------------------------------------------------------------------
# Literals
# ---------------------------------------------------------------------------


def test_integer_literal_bounds(path):
    assert query_file(path, "select id from input where id = -2147483649").column("id") == []
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select id from input where id = 9223372036854775808")
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select id from input where id = -9223372036854775809")
    # The boundary value itself is representable.
    assert query_file(path, "select id from input where id != -9223372036854775808").column("id") == [1, 2, 3, 4, 5]


def test_float_literals(path):
    assert query_file(path, "select id from input where f = .5").column("id") == []
    assert query_file(path, "select id from input where f >= -.5").column("id") == [1, 2, 4, 5]
    assert query_file(path, "select id from input where 1.5e0 = f").column("id") == [1]
    assert query_file(path, "select id from input where f = 10E0").column("id") == [5]
    assert query_file(path, "select id from input where f = +1.5").column("id") == [1]


def test_non_finite_float_literals_rejected(path):
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select id from input where f = 1e999")


def test_unicode_string_literal(path):
    assert query_file(path, "select id from input where 名前 = 'へ'").column("id") == [5]
    assert query_file(path, "select id from input where \"名前\" = 'ろ'").column("id") == [2]


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   ",
        "SELECT",
        "SELECT *",
        "SELECT * FROM",
        "SELECT * FROM input WHERE",
        "SELECT * FROM input WHERE id = ",
        "SELECT * FROM input WHERE = id",
        "SELECT * FROM input WHERE (id = 1",
        "SELECT * FROM input WHERE id = 1)",
        "SELECT * FROM input WHERE id = 1 AND",
        "SELECT * FROM input WHERE NOT",
        "SELECT * FROM input WHERE id IS",
        "SELECT * FROM input WHERE id IS TRUE",
        "SELECT * FROM input WHERE id = 1;",
        "* FROM input",
        "select ** from input",
        "select a,,b from input",
        "select * from input where id = @",
        "select * from input where id ~ 1",
        "select * from input where id = '' 'x'",
        "select * from input where id = 'unterminated",
        "select * from input where id = \"unterminated",
        "INSERT INTO input VALUES (1)",
        "select * from input where id = 1 or",
        "select * from input where id == 1",
        "select * from input where id =! 1",
        "select * from input where id <> 1 1",
        "select * from input where id = 1e",
        "select * from 'input'",
        "select '' from input",
    ],
)
def test_syntax_errors(path, sql):
    with pytest.raises(QuerySyntaxError):
        query_file(path, sql)


def test_empty_sql_is_syntax_error(path):
    with pytest.raises(QuerySyntaxError):
        query_file(path, "")


def test_non_string_sql():
    with pytest.raises(QuerySyntaxError):
        query_file("/nonexistent", 42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ORDER BY
# ---------------------------------------------------------------------------


def test_order_by_int_default_asc(path):
    result = query_file(path, "select id from input order by n")
    # n: 10, NULL, 30, 40, NULL -- NULLs sort last by default.
    assert result.column("id") == [1, 3, 4, 2, 5]


def test_order_by_explicit_asc(path):
    result = query_file(path, "select id from input order by n asc")
    assert result.column("id") == [1, 3, 4, 2, 5]


def test_order_by_desc(path):
    result = query_file(path, "select id from input order by id desc")
    assert result.column("id") == [5, 4, 3, 2, 1]


def test_order_by_desc_nulls_last_by_default(path):
    # DESC reverses values but NULLs stay at the end.
    result = query_file(path, "select id from input order by n desc")
    assert result.column("id") == [4, 3, 1, 2, 5]


def test_order_by_nulls_first_asc(path):
    result = query_file(path, "select id from input order by n asc nulls first")
    assert result.column("id") == [2, 5, 1, 3, 4]


def test_order_by_nulls_first_desc(path):
    result = query_file(path, "select id from input order by n desc nulls first")
    assert result.column("id") == [2, 5, 4, 3, 1]


def test_order_by_nulls_last_explicit(path):
    result = query_file(path, "select id from input order by n desc nulls last")
    assert result.column("id") == [4, 3, 1, 2, 5]


def test_order_by_float(path):
    # f: 1.5, 2.5, NULL, -0.5, 10.0
    result = query_file(path, "select id from input order by f")
    assert result.column("id") == [4, 1, 2, 5, 3]
    result = query_file(path, "select id from input order by f desc nulls first")
    assert result.column("id") == [3, 5, 2, 1, 4]


def test_order_by_utf8_codepoint(path):
    # 名前: い U+3044, ろ U+308D, NULL, は U+306F, へ U+3078
    result = query_file(path, 'select id from input order by "名前"')
    assert result.column("id") == [1, 4, 5, 2, 3]


def test_order_by_bool_false_before_true(path):
    # flag: T, F, T, F, T
    result = query_file(path, "select id from input order by flag")
    assert result.column("id") == [2, 4, 1, 3, 5]
    # DESC reverses the groups but ties still keep file order.
    result = query_file(path, "select id from input order by flag desc")
    assert result.column("id") == [1, 3, 5, 2, 4]


def test_order_by_nullable_bool(path):
    # flag_n: T, F, NULL, T, NULL
    result = query_file(path, "select id from input order by flag_n")
    assert result.column("id") == [2, 1, 4, 3, 5]
    result = query_file(path, "select id from input order by flag_n nulls first")
    assert result.column("id") == [3, 5, 2, 1, 4]


def test_order_by_multiple_columns(path):
    # s: a, b, NULL, a, c
    result = query_file(path, "select id from input order by s, id")
    assert result.column("id") == [1, 4, 2, 5, 3]
    result = query_file(path, "select id from input order by s asc, id desc")
    assert result.column("id") == [4, 1, 2, 5, 3]
    result = query_file(
        path, "select id from input order by s desc nulls first, id"
    )
    assert result.column("id") == [3, 5, 2, 1, 4]


def test_order_by_ties_keep_file_order(tmp_path):
    schema = Schema([ColumnSchema("k", "int64"), ColumnSchema("v", "int64")])
    p = tmp_path / "ties.caef"
    write_file(p, Table(schema, {"k": [1, 1, 1, 1], "v": [40, 10, 30, 20]}))
    result = query_file(p, "select v from input order by k")
    assert result.column("v") == [40, 10, 30, 20]
    # A second key breaks ties without disturbing the semantics of the first.
    result = query_file(p, "select v from input order by k, v desc")
    assert result.column("v") == [40, 30, 20, 10]


def test_order_by_after_where_keeps_filtered_file_order(path):
    # Remaining ids 2(F), 3(T), 4(F), 5(T); FALSE first, ties keep file order.
    result = query_file(
        path, "select id from input where id >= 2 order by flag"
    )
    assert result.column("id") == [2, 4, 3, 5]


def test_order_by_column_need_not_be_projected(path):
    result = query_file(path, "select id from input order by s desc, f asc")
    assert result.column_names == ("id",)
    assert result.column("id") == [5, 2, 4, 1, 3]


def test_order_by_keywords_case_insensitive(path):
    result = query_file(
        path, "select id from input OrDeR By n DeSc NuLlS FiRsT"
    )
    assert result.column("id") == [2, 5, 4, 3, 1]


def test_order_by_unknown_column(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input order by missing")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input order by ID")


def test_order_by_duplicate_column(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input order by id, id")
    with pytest.raises(QueryValidationError):
        query_file(path, 'select id from input order by s, "s"')


# ---------------------------------------------------------------------------
# LIMIT
# ---------------------------------------------------------------------------


def test_limit_alone(path):
    result = query_file(path, "select id from input limit 2")
    assert result.column("id") == [1, 2]


def test_limit_zero_keeps_columns(path):
    result = query_file(path, "select id, s from input limit 0")
    assert result.column_names == ("id", "s")
    assert result.row_count == 0
    assert result.columns == {"id": [], "s": []}


def test_limit_greater_than_rows(path):
    result = query_file(path, "select id from input limit 100")
    assert result.column("id") == [1, 2, 3, 4, 5]


def test_limit_max_uint63(path):
    result = query_file(path, "select id from input limit 9223372036854775807")
    assert result.column("id") == [1, 2, 3, 4, 5]


def test_limit_with_where_keeps_order(path):
    result = query_file(path, "select id from input where id >= 3 limit 1")
    assert result.column("id") == [3]


def test_order_by_with_limit(path):
    result = query_file(path, "select id from input order by id desc limit 2")
    assert result.column("id") == [5, 4]
    result = query_file(
        path,
        "select id from input where flag = true order by id desc limit 2",
    )
    assert result.column("id") == [5, 3]


def test_limit_zero_after_order(path):
    result = query_file(path, "select id from input order by id limit 0")
    assert result.column_names == ("id",)
    assert result.row_count == 0


def test_limit_on_empty_file(tmp_path):
    p = tmp_path / "empty.caef"
    write_file(p, Table(SCHEMA, {name: [] for name in SCHEMA.names}))
    result = query_file(p, "select id from input order by id desc limit 3")
    assert result.column_names == ("id",)
    assert result.row_count == 0


@pytest.mark.parametrize(
    "sql",
    [
        "select id from input order by",
        "select id from input order",
        "select id from input order by id,",
        "select id from input order by id asc desc",
        "select id from input order by id desc asc",
        "select id from input order by id nulls",
        "select id from input order by id nulls sideways",
        "select id from input order by id first",
        "select id from input order by nulls first",
        "select id from input order by id asc nulls",
        "select id from input order by id nulls first asc",
        "select id from input limit",
        "select id from input limit -1",
        "select id from input limit 1.5",
        "select id from input limit .5",
        "select id from input limit 9223372036854775808",
        "select id from input limit 99999999999999999999999",
        "select id from input limit 1 order by id",
        "select id from input where id > 1 limit 2 order by id",
        "select id from input order by id limit 2 limit 3",
        "select id from input order by id order by n",
        "select id from input order by id where id > 1",
        "select id from input limit '2'",
        "select id from input limit true",
    ],
)
def test_order_limit_syntax_errors(path, sql):
    with pytest.raises(QuerySyntaxError):
        query_file(path, sql)


def test_order_limit_syntax_checked_before_file(tmp_path):
    missing = tmp_path / "nope.caef"
    with pytest.raises(QuerySyntaxError):
        query_file(missing, "select id from input order by bogus limit")
    with pytest.raises(QuerySyntaxError):
        query_file(missing, "select id from input order by limit 2")


# ---------------------------------------------------------------------------
# Aggregation: scalar (no GROUP BY)
# ---------------------------------------------------------------------------


AGG_SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("g", "utf8", nullable=True),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("f", "float64", nullable=True),
        ColumnSchema("b", "bool"),
    ]
)

AGG_DATA = {
    "id": [1, 2, 3, 4, 5, 6],
    "g": ["a", "b", None, "a", "c", "a"],
    "n": [10, None, 30, 40, None, 10],
    "f": [1.5, 2.5, None, -0.5, 10.0, 1.5],
    "b": [True, False, True, False, True, False],
}


@pytest.fixture()
def agg_path(tmp_path):
    p = tmp_path / "agg.caef"
    write_file(p, Table(AGG_SCHEMA, AGG_DATA), compression="zlib")
    return p


def _rows(table):
    return [tuple(row) for row in zip(*table._columns)]


def test_count_star_scalar(agg_path):
    result = query_file(agg_path, "select count(*) from input")
    assert result.column_names == ("COUNT(*)",)
    assert result.schema.columns == (ColumnSchema("COUNT(*)", "int64"),)
    assert result.column("COUNT(*)") == [6]


def test_count_star_ignores_where_only(agg_path):
    result = query_file(agg_path, "select count(*) from input where id > 3")
    assert result.column("COUNT(*)") == [3]


def test_count_column_ignores_nulls(agg_path):
    result = query_file(agg_path, "select count(n), count(f), count(g) from input")
    assert _rows(result) == [(4, 5, 5)]
    # Every count column is a non-nullable int64.
    assert result.schema.columns == (
        ColumnSchema("COUNT(n)", "int64"),
        ColumnSchema("COUNT(f)", "int64"),
        ColumnSchema("COUNT(g)", "int64"),
    )


def test_sum_int_keeps_int_type(agg_path):
    result = query_file(agg_path, "select sum(n) from input")
    assert result.schema.columns == (ColumnSchema("SUM(n)", "int64", nullable=True),)
    assert result.column("SUM(n)") == [90]


def test_sum_float_returns_float(agg_path):
    result = query_file(agg_path, "select sum(f) from input")
    assert result.schema.columns == (ColumnSchema("SUM(f)", "float64", nullable=True),)
    assert result.column("SUM(f)") == [15.0]


def test_avg_always_float(agg_path):
    result = query_file(agg_path, "select avg(n), avg(f) from input")
    assert [c.type for c in result.schema.columns] == ["float64", "float64"]
    assert result.column("AVG(n)") == [22.5]
    assert result.column("AVG(f)") == [3.0]


def test_min_max_all_types(agg_path):
    result = query_file(agg_path, "select min(n), max(n), min(f), max(f), min(g), max(g), min(b), max(b) from input")
    assert [(c.name, c.type, c.nullable) for c in result.schema.columns] == [
        ("MIN(n)", "int64", True),
        ("MAX(n)", "int64", True),
        ("MIN(f)", "float64", True),
        ("MAX(f)", "float64", True),
        ("MIN(g)", "utf8", True),
        ("MAX(g)", "utf8", True),
        ("MIN(b)", "bool", True),
        ("MAX(b)", "bool", True),
    ]
    assert _rows(result) == [(10, 40, -0.5, 10.0, "a", "c", False, True)]


def test_function_names_case_insensitive_and_uppercased(agg_path):
    result = query_file(agg_path, "select CoUnT(*) from input")
    assert result.column_names == ("COUNT(*)",)
    result = query_file(agg_path, 'select AvG("n") from input')
    assert result.column_names == ("AVG(n)",)


def test_scalar_aggregates_combine_in_select_order(agg_path):
    result = query_file(agg_path, "select max(n), count(*), avg(f), min(g) from input")
    assert result.column_names == ("MAX(n)", "COUNT(*)", "AVG(f)", "MIN(g)")
    assert _rows(result) == [(40, 6, 3.0, "a")]


def test_empty_filter_scalar_returns_one_row(agg_path):
    result = query_file(
        agg_path,
        "select count(*), count(n), sum(n), avg(n), min(n), max(n), sum(f), avg(f) from input where id > 1000",
    )
    assert result.row_count == 1
    assert _rows(result) == [(0, 0, None, None, None, None, None, None)]
    # COUNT columns stay non-nullable; everything else is nullable.
    assert [c.nullable for c in result.schema.columns] == [
        False, False, True, True, True, True, True, True
    ]


def test_scalar_aggregation_on_empty_file(tmp_path):
    p = tmp_path / "empty.caef"
    write_file(p, Table(AGG_SCHEMA, {name: [] for name in AGG_SCHEMA.names}))
    result = query_file(p, "select count(*), count(n), sum(n), avg(n), min(n), max(n) from input")
    assert result.row_count == 1
    assert _rows(result) == [(0, 0, None, None, None, None)]


def test_no_group_by_plain_column_without_aggregate_unchanged(agg_path):
    # No aggregate and no GROUP BY stays a plain projection.
    result = query_file(agg_path, "select id from input where id <= 2 order by id desc")
    assert result.column("id") == [2, 1]


def test_no_group_by_rejects_plain_columns_with_aggregate(agg_path):
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select id, count(*) from input")
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select g, max(n), n from input")


def test_no_group_by_requires_an_aggregate_when_aggregating(agg_path):
    # A plain-only select is fine, but mixing is the forbidden case above.
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select g, n from input group by g")


def test_aggregate_argument_types(agg_path):
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select sum(g) from input")
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select avg(g) from input")
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select sum(b) from input")
    # MIN/MAX accept every type.
    result = query_file(agg_path, "select min(g), max(b), min(f), max(n) from input")
    assert result.row_count == 1


def test_aggregate_unknown_column(agg_path):
    for fn in ("count", "sum", "avg", "min", "max"):
        with pytest.raises(QueryValidationError):
            query_file(agg_path, f"select {fn}(missing) from input")


def test_star_only_valid_for_count(agg_path):
    for fn in ("sum", "avg", "min", "max"):
        with pytest.raises(QuerySyntaxError):
            query_file(agg_path, f"select {fn}(*) from input")


@pytest.mark.parametrize(
    "sql",
    [
        "select count() from input",
        "select count(a, b) from input",
        "select count(*) over () from input",
        "select count from input",
        "select avg( ) from input",
        "select count(1) from input",
        "select count('x') from input",
        "select min(n",
        "select max n) from input",
        "select count(*) , from input",
        "select count(*) from input group by",
        "select count(*) from input group by g,",
        "select count(*) from input where id = 1 group",
        "select count(*) from input where id = 1 group g",
        "select count(*) from input order g",
        "select count(*) from input group by g limit 1 order by g",
        "select count(*) from input group by g group by n",
    ],
)
def test_aggregate_syntax_errors(agg_path, sql):
    with pytest.raises(QuerySyntaxError):
        query_file(agg_path, sql)


def test_aggregates_not_allowed_in_where(agg_path):
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select id from input where count(*) > 0")
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select count(*) from input where sum(n) > 1")
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select count(*) from input where n > min(n)")


def test_nested_aggregates_rejected(agg_path):
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select sum(count(*)) from input")
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select max(sum(n)) from input")
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select g, count(*) from input group by g order by sum(count(*))")


def test_aggregate_aliases_rejected_as_validation(agg_path):
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select count(*) as c from input")
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select count(*) c from input")
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select sum(n) total from input")


def test_duplicate_aggregate_result_columns(agg_path):
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select count(*), count(*) from input")
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select sum(n), avg(n), sum(n) from input")


def test_duplicate_group_column(agg_path):
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select count(*) from input group by g, g")


def test_unknown_group_by_column(agg_path):
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select count(*) from input group by missing")


def test_int64_sum_overflow(tmp_path):
    schema = Schema([ColumnSchema("n", "int64")])
    p = tmp_path / "big.caef"
    write_file(p, Table(schema, {"n": [2**63 - 1, 1]}))
    with pytest.raises(QueryValidationError):
        query_file(p, "select sum(n) from input")
    # The boundary sum itself is representable.
    p2 = tmp_path / "boundary.caef"
    write_file(p2, Table(schema, {"n": [2**63 - 2, 1]}))
    assert query_file(p2, "select sum(n) from input").column("SUM(n)") == [2**63 - 1]


def test_int64_sum_negative_overflow(tmp_path):
    schema = Schema([ColumnSchema("n", "int64")])
    p = tmp_path / "neg.caef"
    write_file(p, Table(schema, {"n": [-(2**63), -1]}))
    with pytest.raises(QueryValidationError):
        query_file(p, "select sum(n) from input")


def test_float_sum_non_finite(tmp_path):
    schema = Schema([ColumnSchema("f", "float64", nullable=True)])
    p = tmp_path / "huge.caef"
    write_file(p, Table(schema, {"f": [1e308, 1e308]}))
    with pytest.raises(QueryValidationError):
        query_file(p, "select sum(f) from input")
    with pytest.raises(QueryValidationError):
        query_file(p, "select avg(f) from input")


# ---------------------------------------------------------------------------
# Aggregation: GROUP BY
# ---------------------------------------------------------------------------


def test_group_by_basic(agg_path):
    result = query_file(agg_path, "select g, count(*), sum(n) from input group by g")
    assert result.column_names == ("g", "COUNT(*)", "SUM(n)")
    assert _rows(result) == [("a", 3, 60), ("b", 1, None), (None, 1, 30), ("c", 1, None)]


def test_group_by_first_row_order(agg_path):
    # g first appears as a(1), b(2), NULL(3), c(5); no ORDER BY keeps that order.
    result = query_file(agg_path, "select g, count(*) from input group by g")
    assert result.column("g") == ["a", "b", None, "c"]


def test_null_key_is_one_group(agg_path):
    result = query_file(
        agg_path, "select n, count(*) from input group by n order by count(*) desc, n asc nulls first"
    )
    # n: 10, NULL, 30, 40, NULL, 10 -> NULL(2),10(2),30(1),40(1)
    assert _rows(result) == [(None, 2), (10, 2), (30, 1), (40, 1)]


def test_group_by_multiple_columns(agg_path):
    result = query_file(
        agg_path, "select g, b, count(*) from input group by g, b order by g nulls first, b"
    )
    assert _rows(result) == [
        (None, True, 1),
        ("a", False, 2),
        ("a", True, 1),
        ("b", False, 1),
        ("c", True, 1),
    ]


def test_group_by_preserves_group_column_schema(agg_path):
    result = query_file(agg_path, "select g, n, count(*) from input group by g, n")
    by_name = {c.name: c for c in result.schema.columns}
    assert (by_name["g"].type, by_name["g"].nullable) == ("utf8", True)
    assert (by_name["n"].type, by_name["n"].nullable) == ("int64", True)


def test_group_by_column_need_not_be_selected(agg_path):
    result = query_file(agg_path, "select count(*) from input group by g")
    assert result.column_names == ("COUNT(*)",)
    assert result.column("COUNT(*)") == [3, 1, 1, 1]  # a, b, NULL, c first-row order


def test_group_by_where_runs_first(agg_path):
    result = query_file(
        agg_path, "select g, count(*) from input where id >= 3 group by g"
    )
    # surviving rows: 3(NULL),4(a),5(c),6(a)
    assert _rows(result) == [(None, 1), ("a", 2), ("c", 1)]


def test_group_by_empty_filter_returns_zero_rows(agg_path):
    result = query_file(
        agg_path, "select g, count(*) from input where id > 1000 group by g"
    )
    assert result.row_count == 0
    assert result.column_names == ("g", "COUNT(*)")
    assert result.columns == {"g": [], "COUNT(*)": []}


def test_grouped_plain_column_must_be_grouped(agg_path):
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select g, n, count(*) from input group by g")
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select id, count(*) from input group by g")


def test_star_cannot_mix_with_grouping(agg_path):
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select * from input group by g")
    # SELECT *, aggregate is a syntax error (star cannot be followed by more items).
    with pytest.raises(QuerySyntaxError):
        query_file(agg_path, "select *, count(*) from input")


def test_duplicate_grouped_result_columns(agg_path):
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select g, g from input group by g")
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select g, g, count(*) from input group by g")


def test_group_by_aggregate_within_group_nulls_ignored(agg_path):
    # Group 'b' has n NULL only -> COUNT(n)=0 and other aggs NULL.
    result = query_file(
        agg_path,
        "select g, count(n), sum(n), avg(n), min(n), max(n) from input group by g order by g nulls first",
    )
    rows = dict((row[0], row[1:]) for row in _rows(result))
    assert rows["b"] == (0, None, None, None, None)
    assert rows["a"] == (3, 60, 20.0, 10, 40)
    assert rows[None] == (1, 30, 30.0, 30, 30)


def test_group_by_int_overflow_per_group(tmp_path):
    schema = Schema([
        ColumnSchema("g", "utf8"),
        ColumnSchema("n", "int64"),
    ])
    p = tmp_path / "g.caef"
    write_file(
        p,
        Table(
            schema,
            {"g": ["a", "a", "b", "b"], "n": [2**63 - 1, 1, 1, 2]},
        ),
    )
    with pytest.raises(QueryValidationError):
        query_file(p, "select g, sum(n) from input group by g")
    # Group 'b' alone aggregates fine.
    result = query_file(p, "select g, sum(n) from input where g = 'b' group by g")
    assert _rows(result) == [("b", 3)]


def test_group_by_min_max_types(tmp_path):
    schema = Schema([
        ColumnSchema("g", "utf8"),
        ColumnSchema("f", "float64", nullable=True),
        ColumnSchema("b", "bool"),
        ColumnSchema("s", "utf8", nullable=True),
    ])
    p = tmp_path / "m.caef"
    write_file(
        p,
        Table(
            schema,
            {
                "g": ["x", "x", "y"],
                "f": [1.5, 0.25, -2.0],
                "b": [True, False, True],
                "s": ["z", "a", "m"],
            },
        ),
    )
    result = query_file(
        p, "select g, min(f), max(f), min(b), max(b), min(s), max(s) from input group by g order by g"
    )
    assert _rows(result) == [
        ("x", 0.25, 1.5, False, True, "a", "z"),
        ("y", -2.0, -2.0, True, True, "m", "m"),
    ]


# ---------------------------------------------------------------------------
# Aggregation: ORDER BY / LIMIT over results
# ---------------------------------------------------------------------------


def test_aggregate_order_by_grouped_column(agg_path):
    result = query_file(
        agg_path, "select g, count(*) from input group by g order by g nulls first"
    )
    assert result.column("g") == [None, "a", "b", "c"]


def test_aggregate_order_by_aggregate(agg_path):
    result = query_file(
        agg_path, "select g, count(*) from input group by g order by count(*) desc, g"
    )
    assert _rows(result) == [("a", 3), ("b", 1), ("c", 1), (None, 1)]


def test_aggregate_order_by_nulls_first_last(agg_path):
    last = query_file(
        agg_path, "select g, count(*) from input group by g order by g"
    )
    assert last.column("g") == ["a", "b", "c", None]
    first = query_file(
        agg_path, "select g, count(*) from input group by g order by g desc nulls first"
    )
    assert first.column("g") == [None, "c", "b", "a"]


def test_aggregate_order_by_must_reference_selected_result(agg_path):
    # A grouped column not projected cannot be ordered by.
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select count(*) from input group by g order by g")
    # An aggregate not projected cannot be ordered by.
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select g, count(*) from input group by g order by sum(n)")
    # Scalar aggregate ordering by a source column is invalid.
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select count(*) from input order by id")
    # An aggregate in a plain (non-aggregate) query is not a selectable result.
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select id from input order by count(*)")


def test_aggregate_order_by_repeats_selected_aggregate(agg_path):
    # Same aggregate text resolves to the single projected column.
    # SUM(n): a=60, b=NULL, NULL-key group=30, c=NULL.
    result = query_file(
        agg_path, "select sum(n) from input group by g order by sum(n) desc nulls first"
    )
    assert result.column_names == ("SUM(n)",)
    assert result.column("SUM(n)") == [None, None, 60, 30]


def test_aggregate_order_by_unknown_column(agg_path):
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select g, count(*) from input group by g order by missing")
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select g, count(*) from input group by g order by avg(missing)")


def test_aggregate_order_by_duplicate(agg_path):
    with pytest.raises(QueryValidationError):
        query_file(
            agg_path, "select g, count(*) from input group by g order by g, g"
        )
    with pytest.raises(QueryValidationError):
        query_file(
            agg_path,
            "select g, count(*) from input group by g order by count(*), count(*)",
        )


def test_aggregate_limit_after_sorting(agg_path):
    result = query_file(
        agg_path,
        "select g, count(*) from input group by g order by g limit 2",
    )
    assert _rows(result) == [("a", 3), ("b", 1)]
    result = query_file(
        agg_path,
        "select g, count(*) from input group by g order by count(*) desc, g limit 1",
    )
    assert _rows(result) == [("a", 3)]


def test_scalar_aggregate_limit(agg_path):
    result = query_file(agg_path, "select count(*) from input limit 0")
    assert result.row_count == 0
    assert result.column_names == ("COUNT(*)",)
    result = query_file(agg_path, "select count(*) from input limit 5")
    assert result.column("COUNT(*)") == [6]


# ---------------------------------------------------------------------------
# Aggregation: determinism and CLI
# ---------------------------------------------------------------------------


def test_aggregate_result_deterministic_bytes(agg_path):
    import contextlib
    import io

    sql = "select g, count(*), sum(n), avg(f) from input group by g order by g nulls first"
    outs = set()
    for _ in range(3):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert main(["query", str(agg_path), sql]) == 0
        outs.add(buf.getvalue())
    assert len(outs) == 1


def test_cli_scalar_aggregate_json(tmp_path, capsys, agg_path):
    code = main(["query", str(agg_path), "select count(*), avg(n) from input where g = 'a'"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [
        {"name": "COUNT(*)", "type": "int64", "nullable": False},
        {"name": "AVG(n)", "type": "float64", "nullable": True},
    ]
    assert payload["rows"] == [[3, 20.0]]


def test_cli_grouped_empty_filter(tmp_path, capsys, agg_path):
    code = main(
        ["query", str(agg_path), "select g, count(*) from input where id > 9 group by g"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [
        {"name": "g", "type": "utf8", "nullable": True},
        {"name": "COUNT(*)", "type": "int64", "nullable": False},
    ]
    assert payload["rows"] == []


def test_cli_aggregate_validation_error_exit_code(tmp_path, capsys, agg_path):
    code = main(["query", str(agg_path), "select sum(g) from input"])
    assert code == 2
    assert capsys.readouterr().out == ""


def test_cli_aggregate_syntax_error_exit_code(tmp_path, capsys, agg_path):
    code = main(["query", str(agg_path), "select count(*) from input group by"])
    assert code == 2
    assert capsys.readouterr().out == ""


def test_star_with_group_by_is_validation_error(agg_path):
    with pytest.raises(QueryValidationError):
        query_file(agg_path, "select * from input where id = 1 group by id")


# ---------------------------------------------------------------------------
# Result shape, determinism and file semantics
# ---------------------------------------------------------------------------


def test_empty_result_keeps_columns(path):
    result = query_file(path, "select id, s, flag from input where id > 1000")
    assert result.column_names == ("id", "s", "flag")
    assert result.schema.columns == (
        ColumnSchema("id", "int64"),
        ColumnSchema("s", "utf8", nullable=True),
        ColumnSchema("flag", "bool"),
    )
    assert result.row_count == 0
    assert result.columns == {"id": [], "s": [], "flag": []}


def test_query_on_empty_file(tmp_path):
    p = tmp_path / "empty.caef"
    write_file(p, Table(SCHEMA, {name: [] for name in SCHEMA.names}))
    result = query_file(p, "select id from input where id = 1 or s is null")
    assert result.column_names == ("id",)
    assert result.row_count == 0


def test_query_does_not_mutate_file(path):
    before = path.read_bytes()
    query_file(path, "select id, n from input where f is null or not flag")
    query_file(path, "select * from input")
    assert path.read_bytes() == before


def test_corrupt_file_raises_format_error(path):
    blob = bytearray(path.read_bytes())
    blob[-9] ^= 0xFF
    path.write_bytes(blob)
    with pytest.raises(ColumnarFormatError):
        query_file(path, "select * from input")


def test_missing_file_raises_os_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        query_file(tmp_path / "nope.caef", "select * from input")


def test_syntax_checked_before_file_is_read(tmp_path, path):
    # SQL is parsed before the file is opened.
    missing = tmp_path / "nope.caef"
    with pytest.raises(QuerySyntaxError):
        query_file(missing, "this is not sql")
    # With valid SQL, a missing file surfaces as an OSError.
    with pytest.raises(FileNotFoundError):
        query_file(missing, "select * from input")
    # A malformed file with valid SQL surfaces the format error.
    bad = tmp_path / "bad.caef"
    bad.write_bytes(b"garbage")
    with pytest.raises(ColumnarFormatError):
        query_file(bad, "select * from input")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_query_json(tmp_path, capsys, path):
    code = main(["query", str(path), "select id,s from input where id <= 2"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert list(payload) == ["columns", "rows"]
    assert payload["columns"] == [
        {"name": "id", "type": "int64", "nullable": False},
        {"name": "s", "type": "utf8", "nullable": True},
    ]
    assert payload["rows"] == [[1, "a"], [2, "b"]]
    assert out.endswith("\n")
    # Single-line UTF-8 JSON.
    assert out.count("\n") == 1


def test_cli_query_unicode(tmp_path, capsys, path):
    assert main(["query", str(path), "select 名前 from input where id = 4"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["columns"][0]["name"] == "名前"
    assert payload["rows"] == [["は"]]
    assert "は" in out  # ensure_ascii=False


def test_cli_query_empty_result_columns_preserved(tmp_path, capsys, path):
    code = main(["query", str(path), "select flag from input where id > 99"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [{"name": "flag", "type": "bool", "nullable": False}]
    assert payload["rows"] == []


def test_cli_query_deterministic_bytes(tmp_path, path):
    outs = set()
    import io
    import contextlib

    sql = "select s, f from input where n is not null"
    for _ in range(3):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert main(["query", str(path), sql]) == 0
        outs.add(buf.getvalue())
    assert len(outs) == 1


def test_cli_query_syntax_error(tmp_path, capsys, path):
    code = main(["query", str(path), "select from input"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_query_validation_error(tmp_path, capsys, path):
    code = main(["query", str(path), "select missing from input"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_query_format_error(tmp_path, capsys):
    bad = tmp_path / "bad"
    bad.write_bytes(b"not a columnar file")
    code = main(["query", str(bad), "select * from input"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_query_os_error(tmp_path, capsys):
    code = main(["query", str(tmp_path / "missing"), "select * from input"])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_query_order_by_limit(tmp_path, capsys, path):
    code = main(
        [
            "query",
            str(path),
            "select id from input where id >= 2 order by n desc nulls first limit 3",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [{"name": "id", "type": "int64", "nullable": False}]
    # Rows 2,5 are the NULL-n rows (first, file order), then n=40 (id 4).
    assert payload["rows"] == [[2], [5], [4]]


def test_cli_query_limit_zero_preserves_columns(tmp_path, capsys, path):
    code = main(["query", str(path), "select id, s from input limit 0"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [
        {"name": "id", "type": "int64", "nullable": False},
        {"name": "s", "type": "utf8", "nullable": True},
    ]
    assert payload["rows"] == []


def test_cli_query_order_deterministic_bytes(tmp_path, path):
    import contextlib
    import io

    sql = "select s, f from input order by s, id desc"
    outs = set()
    for _ in range(3):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert main(["query", str(path), sql]) == 0
        outs.add(buf.getvalue())
    assert len(outs) == 1


def test_cli_query_order_syntax_error_exit_code(tmp_path, capsys, path):
    code = main(["query", str(path), "select id from input order by id nulls"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_query_order_validation_error_exit_code(tmp_path, capsys, path):
    code = main(["query", str(path), "select id from input order by missing limit 2"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_query_limit_syntax_error_exit_code(tmp_path, capsys, path):
    code = main(["query", str(path), "select id from input limit -1"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_query_subprocess(tmp_path, path):
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "columnar_analytics.cli",
            "query",
            str(path),
            "select id from input where s = 'c'",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["rows"] == [[5]]


def test_cli_existing_commands_unchanged(capsys):
    assert main(["version"]) == 0
