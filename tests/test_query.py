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
# Aggregates without GROUP BY
# ---------------------------------------------------------------------------


def test_count_star(path):
    result = query_file(path, "select count(*) from input")
    assert result.column_names == ("COUNT(*)",)
    assert result.schema.columns == (ColumnSchema("COUNT(*)", "int64", nullable=False),)
    assert result.column("COUNT(*)") == [5]


def test_count_star_after_where(path):
    result = query_file(path, "select count(*) from input where id >= 3")
    assert result.column("COUNT(*)") == [3]


def test_count_column_ignores_nulls(path):
    assert query_file(path, "select count(n) from input").column("COUNT(n)") == [3]
    assert query_file(path, "select count(f) from input").column("COUNT(f)") == [4]
    assert query_file(path, "select count(s) from input").column("COUNT(s)") == [4]


def test_count_function_name_case_insensitive(path):
    result = query_file(path, "select CoUnT(*) from input")
    assert result.column_names == ("COUNT(*)",)


def test_sum_keeps_input_type(path):
    result = query_file(path, "select sum(n) from input")
    assert result.schema.columns == (ColumnSchema("SUM(n)", "int64", nullable=True),)
    assert result.column("SUM(n)") == [80]
    result = query_file(path, "select sum(f) from input")
    assert result.schema.columns == (ColumnSchema("SUM(f)", "float64", nullable=True),)
    assert result.column("SUM(f)") == [13.5]


def test_avg_is_float64(path):
    result = query_file(path, "select avg(n) from input")
    assert result.schema.columns == (ColumnSchema("AVG(n)", "float64", nullable=True),)
    assert result.column("AVG(n)") == [80 / 3]
    assert query_file(path, "select avg(id) from input").column("AVG(id)") == [3.0]


def test_min_max_types(path):
    result = query_file(path, "select min(n), max(n), min(f), max(f), min(s), max(s) from input")
    assert result.schema.columns == (
        ColumnSchema("MIN(n)", "int64", nullable=True),
        ColumnSchema("MAX(n)", "int64", nullable=True),
        ColumnSchema("MIN(f)", "float64", nullable=True),
        ColumnSchema("MAX(f)", "float64", nullable=True),
        ColumnSchema("MIN(s)", "utf8", nullable=True),
        ColumnSchema("MAX(s)", "utf8", nullable=True),
    )
    assert result.row_count == 1
    assert result.column("MIN(n)") == [10]
    assert result.column("MAX(n)") == [40]
    assert result.column("MIN(f)") == [-0.5]
    assert result.column("MAX(f)") == [10.0]
    assert result.column("MIN(s)") == ["a"]
    assert result.column("MAX(s)") == ["c"]


def test_min_max_bool(path):
    result = query_file(path, "select min(flag), max(flag), min(flag_n), max(flag_n) from input")
    assert result.column("MIN(flag)") == [False]
    assert result.column("MAX(flag)") == [True]
    assert result.column("MIN(flag_n)") == [False]
    assert result.column("MAX(flag_n)") == [True]


def test_multiple_aggregates_select_order(path):
    result = query_file(path, "select max(id), min(id), count(*), sum(id), avg(id) from input")
    assert result.column_names == ("MAX(id)", "MIN(id)", "COUNT(*)", "SUM(id)", "AVG(id)")
    assert result.column("MAX(id)") == [5]
    assert result.column("MIN(id)") == [1]
    assert result.column("COUNT(*)") == [5]
    assert result.column("SUM(id)") == [15]
    assert result.column("AVG(id)") == [3.0]


def test_no_group_by_empty_filter_returns_one_row(path):
    result = query_file(path, "select count(*), count(n), sum(n), avg(n), min(n), max(n) from input where id > 1000")
    assert result.row_count == 1
    assert result.column("COUNT(*)") == [0]
    assert result.column("COUNT(n)") == [0]
    assert result.column("SUM(n)") == [None]
    assert result.column("AVG(n)") == [None]
    assert result.column("MIN(n)") == [None]
    assert result.column("MAX(n)") == [None]


def test_no_group_by_empty_file_returns_one_row(tmp_path):
    p = tmp_path / "empty.caef"
    write_file(p, Table(SCHEMA, {name: [] for name in SCHEMA.names}))
    result = query_file(p, "select count(*), count(n), sum(f), avg(f), min(s), max(s) from input")
    assert result.row_count == 1
    assert result.column("COUNT(*)") == [0]
    assert result.column("SUM(f)") == [None]


def test_count_columns_never_nullable_others_nullable(path):
    result = query_file(path, "select count(*), count(n), sum(n), avg(n), min(n), max(n) from input")
    cols = {c.name: c.nullable for c in result.schema.columns}
    assert cols == {
        "COUNT(*)": False,
        "COUNT(n)": False,
        "SUM(n)": True,
        "AVG(n)": True,
        "MIN(n)": True,
        "MAX(n)": True,
    }


def test_sum_overflow_int64(tmp_path):
    schema = Schema([ColumnSchema("x", "int64")])
    p = tmp_path / "big.caef"
    write_file(p, Table(schema, {"x": [2**63 - 1, 1]}))
    with pytest.raises(QueryValidationError):
        query_file(p, "select sum(x) from input")


def test_sum_int64_cancellation_can_stay_in_range(tmp_path):
    schema = Schema([ColumnSchema("x", "int64")])
    p = tmp_path / "cancel.caef"
    write_file(p, Table(schema, {"x": [2**63 - 1, -1, 2, -(2**63 - 1), -1]}))
    # Final total is 0 even though running prefixes leave the range.
    assert query_file(p, "select sum(x) from input").column("SUM(x)") == [0]


def test_float_sum_non_finite_raises(tmp_path):
    schema = Schema([ColumnSchema("d", "float64")])
    p = tmp_path / "huge.caef"
    big = 1.7976931348623157e308
    write_file(p, Table(schema, {"d": [big, big]}))
    with pytest.raises(QueryValidationError):
        query_file(p, "select sum(d) from input")
    with pytest.raises(QueryValidationError):
        query_file(p, "select avg(d) from input")


def test_sum_avg_reject_non_numeric(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select sum(s) from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select avg(flag) from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select sum(flag_n) from input")


def test_min_max_accept_all_types(path):
    assert query_file(path, "select min(id), max(id) from input").column("MAX(id)") == [5]
    assert query_file(path, "select min(s) from input").column("MIN(s)") == ["a"]
    assert query_file(path, "select max(flag) from input").column("MAX(flag)") == [True]


def test_plain_column_without_group_by_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id, count(*) from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(*), n from input")


def test_aggregate_in_where_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(*) from input where count(*) > 0")
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(*) from input where sum(id) > 1")


def test_nested_aggregate_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select sum(count(*)) from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(sum(id)) from input")


def test_aggregate_alias_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(*) c from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(*) as total from input")


@pytest.mark.parametrize(
    "sql",
    [
        "select count() from input",
        "select count(id, n) from input",
        "select sum() from input",
        "select sum(*) from input",
        "select avg(*) from input",
        "select min(id, n) from input",
        "select count(1) from input",
        "select count(id from input",
        "select count id) from input",
        "select foo(id) from input",
    ],
)
def test_aggregate_syntax_errors(path, sql):
    with pytest.raises(QuerySyntaxError):
        query_file(path, sql)


def test_unknown_aggregate_argument(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(missing) from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select max(missing) from input")


def test_duplicate_aggregate_result(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(*), count(*) from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select sum(id), sum(id) from input")


def test_column_named_like_aggregate_is_plain(tmp_path):
    schema = Schema([ColumnSchema("count", "int64"), ColumnSchema("sum", "int64", nullable=True)])
    p = tmp_path / "named.caef"
    write_file(p, Table(schema, {"count": [1, 2, 3], "sum": [4, None, 6]}))
    assert query_file(p, "select count from input").column("count") == [1, 2, 3]
    assert query_file(p, "select count(*), sum(count), avg(sum) from input").columns == {
        "COUNT(*)": [3],
        "SUM(count)": [6],
        "AVG(sum)": [5.0],
    }


# ---------------------------------------------------------------------------
# GROUP BY
# ---------------------------------------------------------------------------


@pytest.fixture()
def grouped_path(tmp_path):
    schema = Schema([
        ColumnSchema("g", "utf8", nullable=True),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("v", "int64", nullable=True),
        ColumnSchema("d", "float64", nullable=True),
    ])
    # First-occurrence order of g is: 'b', None, 'a'.
    table = Table(
        schema,
        {
            "g": ["b", None, "a", "b", "a", None, "b"],
            "k": [1, 1, 2, 2, 1, 2, 1],
            "v": [10, None, 30, 40, 50, 60, None],
            "d": [1.0, 2.0, None, 4.0, 5.0, None, 7.0],
        },
    )
    p = tmp_path / "grouped.caef"
    write_file(p, table)
    return p


def test_group_by_basic(grouped_path):
    result = query_file(grouped_path, "select g, count(*), sum(v) from input group by g")
    assert result.column_names == ("g", "COUNT(*)", "SUM(v)")
    # Groups follow the first selected row: b, NULL, a.
    assert result.column("g") == ["b", None, "a"]
    assert result.column("COUNT(*)") == [3, 2, 2]
    assert result.column("SUM(v)") == [50, 60, 80]


def test_group_by_group_column_preserves_schema(grouped_path):
    result = query_file(grouped_path, "select g, count(*) from input group by g")
    assert result.schema.columns[0] == ColumnSchema("g", "utf8", nullable=True)
    assert result.schema.columns[1] == ColumnSchema("COUNT(*)", "int64", nullable=False)


def test_group_by_null_is_one_group(grouped_path):
    result = query_file(grouped_path, "select g, count(*) from input group by g")
    gcol = result.column("g")
    assert gcol.count(None) == 1
    assert dict(zip(gcol, result.column("COUNT(*)")))[None] == 2


def test_group_by_multiple_keys(grouped_path):
    result = query_file(
        grouped_path,
        "select g, k, count(*), count(v), avg(d) from input group by g, k",
    )
    rows = list(zip(result.column("g"), result.column("k"), result.column("COUNT(*)"), result.column("COUNT(v)")))
    # First-row order: (b,1),(None,1),(a,2),(b,2),(a,1),(None,2)
    assert rows == [
        ("b", 1, 2, 1),
        (None, 1, 1, 0),
        ("a", 2, 1, 1),
        ("b", 2, 1, 1),
        ("a", 1, 1, 1),
        (None, 2, 1, 1),
    ]


def test_group_by_after_where(grouped_path):
    result = query_file(
        grouped_path,
        "select g, count(*) from input where v is not null group by g",
    )
    pairs = dict(zip(result.column("g"), result.column("COUNT(*)")))
    # b rows with v present: v=10 and v=40 (last b row has NULL v).
    assert pairs == {"b": 2, "a": 2, None: 1}
    # Filtered first-row order: b (row 1), a (row 3), NULL (row 6).
    assert result.column("g") == ["b", "a", None]


def test_group_by_empty_filter_returns_zero_rows(grouped_path):
    result = query_file(
        grouped_path,
        "select g, count(*), sum(v) from input where v > 1000 group by g",
    )
    assert result.row_count == 0
    assert result.column_names == ("g", "COUNT(*)", "SUM(v)")
    assert result.columns == {"g": [], "COUNT(*)": [], "SUM(v)": []}


def test_group_by_select_order(grouped_path):
    result = query_file(
        grouped_path, "select count(*), g, sum(v) from input group by g"
    )
    assert result.column_names == ("COUNT(*)", "g", "SUM(v)")
    assert result.column("g") == ["b", None, "a"]


def test_group_by_plain_column_must_be_grouped(grouped_path):
    with pytest.raises(QueryValidationError):
        query_file(grouped_path, "select g, v, count(*) from input group by g")
    with pytest.raises(QueryValidationError):
        query_file(grouped_path, "select k, count(*) from input group by g")


def test_group_by_without_aggregate_is_allowed(grouped_path):
    # Grouped columns only: effectively distinct groups in first-row order.
    result = query_file(grouped_path, "select g from input group by g")
    assert result.column("g") == ["b", None, "a"]
    assert result.row_count == 3
    result = query_file(grouped_path, "select g, k from input group by g, k")
    assert result.row_count == 6


def test_group_by_unknown_and_duplicate(grouped_path):
    with pytest.raises(QueryValidationError):
        query_file(grouped_path, "select g, count(*) from input group by missing")
    with pytest.raises(QueryValidationError):
        query_file(grouped_path, "select g, count(*) from input group by g, g")


def test_group_by_duplicate_group_column_in_projection(grouped_path):
    with pytest.raises(QueryValidationError):
        query_file(grouped_path, "select g, g, count(*) from input group by g")


def test_star_mixed_with_aggregation(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select * from input group by id")
    with pytest.raises(QueryValidationError):
        query_file(path, "select *, count(*) from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select count(*), * from input")
    with pytest.raises(QueryValidationError):
        query_file(path, "select *, count(*) from input group by id")


def test_group_by_syntax_errors(path):
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select count(*) from input group by")
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select count(*) from input group by id,")
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select count(*) from input group group by id")


def test_clause_order_group_between_where_and_order(path):
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select count(*) from input order by id group by id")
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select count(*) from input group by id where id > 1")
    with pytest.raises(QuerySyntaxError):
        query_file(path, "select count(*) from input limit 1 group by id")


# ---------------------------------------------------------------------------
# ORDER BY / LIMIT over aggregates and groups
# ---------------------------------------------------------------------------


def test_order_by_aggregate_desc(grouped_path):
    result = query_file(
        grouped_path, "select g, count(*) from input group by g order by count(*) desc"
    )
    assert list(zip(result.column("g"), result.column("COUNT(*)"))) == [
        ("b", 3),
        (None, 2),
        ("a", 2),
    ]


def test_order_by_aggregate_then_group_nulls(grouped_path):
    # Ties on COUNT(*) (a and NULL) keep first-row order; NULL g sorts last by
    # default, so the tie is broken to (a, None).
    result = query_file(
        grouped_path,
        "select g, count(*) from input group by g order by count(*), g",
    )
    assert list(zip(result.column("g"), result.column("COUNT(*)"))) == [
        ("a", 2),
        (None, 2),
        ("b", 3),
    ]
    result = query_file(
        grouped_path,
        "select g, count(*) from input group by g order by count(*), g nulls first",
    )
    assert list(zip(result.column("g"), result.column("COUNT(*)"))) == [
        (None, 2),
        ("a", 2),
        ("b", 3),
    ]


def test_order_by_selected_group_column(grouped_path):
    result = query_file(
        grouped_path, "select g, sum(v) from input group by g order by g desc nulls first"
    )
    assert result.column("g") == [None, "b", "a"]


def test_order_by_unselected_result_rejected(grouped_path):
    with pytest.raises(QueryValidationError):
        query_file(grouped_path, "select g, count(*) from input group by g order by sum(v)")
    with pytest.raises(QueryValidationError):
        query_file(grouped_path, "select g, count(*) from input group by g order by v")
    with pytest.raises(QueryValidationError):
        query_file(grouped_path, "select g, count(*) from input group by g order by count(v)")
    with pytest.raises(QueryValidationError):
        query_file(grouped_path, "select g, count(*) from input group by g order by k")
    with pytest.raises(QueryValidationError):
        query_file(grouped_path, "select g, count(*) from input group by g order by missing")


def test_order_by_selected_aggregate_ok(grouped_path):
    result = query_file(
        grouped_path,
        "select g, sum(v), count(*) from input group by g order by sum(v) desc nulls last",
    )
    # SUM(v): a=80, NULL group=60, b=50; NULL values sort last regardless.
    assert result.column("g") == ["a", None, "b"]


def test_order_by_duplicate_results(grouped_path):
    with pytest.raises(QueryValidationError):
        query_file(
            grouped_path,
            "select g, count(*) from input group by g order by count(*), count(*)",
        )
    with pytest.raises(QueryValidationError):
        query_file(
            grouped_path,
            "select g, count(*) from input group by g order by g, g",
        )


def test_limit_after_group_order(grouped_path):
    result = query_file(
        grouped_path,
        "select g, count(*) from input group by g order by count(*) desc limit 1",
    )
    assert result.row_count == 1
    assert result.column("g") == ["b"]
    assert result.column("COUNT(*)") == [3]


def test_limit_zero_groups(grouped_path):
    result = query_file(
        grouped_path, "select g, count(*) from input group by g limit 0"
    )
    assert result.row_count == 0
    assert result.column_names == ("g", "COUNT(*)")


def test_limit_without_order_keeps_first_row_group_order(grouped_path):
    result = query_file(
        grouped_path, "select g, count(*) from input group by g limit 2"
    )
    assert result.column("g") == ["b", None]


def test_plain_query_order_by_aggregate_without_grouping_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input order by count(*)")
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input order by sum(id)")


def test_aggregate_result_byte_stable(grouped_path):
    import contextlib
    import io
    import json

    sql = "select g, k, count(*), sum(v), avg(d) from input where d is not null group by g, k order by g nulls first, k desc"
    outs = set()
    for _ in range(3):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert main(["query", str(grouped_path), sql]) == 0
        outs.add(buf.getvalue())
    assert len(outs) == 1
    payload = json.loads(next(iter(outs)))
    assert [c["name"] for c in payload["columns"]] == ["g", "k", "COUNT(*)", "SUM(v)", "AVG(d)"]


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


def test_cli_query_aggregate(tmp_path, capsys, path):
    code = main(
        [
            "query",
            str(path),
            "select s, count(*), sum(n), avg(f) from input where s is not null group by s order by count(*) desc, s",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [
        {"name": "s", "type": "utf8", "nullable": True},
        {"name": "COUNT(*)", "type": "int64", "nullable": False},
        {"name": "SUM(n)", "type": "int64", "nullable": True},
        {"name": "AVG(f)", "type": "float64", "nullable": True},
    ]
    # s='a' has rows 1 and 4 (count 2), then 'b' and 'c' (count 1, sorted).
    assert payload["rows"][0][0] == "a"
    assert payload["rows"][0][1] == 2
    assert [row[0] for row in payload["rows"]] == ["a", "b", "c"]


def test_cli_query_aggregate_empty_one_row(tmp_path, capsys, path):
    code = main(["query", str(path), "select count(*), sum(n) from input where id > 99"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [
        {"name": "COUNT(*)", "type": "int64", "nullable": False},
        {"name": "SUM(n)", "type": "int64", "nullable": True},
    ]
    assert payload["rows"] == [[0, None]]


def test_cli_query_grouped_empty_zero_rows(tmp_path, capsys, path):
    code = main(
        ["query", str(path), "select s, count(*) from input where id > 99 group by s"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == []
    assert [c["name"] for c in payload["columns"]] == ["s", "COUNT(*)"]


def test_cli_query_aggregate_validation_error_exit_code(tmp_path, capsys, path):
    code = main(["query", str(path), "select n, count(*) from input"])
    assert code == 2
    assert capsys.readouterr().out == ""


def test_cli_query_aggregate_syntax_error_exit_code(tmp_path, capsys, path):
    code = main(["query", str(path), "select count() from input"])
    assert code == 2
    assert capsys.readouterr().out == ""


def test_cli_query_aggregate_subprocess(tmp_path, path):
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "columnar_analytics.cli",
            "query",
            str(path),
            "select flag, count(*) from input group by flag order by flag",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["columns"] == [
        {"name": "flag", "type": "bool", "nullable": False},
        {"name": "COUNT(*)", "type": "int64", "nullable": False},
    ]
    assert payload["rows"] == [[False, 2], [True, 3]]
