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
        "select * from input where id = 1 group by id",
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
