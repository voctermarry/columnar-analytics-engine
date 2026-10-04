"""Tests for the IN / NOT IN predicate (WHERE, CASE WHEN, HAVING, pushdown)."""

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
    query_file,
    query_files,
    write_file,
    write_partitioned_file,
)
from columnar_analytics.cli import main


SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("f", "float64", nullable=True),
        ColumnSchema("s", "utf8", nullable=True),
        ColumnSchema("flag", "bool", nullable=True),
    ]
)

DATA = {
    "id": [1, 2, 3, 4, 5, 6],
    "n": [10, None, 30, 40, None, 60],
    "f": [1.5, 2.5, None, -0.5, 10.0, 1.0],
    "s": ["a", "b", None, "a", "c", "b"],
    "flag": [True, False, True, None, False, True],
}


@pytest.fixture()
def table():
    return Table(SCHEMA, DATA)


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "t.caef"
    write_file(p, Table(SCHEMA, DATA))
    return p


@pytest.fixture()
def v2_path(tmp_path):
    p = tmp_path / "t2.caef"
    write_partitioned_file(p, Table(SCHEMA, DATA), 3)
    return p


# ---------------------------------------------------------------------------
# Result semantics / three-valued logic
# ---------------------------------------------------------------------------


def test_in_basic(path):
    assert query_file(path, "SELECT id FROM input WHERE n IN (10, 30, 99)").column(
        "id"
    ) == [1, 3]
    # Duplicate options change nothing.
    assert query_file(path, "SELECT id FROM input WHERE n IN (10, 10, 10)").column(
        "id"
    ) == [1]


def test_not_in_basic(path):
    assert query_file(path, "SELECT id FROM input WHERE n NOT IN (10, 30, 40, 60)").column(
        "id"
    ) == []
    assert query_file(path, "SELECT id FROM input WHERE id NOT IN (1, 2)").column(
        "id"
    ) == [3, 4, 5, 6]


def test_in_null_operand_is_unknown(path):
    # Rows whose n is NULL never match even when the list has no NULL.
    assert query_file(path, "SELECT id FROM input WHERE n IN (10, 30)").column(
        "id"
    ) == [1, 3]
    assert query_file(path, "SELECT id FROM input WHERE n NOT IN (10, 30)").column(
        "id"
    ) == [4, 6]


def test_in_with_null_option(path):
    # Match wins over the NULL option.
    assert query_file(path, "SELECT id FROM input WHERE n IN (10, NULL)").column(
        "id"
    ) == [1]
    # NOT IN: a matching option is FALSE; a non-matching non-NULL value is
    # UNKNOWN because the list has NULL; NULL rows are UNKNOWN too.
    assert query_file(path, "SELECT id FROM input WHERE n NOT IN (10, NULL)").column(
        "id"
    ) == []
    # No match but the list has NULL -> UNKNOWN for IN, so no row survives.
    assert query_file(path, "SELECT id FROM input WHERE n IN (99, NULL)").column(
        "id"
    ) == []


def test_all_null_list(path):
    assert query_file(path, "SELECT id FROM input WHERE id IN (NULL)").row_count == 0
    assert query_file(path, "SELECT id FROM input WHERE id NOT IN (NULL)").row_count == 0
    # Legal on every operand type, including bool and utf8.
    assert query_file(path, "SELECT id FROM input WHERE s IN (NULL, NULL)").row_count == 0
    assert query_file(path, "SELECT id FROM input WHERE flag IN (NULL)").row_count == 0


def test_in_numeric_mixing(path):
    # int64 operand vs float64 options and vice versa.
    assert query_file(path, "SELECT id FROM input WHERE n IN (10.0, 30.5)").column(
        "id"
    ) == [1]
    assert query_file(path, "SELECT id FROM input WHERE f IN (1, 10)").column(
        "id"
    ) == [5, 6]


def test_in_bool_and_utf8(path):
    assert query_file(path, "SELECT id FROM input WHERE s IN ('a', 'c')").column(
        "id"
    ) == [1, 4, 5]
    assert query_file(path, "SELECT id FROM input WHERE flag IN (FALSE)").column(
        "id"
    ) == [2, 5]
    assert query_file(path, "SELECT id FROM input WHERE flag NOT IN (TRUE)").column(
        "id"
    ) == [2, 5]


def test_in_signed_literals(path):
    assert query_file(path, "SELECT id FROM input WHERE f IN (-0.5, +1.5)").column(
        "id"
    ) == [1, 4]
    assert query_file(path, "SELECT id FROM input WHERE n IN (+30, -7)").column(
        "id"
    ) == [3]


def test_in_scalar_expression_operand(path):
    assert query_file(path, "SELECT id FROM input WHERE id + 1 IN (3, 5, 99)").column(
        "id"
    ) == [2, 4]
    assert query_file(path, "SELECT id FROM input WHERE id * 2 NOT IN (4, 8)").column(
        "id"
    ) == [1, 3, 5, 6]


def test_in_case_expression_operand(path):
    sql = (
        "SELECT id FROM input WHERE "
        "CASE WHEN id = 1 THEN 10 ELSE 0 END IN (10)"
    )
    assert query_file(path, sql).column("id") == [1]


def test_in_inside_case_when(path):
    result = query_file(
        path,
        "SELECT id, CASE WHEN n IN (10, 40) THEN 1 ELSE 0 END AS x FROM input",
    )
    assert result.column("x") == [1, 0, 0, 1, 0, 0]
    # UNKNOWN (NULL operand) and FALSE both fall through to later branches.
    result = query_file(
        path,
        "SELECT id, CASE WHEN n IN (10) THEN 'one' "
        "WHEN n IS NULL THEN 'null' ELSE 'other' END AS label FROM input",
    )
    assert result.column("label") == ["one", "null", "other", "other", "null", "other"]


def test_in_operand_errors_still_raise(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT id FROM input WHERE id / (id - id) IN (1)")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT id FROM input WHERE 1e308 * 1e308 IN (1.0)")


# ---------------------------------------------------------------------------
# HAVING
# ---------------------------------------------------------------------------


def test_in_having_group_column(path):
    result = query_file(
        path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING s IN ('a', 'b')"
    )
    assert result.column("s") == ["a", "b"]
    assert result.column("COUNT(*)") == [2, 2]


def test_not_in_having_group_column(path):
    result = query_file(
        path,
        "SELECT flag, COUNT(*) FROM input GROUP BY flag HAVING flag NOT IN (TRUE)",
    )
    # The NULL-key group is UNKNOWN and dropped; FALSE groups survive.
    assert result.column("flag") == [False]
    assert result.column("COUNT(*)") == [2]


def test_in_having_aggregate(path):
    result = query_file(
        path,
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) IN (1, 2)",
    )
    # Every group (including the NULL-s group) has a count of 1 or 2.
    assert result.column("s") == ["a", "b", None, "c"]
    assert result.column("COUNT(*)") == [2, 2, 1, 1]


def test_in_having_null_options(path):
    # Every group count is non-NULL; a NULL option with no exact match is
    # UNKNOWN, so no group survives.
    result = query_file(
        path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) IN (99, NULL)"
    )
    assert result.row_count == 0


def test_in_having_illegal_operand(path):
    with pytest.raises((QuerySyntaxError, QueryValidationError)):
        query_file(
            path,
            "SELECT s, COUNT(*) FROM input GROUP BY s HAVING s + 1 IN (3)",
        )
    with pytest.raises((QuerySyntaxError, QueryValidationError)):
        query_file(
            path,
            "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) + 1 IN (3)",
        )
    with pytest.raises((QuerySyntaxError, QueryValidationError)):
        query_file(
            path,
            "SELECT s, COUNT(*) FROM input GROUP BY s HAVING s IN (1, 2)",
        )


# ---------------------------------------------------------------------------
# Binding / type rules
# ---------------------------------------------------------------------------


def test_in_type_mismatch(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT id FROM input WHERE n IN (TRUE)")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT id FROM input WHERE s IN (1)")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT id FROM input WHERE flag IN (1, 2)")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT id FROM input WHERE s IN ('a', 1)")


def test_unknown_operand_column(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT id FROM input WHERE missing IN (1)")


# ---------------------------------------------------------------------------
# Syntax errors -- all raised before any file is opened
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM input WHERE n IN ()",
        "SELECT id FROM input WHERE n IN (1,)",
        "SELECT id FROM input WHERE n IN (1, 2,)",
        "SELECT id FROM input WHERE n IN (1 2)",
        "SELECT id FROM input WHERE n IN 1",
        "SELECT id FROM input WHERE n NOT IN",
        "SELECT id FROM input WHERE n NOT IN IN (1)",
        "SELECT id FROM input WHERE n IN (s)",
        "SELECT id FROM input WHERE n IN (COUNT(*))",
        "SELECT id FROM input WHERE n IN (1 + 1)",
        "SELECT id FROM input WHERE n IN (NULL, s)",
        "SELECT id FROM input WHERE n IN ('a'",
        "SELECT id FROM input HAVING COUNT(*) IN (1,)",
    ],
)
def test_in_syntax_errors(sql):
    with pytest.raises(QuerySyntaxError):
        query_file("/nonexistent/path.caef", sql)


def test_in_keyword_case_insensitive(path):
    assert query_file(path, "select id from input where n in (10)").column("id") == [1]
    assert query_file(path, "select id from input where n not in (10, 30, 40, 60)").column(
        "id"
    ) == []


def test_in_as_column_name_still_works(tmp_path):
    # "in" is only a predicate keyword in the IN position; a column so
    # named keeps working.
    schema = Schema([ColumnSchema("in", "int64")])
    p = tmp_path / "c.caef"
    write_file(p, Table(schema, {"in": [1, 2]}))
    assert query_file(p, "SELECT in FROM input").column("in") == [1, 2]
    assert query_file(p, "SELECT in FROM input WHERE in IN (2)").column("in") == [2]


# ---------------------------------------------------------------------------
# Explain plan shape
# ---------------------------------------------------------------------------


def test_explain_in_tree(path):
    plan = explain_file(path, "SELECT id FROM input WHERE n IN (10, 30, NULL)")
    cond = next(op for op in plan["operators"] if op["operator"] == "Filter")[
        "condition"
    ]
    assert cond == {
        "kind": "in",
        "negated": False,
        "operand": {"kind": "column", "name": "n"},
        "options": [
            {"kind": "literal", "type": "int64", "value": 10},
            {"kind": "literal", "type": "int64", "value": 30},
            {"kind": "literal", "type": "null", "value": None},
        ],
    }


def test_explain_not_in_tree(path):
    plan = explain_file(path, "SELECT id FROM input WHERE s NOT IN ('a')")
    cond = next(op for op in plan["operators"] if op["operator"] == "Filter")[
        "condition"
    ]
    assert cond["kind"] == "in"
    assert cond["negated"] is True
    assert cond["operand"] == {"kind": "column", "name": "s"}
    assert cond["options"] == [
        {"kind": "literal", "type": "utf8", "value": "a"}
    ]


def test_explain_having_in_tree(path):
    plan = explain_file(
        path,
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING s IN ('a')",
    )
    cond = next(op for op in plan["operators"] if op["operator"] == "Having")[
        "condition"
    ]
    assert cond["kind"] == "in"
    assert cond["negated"] is False
    assert cond["operand"] == {"kind": "column", "name": "s"}


# ---------------------------------------------------------------------------
# v2 statistics pushdown
# ---------------------------------------------------------------------------


def test_v2_in_equivalence(path, v2_path):
    queries = [
        "SELECT id FROM input WHERE n IN (10, 40)",
        "SELECT id FROM input WHERE n NOT IN (10, 40)",
        "SELECT id FROM input WHERE n IN (10, NULL)",
        "SELECT id FROM input WHERE n NOT IN (10, NULL)",
        "SELECT id FROM input WHERE n IN (NULL)",
        "SELECT id FROM input WHERE s IN ('a', 'z', NULL)",
        "SELECT id FROM input WHERE flag IN (TRUE)",
        "SELECT id FROM input WHERE f IN (1.0, 10.0)",
        "SELECT id FROM input WHERE n IN (30, 60) AND s IS NOT NULL",
        "SELECT id FROM input WHERE n IN (999)",
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING s IN ('a', 'b')",
    ]
    for sql in queries:
        v1 = query_file(path, sql)
        v2 = query_file(v2_path, sql)
        assert v1.schema == v2.schema
        assert v1.columns == v2.columns, sql


def test_in_pushdown_prunes_groups(v2_path):
    # Groups of 3 rows: n ranges [10, 30] then [40, 60].
    plan = explain_file(v2_path, "SELECT id FROM input WHERE n IN (60)")
    scan = plan["operators"][0]
    assert scan["row_groups_total"] == 2
    assert scan["row_groups_selected"] == 1
    assert scan["pushed_condition"]["kind"] == "in"
    assert scan["pushed_condition"]["negated"] is False
    assert query_file(v2_path, "SELECT id FROM input WHERE n IN (60)").column(
        "id"
    ) == [6]


def test_in_pushdown_range_overlap_keeps_group(v2_path):
    plan = explain_file(v2_path, "SELECT id FROM input WHERE n IN (10, 60)")
    assert plan["operators"][0]["row_groups_selected"] == 2


def test_in_pushdown_all_null_options(v2_path):
    plan = explain_file(v2_path, "SELECT id FROM input WHERE n IN (NULL)")
    assert plan["operators"][0]["row_groups_selected"] == 0
    assert query_file(v2_path, "SELECT COUNT(*) FROM input WHERE n IN (NULL)").column(
        "COUNT(*)"
    ) == [0]


def test_in_pushdown_option_null_does_not_save_group(v2_path):
    # NULL alone can never make IN TRUE; an out-of-range non-NULL option
    # with NULL still excludes every group.
    plan = explain_file(v2_path, "SELECT id FROM input WHERE n IN (999, NULL)")
    assert plan["operators"][0]["row_groups_selected"] == 0


def test_not_in_is_not_pushed(v2_path):
    plan = explain_file(v2_path, "SELECT id FROM input WHERE n NOT IN (60)")
    scan = plan["operators"][0]
    assert scan["row_groups_selected"] == 2
    assert scan["pushed_condition"] is None


def test_expression_in_is_not_pushed(v2_path):
    plan = explain_file(v2_path, "SELECT id FROM input WHERE n + 0 IN (60)")
    scan = plan["operators"][0]
    assert scan["row_groups_selected"] == 2
    assert scan["pushed_condition"] is None


def test_in_under_or_not_pushed(v2_path):
    plan = explain_file(
        v2_path, "SELECT id FROM input WHERE n IN (30) OR n IN (60)"
    )
    scan = plan["operators"][0]
    assert scan["row_groups_selected"] == 2
    assert scan["pushed_condition"] is None


def test_in_pushdown_combined_with_comparison(v2_path):
    plan = explain_file(
        v2_path, "SELECT id FROM input WHERE n IN (60) AND id > 5"
    )
    scan = plan["operators"][0]
    assert scan["row_groups_selected"] == 1
    pushed = scan["pushed_condition"]
    assert pushed["kind"] == "logic"
    kinds = [op["kind"] for op in pushed["operands"]]
    assert "in" in kinds and "comparison" in kinds


def test_join_chain_in_pushdown(tmp_path):
    right_schema = Schema(
        [ColumnSchema("rid", "int64"), ColumnSchema("k", "int64")]
    )
    left = tmp_path / "l.caef"
    right = tmp_path / "r.caef"
    write_partitioned_file(left, Table(SCHEMA, DATA), 3)
    write_partitioned_file(
        right, Table(right_schema, {"rid": [10, 20], "k": [1, 6]}), 2
    )
    sources = {"l": left, "r": right}
    plan = explain_files(
        sources,
        "SELECT l.id, r.rid FROM l INNER JOIN r ON l.id = r.k "
        "WHERE l.n IN (60)",
    )
    scans = {op["source"]: op for op in plan["operators"] if op["operator"] == "Scan"}
    assert scans["l"]["row_groups_selected"] == 1
    assert scans["l"]["pushed_condition"]["kind"] == "in"
    assert scans["l"]["pushed_condition"]["operand"]["name"] == "l.n"
    assert scans["r"]["pushed_condition"] is None
    result = query_files(
        sources,
        "SELECT l.id, r.rid FROM l INNER JOIN r ON l.id = r.k "
        "WHERE l.n IN (60)",
    )
    assert result.column("l.id") == [6]

    # An OUTER step disables pruning on every source.
    plan = explain_files(
        sources,
        "SELECT l.id FROM l LEFT JOIN r ON l.id = r.k WHERE l.n IN (60)",
    )
    for op in plan["operators"]:
        if op["operator"] == "Scan":
            assert "pushed_condition" not in op


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_in_query(path, capsys):
    code = main(["query", str(path), "SELECT id FROM input WHERE n IN (10, 30)"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row[0] for row in payload["rows"]] == [1, 3]


def test_cli_in_syntax_error_exit_code(path, capsys):
    code = main(["query", str(path), "SELECT id FROM input WHERE n IN ()"])
    assert code == 2
    assert capsys.readouterr().out == ""
