"""Tests for the IN / NOT IN discrete-set predicate (parse, bind, plan, pushdown)."""

from __future__ import annotations

import json
import struct

import pytest

from columnar_analytics import (
    ColumnSchema,
    ColumnarFormatError,
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
    "n": [10, None, 30, None, 50, 60],
    "f": [1.5, 2.5, None, -0.5, 10.0, 3.25],
    "s": ["a", "b", None, "a", "c", "b"],
    "flag": [True, False, None, True, False, True],
}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "t.caef"
    write_file(p, Table(SCHEMA, DATA))
    return p


@pytest.fixture()
def v2_path(tmp_path):
    p = tmp_path / "t2.caef"
    write_partitioned_file(p, Table(SCHEMA, DATA), 2, compression="zlib")
    return p


# ---------------------------------------------------------------------------
# Three-valued semantics in WHERE
# ---------------------------------------------------------------------------


def test_in_matches_any_option(path):
    result = query_file(path, "SELECT id FROM input WHERE n IN (50, 10) ORDER BY id")
    assert result.column("id") == [1, 5]
    # Duplicate options do not change the result.
    result = query_file(path, "SELECT id FROM input WHERE n IN (10, 10, 50, 50)")
    assert result.column("id") == [1, 5]


def test_in_nulls_drop_rows(path):
    # The NULL rows never equal any option; IN behaves like "no match".
    assert query_file(path, "SELECT id FROM input WHERE n IN (10, 30)").column(
        "id"
    ) == [1, 3]


def test_not_in_with_null_option_is_unknown_for_unmatched(path):
    # Rows matching the option are FALSE; the NULL rows and unmatched rows
    # become UNKNOWN when the list contains NULL -> all dropped.
    assert query_file(path, "SELECT id FROM input WHERE n NOT IN (10, 30, NULL)").row_count == 0
    # Without a NULL option, non-NULL unmatched rows survive; NULL rows do not.
    assert query_file(path, "SELECT id FROM input WHERE n NOT IN (10, 30)").column(
        "id"
    ) == [5, 6]


def test_all_null_list_is_always_unknown(path):
    assert query_file(path, "SELECT id FROM input WHERE n IN (NULL)").row_count == 0
    assert query_file(path, "SELECT id FROM input WHERE n NOT IN (NULL)").row_count == 0
    # Even a non-nullable column yields zero rows.
    assert query_file(path, "SELECT id FROM input WHERE id NOT IN (NULL)").row_count == 0


def test_null_operand_is_unknown(path, v2_path):
    for p in (path, v2_path):
        assert query_file(p, "SELECT id FROM input WHERE n IN (10, NULL)").column(
            "id"
        ) == [1]
        assert query_file(p, "SELECT id FROM input WHERE n NOT IN (10, 20)").column(
            "id"
        ) == [3, 5, 6]


def test_in_bool_utf8_and_numeric_families(path):
    assert query_file(path, "SELECT id FROM input WHERE flag IN (TRUE)").column(
        "id"
    ) == [1, 4, 6]
    # flag=TRUE rows do not equal FALSE, but the NULL option makes the
    # predicate UNKNOWN for them -> none survive.
    assert query_file(path, "SELECT id FROM input WHERE flag NOT IN (FALSE, NULL)").row_count == 0
    assert query_file(path, "SELECT id FROM input WHERE flag NOT IN (FALSE)").column(
        "id"
    ) == [1, 4, 6]
    assert query_file(path, "SELECT id FROM input WHERE s IN ('c', 'a')").column(
        "id"
    ) == [1, 4, 5]
    # int64 and float64 options may mix and compare numerically.
    assert query_file(path, "SELECT id FROM input WHERE id IN (1.0, 3, 4.5)").column(
        "id"
    ) == [1, 3]
    assert query_file(path, "SELECT id FROM input WHERE f IN (2.5, -0.5, 10)").column(
        "id"
    ) == [2, 4, 5]


def test_in_accepts_signed_literals(path):
    assert query_file(path, "SELECT id FROM input WHERE f IN (-0.5, +1.5)").column(
        "id"
    ) == [1, 4]
    assert query_file(path, "SELECT id FROM input WHERE id IN (-1, +1, 2)").column(
        "id"
    ) == [1, 2]


def test_in_within_boolean_combinators(path):
    result = query_file(
        path,
        "SELECT id FROM input WHERE (n IN (10, 50) OR s = 'b') AND id NOT IN (1) ORDER BY id",
    )
    assert result.column("id") == [2, 5, 6]
    # Prefix NOT keeps its existing precedence: "NOT n IN (...)" is parsed as
    # "(NOT n) IN (...)" and fails binding just like "NOT n = 10"; parenthesise
    # the predicate to negate it.
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT id FROM input WHERE NOT n IN (10, 30, 50, 60)")
    result = query_file(
        path, "SELECT id FROM input WHERE NOT (n IN (10, 30, 50, 60))"
    )
    assert result.row_count == 0  # only NULL rows remain, which are UNKNOWN


def test_in_case_when(path):
    result = query_file(
        path,
        "SELECT CASE WHEN id IN (1, 2, 3) THEN 1 ELSE 0 END AS x "
        "FROM input ORDER BY id",
    )
    assert result.column("x") == [1, 1, 1, 0, 0, 0]
    # A FALSE/UNKNOWN WHEN keeps walking the branches; NULL option produces
    # UNKNOWN, never TRUE.
    result = query_file(
        path,
        "SELECT CASE WHEN n IN (10, NULL) THEN 'hit' "
        "WHEN n IN (30) THEN 'mid' ELSE 'rest' END AS r "
        "FROM input ORDER BY id",
    )
    assert result.column("r") == ["hit", "rest", "mid", "rest", "rest", "rest"]


def test_in_expression_operand(path):
    assert query_file(path, "SELECT id FROM input WHERE id + 1 IN (2, 6)").column(
        "id"
    ) == [1, 5]
    assert query_file(
        path, "SELECT id FROM input WHERE 2 * id NOT IN (2, 4) ORDER BY id"
    ).column("id") == [3, 4, 5, 6]


def test_operand_errors_still_raise():
    schema = Schema([ColumnSchema("z", "int64"), ColumnSchema("f", "float64")])
    import tempfile, os

    d = tempfile.mkdtemp()
    q = os.path.join(d, "z.caef")
    write_file(q, Table(schema, {"z": [0, 1], "f": [1.0, 2.0]}))
    with pytest.raises(QueryValidationError):
        query_file(q, "SELECT z FROM input WHERE 1 / z IN (1)")
    with pytest.raises(QueryValidationError):
        query_file(q, "SELECT z FROM input WHERE 1e308 * f IN (1.0)")


# ---------------------------------------------------------------------------
# HAVING
# ---------------------------------------------------------------------------


def test_having_in_group_column_and_aggregate(path):
    result = query_file(
        path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING s IN ('a', 'b') ORDER BY s"
    )
    assert rows_of(result) == [["a", 2], ["b", 2]]
    result = query_file(
        path,
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) IN (1, 2) ORDER BY s",
    )
    # s counts: a=2, b=2, c=1 and the NULL-s group also has 1.
    assert result.column("s") == ["a", "b", "c", None]


def test_having_not_in_with_null(path):
    # The NULL s-group has COUNT 1; NOT IN (2) keeps the count-1 groups.
    result = query_file(
        path,
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) NOT IN (2) ORDER BY s",
    )
    assert result.column("COUNT(*)") == [1, 1]
    # Adding NULL makes the predicate UNKNOWN for count-1 groups too.
    assert (
        query_file(
            path,
            "SELECT s FROM input GROUP BY s HAVING COUNT(*) NOT IN (2, NULL)",
        ).row_count
        == 0
    )


def test_having_in_rejects_expression_operand(path):
    # HAVING keeps rejecting scalar arithmetic on a leaf operand (an
    # existing validation classification shared by every HAVING predicate).
    with pytest.raises(QueryValidationError):
        query_file(
            path, "SELECT s FROM input GROUP BY s HAVING COUNT(*) + 1 IN (2)"
        )


def test_having_in_rejects_ungrouped_column(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT s, COUNT(*) FROM input GROUP BY s HAVING id IN (1)")


# ---------------------------------------------------------------------------
# Parser: syntax errors are raised before any file is opened
# ---------------------------------------------------------------------------


SYNTAX_BAD = [
    "SELECT id FROM input WHERE id IN ()",
    "SELECT id FROM input WHERE id IN (1,)",
    "SELECT id FROM input WHERE id IN (1, 2",
    "SELECT id FROM input WHERE id IN 1)",
    "SELECT id FROM input WHERE id IN",
    "SELECT id FROM input WHERE id IN (id)",
    "SELECT id FROM input WHERE id IN (1 + 1)",
    "SELECT id FROM input WHERE id IN (1, id)",
    "SELECT id FROM input WHERE id IN (COUNT(*))",
    "SELECT id FROM input WHERE id NOT IN",
    "SELECT id FROM input WHERE NOT IN (1)",
    "SELECT id FROM input WHERE (IN (1))",
    "SELECT g, COUNT(*) FROM input GROUP BY g HAVING g IN ()",
]


@pytest.mark.parametrize("sql", SYNTAX_BAD)
def test_in_syntax_errors(path, sql):
    with pytest.raises(QuerySyntaxError):
        query_file(path, sql)


def test_in_syntax_error_before_file_access(tmp_path):
    missing = tmp_path / "missing.caef"
    with pytest.raises(QuerySyntaxError):
        query_file(missing, "SELECT id FROM input WHERE id IN (1,)")
    with pytest.raises(QuerySyntaxError):
        explain_file(missing, "SELECT id FROM input WHERE id NOT IN")


def test_in_not_predicate_only_inside_lists(path):
    # A bare NULL comparison remains a syntax error; NULL is list-only.
    with pytest.raises(QuerySyntaxError):
        query_file(path, "SELECT id FROM input WHERE n = NULL")


# ---------------------------------------------------------------------------
# Binding: type compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM input WHERE flag IN (1, 2)",
        "SELECT id FROM input WHERE flag IN ('yes')",
        "SELECT id FROM input WHERE s IN (1)",
        "SELECT id FROM input WHERE s IN (TRUE)",
        "SELECT id FROM input WHERE id IN ('1')",
        "SELECT id FROM input WHERE id IN (TRUE)",
        "SELECT id FROM input WHERE n IN (1, TRUE)",
        "SELECT id FROM input WHERE id IN ('a', 1)",
        "SELECT g, COUNT(*) FROM input GROUP BY g HAVING COUNT(*) IN ('x')",
    ],
)
def test_in_type_errors(path, sql):
    with pytest.raises(QueryValidationError):
        query_file(path, sql)


def test_in_unknown_column(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT id FROM input WHERE nope IN (1)")


# ---------------------------------------------------------------------------
# Explain plan
# ---------------------------------------------------------------------------


def test_explain_in_tree(path):
    plan = explain_file(path, "SELECT id FROM input WHERE id NOT IN (3, 1, NULL)")
    cond = [op for op in plan["operators"] if op["operator"] == "Filter"][0][
        "condition"
    ]
    assert cond == {
        "kind": "in",
        "negated": True,
        "operand": {"kind": "column", "name": "id"},
        "options": [
            {"kind": "literal", "type": "int64", "value": 3},
            {"kind": "literal", "type": "int64", "value": 1},
            {"kind": "literal", "type": None, "value": None},
        ],
    }


def test_explain_in_nested_in_case_and_logic(path):
    plan = explain_file(
        path,
        "SELECT CASE WHEN id IN (1) OR s IN ('a') THEN 1 ELSE 0 END AS x FROM input",
    )
    project = [op for op in plan["operators"] if op["operator"] == "Project"][0]
    expr = project["expressions"][0]["expression"]
    assert expr["kind"] == "case"
    when = expr["cases"][0]["when"]
    assert when["kind"] == "logic"
    assert when["operands"][0]["kind"] == "in"
    assert when["operands"][0]["negated"] is False
    assert when["operands"][0]["options"] == [
        {"kind": "literal", "type": "int64", "value": 1}
    ]


def test_explain_having_in_tree(path):
    plan = explain_file(
        path,
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING s IN ('a') AND COUNT(*) NOT IN (0, NULL)",
    )
    having = [op for op in plan["operators"] if op["operator"] == "Having"][0][
        "condition"
    ]
    assert having["kind"] == "logic"
    left, right = having["operands"]
    assert left == {
        "kind": "in",
        "negated": False,
        "operand": {"kind": "column", "name": "s"},
        "options": [{"kind": "literal", "type": "utf8", "value": "a"}],
    }
    assert right["kind"] == "in"
    assert right["negated"] is True
    assert [o["value"] for o in right["options"]] == [0, None]


def test_explain_repeatable_bytes(path):
    import json as _json

    first = _json.dumps(
        explain_file(path, "SELECT id FROM input WHERE n IN (10, 30, NULL)"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )
    second = _json.dumps(
        explain_file(path, "SELECT id FROM input WHERE n IN (10, 30, NULL)"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=False,
    )
    assert first == second


# ---------------------------------------------------------------------------
# v2 row-group statistics pushdown
# ---------------------------------------------------------------------------


def test_v1_v2_in_equivalence(path, v2_path):
    queries = [
        "SELECT id FROM input WHERE n IN (10, 50)",
        "SELECT id FROM input WHERE n NOT IN (10, 30)",
        "SELECT id FROM input WHERE n IN (10, NULL)",
        "SELECT id FROM input WHERE n NOT IN (1, 2, 3)",
        "SELECT id FROM input WHERE n IN (NULL)",
        "SELECT id FROM input WHERE s IN ('a', 'c')",
        "SELECT id FROM input WHERE flag IN (TRUE, FALSE, NULL)",
        "SELECT id FROM input WHERE id + 1 IN (2, 4)",
        "SELECT id FROM input WHERE n IN (10, 999) AND s IS NOT NULL",
        "SELECT s, COUNT(*) FROM input GROUP BY s HAVING COUNT(*) IN (2)",
        "SELECT DISTINCT s FROM input WHERE s IN ('a', 'b')",
    ]
    for sql in queries:
        a = query_file(path, sql)
        b = query_file(v2_path, sql)
        assert a.schema == b.schema, sql
        assert a.columns == b.columns, sql


def test_in_pushdown_selects_groups(v2_path):
    plan = explain_file(v2_path, "SELECT id FROM input WHERE n IN (10, 99)")
    scan = plan["operators"][0]
    # Groups of two rows: group 0 has n [10, NULL] (candidate 10),
    # group 1 has [30, NULL] (no candidate), group 2 has [50, 60] (none).
    assert scan["row_groups_total"] == 3
    assert scan["row_groups_selected"] == 1
    pushed = scan["pushed_condition"]
    assert pushed["kind"] == "in"
    assert pushed["negated"] is False
    assert pushed["operand"] == {"kind": "column", "name": "n"}
    assert [o["value"] for o in pushed["options"]] == [10, 99]
    # The selected rows still match the full predicate.
    assert query_file(v2_path, "SELECT id FROM input WHERE n IN (10, 99)").column(
        "id"
    ) == [1]


def test_in_pushdown_all_null_options_skips_every_group(v2_path):
    plan = explain_file(v2_path, "SELECT id FROM input WHERE n IN (NULL)")
    scan = plan["operators"][0]
    assert scan["row_groups_selected"] == 0
    assert scan["pushed_condition"]["options"] == [
        {"kind": "literal", "type": None, "value": None}
    ]


def test_in_pushdown_all_null_column_group(v2_path, tmp_path):
    # A group whose only rows have NULL in the target column is skipped even
    # though the overall column min/max span the candidate.
    schema = Schema([ColumnSchema("id", "int64"), ColumnSchema("n", "int64", nullable=True)])
    p = tmp_path / "allnull.caef"
    write_partitioned_file(
        p,
        Table(schema, {"id": [1, 2, 3, 4], "n": [5, 5, None, None]}),
        2,
    )
    plan = explain_file(p, "SELECT id FROM input WHERE n IN (5)")
    assert plan["operators"][0]["row_groups_selected"] == 1
    assert query_file(p, "SELECT id FROM input WHERE n IN (5)").column("id") == [1, 2]


def test_not_in_and_expression_operand_not_pushed(v2_path):
    plan = explain_file(v2_path, "SELECT id FROM input WHERE n NOT IN (10, 30)")
    scan = plan["operators"][0]
    assert scan["row_groups_selected"] == scan["row_groups_total"]
    assert scan["pushed_condition"] is None

    plan = explain_file(v2_path, "SELECT id FROM input WHERE id + 1 IN (2, 4)")
    assert plan["operators"][0]["pushed_condition"] is None

    # IN under OR is not pushed either.
    plan = explain_file(v2_path, "SELECT id FROM input WHERE n IN (10) OR id = 1")
    assert plan["operators"][0]["pushed_condition"] is None


def test_pushed_in_group_block_not_decoded(tmp_path):
    schema = Schema([ColumnSchema("id", "int64"), ColumnSchema("n", "int64", nullable=True)])
    p = tmp_path / "corrupt.caef"
    write_partitioned_file(
        p, Table(schema, {"id": [1, 2, 3, 4], "n": [10, 20, 30, 40]}), 2
    )
    _corrupt_block(p, 0, "n")
    # The IN list excludes the corrupt first group (values 10, 20).
    result = query_file(p, "SELECT id FROM input WHERE n IN (30, 40)")
    assert result.column("id") == [3, 4]
    # Without pruning the corruption surfaces.
    with pytest.raises(ColumnarFormatError):
        query_file(p, "SELECT id FROM input WHERE n IN (10, 30)")


def _corrupt_block(path, group_index, column_name):
    blob = bytearray(path.read_bytes())
    header_len = struct.unpack("<I", blob[5:9])[0]
    header = json.loads(bytes(blob[9 : 9 + header_len]).decode())
    col_index = [c["name"] for c in header["columns"]].index(column_name)
    block = header["row_groups"][group_index]["columns"][col_index]
    blob[9 + header_len + block["offset"]] ^= 0xFF
    path.write_bytes(blob)


def test_in_pushdown_inner_join_chain(tmp_path, v2_path):
    right_schema = Schema(
        [ColumnSchema("rid", "int64"), ColumnSchema("k", "int64", nullable=True)]
    )
    rv2 = tmp_path / "r.caef"
    write_partitioned_file(
        rv2, Table(right_schema, {"rid": [10, 20, 30], "k": [1, 50, 60]}), 1
    )
    sources = {"l": v2_path, "r": rv2}
    sql = "SELECT l.id, r.rid FROM l INNER JOIN r ON l.id = r.k WHERE r.k IN (1, 2)"
    plan = explain_files(sources, sql)
    scans = {op["source"]: op for op in plan["operators"] if op["operator"] == "Scan"}
    # r's groups hold k=1, k=50, k=60: only the first group is selected.
    assert scans["r"]["row_groups_selected"] == 1
    pushed = scans["r"]["pushed_condition"]
    assert pushed["kind"] == "in"
    assert pushed["operand"] == {"kind": "column", "name": "r.k"}
    # l has no pushed IN leaf -> selected == total.
    assert scans["l"]["row_groups_selected"] == scans["l"]["row_groups_total"]
    result = query_files(sources, sql)
    assert result.column("r.rid") == [10]


def test_in_pushdown_disabled_for_outer_chain(tmp_path, v2_path):
    right_schema = Schema([ColumnSchema("rid", "int64"), ColumnSchema("k", "int64")])
    rv2 = tmp_path / "r.caef"
    write_partitioned_file(rv2, Table(right_schema, {"rid": [10], "k": [1]}), 1)
    plan = explain_files(
        {"l": v2_path, "r": rv2},
        "SELECT l.id FROM l LEFT JOIN r ON l.id = r.k WHERE l.n IN (10, 50)",
    )
    scan = plan["operators"][0]
    assert "row_groups_total" not in scan
    assert "pushed_condition" not in scan


# ---------------------------------------------------------------------------
# Export, CLI and repeatability
# ---------------------------------------------------------------------------


def test_export_v1_v2_bytes_identical(path, v2_path, tmp_path):
    e1 = tmp_path / "a.csv"
    e2 = tmp_path / "b.csv"
    sql = "SELECT id FROM input WHERE n IN (10, 30, NULL) ORDER BY id"
    n1 = export_query_file(path, sql, e1, "csv")
    n2 = export_query_file(v2_path, sql, e2, "csv")
    assert n1 == n2 == 2
    assert e1.read_bytes() == e2.read_bytes() == b"id\n1\n3\n"

    j1 = tmp_path / "a.jsonl"
    j2 = tmp_path / "b.jsonl"
    sql = "SELECT id, s FROM input WHERE s IN ('a', 'c') ORDER BY id"
    assert export_query_file(path, sql, j1, "jsonl") == export_query_file(
        v2_path, sql, j2, "jsonl"
    )
    assert j1.read_bytes() == j2.read_bytes()


def test_cli_in_query_explain(tmp_path, path, capsys):
    rc = main(["query", str(path), "SELECT id FROM input WHERE n IN (10, 50)"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == [[1], [5]]

    rc = main(["explain", str(path), "SELECT id FROM input WHERE s NOT IN ('a')"])
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    cond = [op for op in plan["operators"] if op["operator"] == "Filter"][0][
        "condition"
    ]
    assert cond["kind"] == "in"
    assert cond["negated"] is True

    rc = main(["query", str(path), "SELECT id FROM input WHERE id IN ()"])
    assert rc == 2


def rows_of(table):
    return [
        [table._columns[c][r] for c in range(len(table.schema.columns))]
        for r in range(table.row_count)
    ]
