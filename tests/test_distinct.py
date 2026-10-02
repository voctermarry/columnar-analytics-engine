"""Tests for SELECT DISTINCT row deduplication across query / explain / export.

Covers single-file and two-file (joined) statements, the CLI entries and the
explain plan shape.  Non-DISTINCT behaviour is covered by the other suites;
the handful of checks here additionally pins the absence of the new operator.
"""

from __future__ import annotations

import json

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
    export_query_files,
    query_file,
    query_files,
    read_file,
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
    "id": [1, 2, 3, 4, 5, 6, 7],
    "n": [10, None, 10, 10, None, 30, 10],
    "f": [1.5, 2.5, 0.0, -0.0, None, 1.5, 2.5],
    "s": ["a", "b", None, "a", "c", "a", None],
    "flag": [True, False, True, True, False, False, True],
}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "t.caef"
    write_file(p, Table(SCHEMA, DATA), compression="zlib", dictionary_encoding=["s"])
    return p


LEFT_SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("name", "utf8", nullable=True),
    ]
)
LEFT_DATA = {
    "id": [1, 2, 3, 4],
    "k": [10, 20, 10, None],
    "name": ["a", "b", "c", "d"],
}
RIGHT_SCHEMA = Schema(
    [
        ColumnSchema("rid", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("tag", "utf8", nullable=True),
    ]
)
RIGHT_DATA = {"rid": [100, 200, 300], "k": [10, 10, 30], "tag": ["x", "y", "z"]}


@pytest.fixture()
def sources(tmp_path):
    left = tmp_path / "left.caef"
    right = tmp_path / "right.caef"
    write_file(left, Table(LEFT_SCHEMA, LEFT_DATA))
    write_file(right, Table(RIGHT_SCHEMA, RIGHT_DATA))
    return {"l": left, "r": right}


def rows(result):
    return [
        [result._columns[c][i] for c in range(len(result.schema.columns))]
        for i in range(result.row_count)
    ]


def operator_kinds(plan):
    return [op["operator"] for op in plan["operators"]]


def find_operator(plan, name):
    for op in plan["operators"]:
        if op["operator"] == name:
            return op
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Basic deduplication semantics
# ---------------------------------------------------------------------------


def test_distinct_single_column_first_occurrence_order(path):
    result = query_file(path, "SELECT DISTINCT s FROM input")
    assert result.column_names == ("s",)
    # a (id 1), b (id 2), NULL (id 3), c (id 5); duplicates at id 4/6/7 dropped.
    assert result.column("s") == ["a", "b", None, "c"]


def test_distinct_multiple_columns_compare_full_rows(path):
    result = query_file(path, "SELECT DISTINCT s, flag FROM input")
    assert rows(result) == [
        ["a", True],
        ["b", False],
        [None, True],
        ["c", False],
        ["a", False],
    ]
    # (a, True) repeats at id 4 and (NULL, True) at id 7; both collapse.


def test_distinct_star(path):
    result = query_file(path, "SELECT DISTINCT * FROM input")
    assert result.schema == SCHEMA
    # Every full row is unique, so nothing is removed and order is file order.
    assert result.row_count == 7
    assert result.column("id") == DATA["id"]


def test_distinct_star_collapses_identical_rows(tmp_path):
    schema = Schema([ColumnSchema("g", "int64"), ColumnSchema("t", "utf8")])
    p = tmp_path / "dup.caef"
    write_file(p, Table(schema, {"g": [1, 1, 2, 1], "t": ["a", "a", "a", "b"]}))
    result = query_file(p, "SELECT DISTINCT * FROM input")
    assert result.schema == schema
    assert rows(result) == [[1, "a"], [2, "a"], [1, "b"]]


def test_distinct_nulls_compare_equal_but_not_to_values(path):
    result = query_file(path, "SELECT DISTINCT n FROM input")
    assert result.column("n") == [10, None, 30]


def test_distinct_float_signed_zero_collapses_keeping_first_value(path):
    result = query_file(path, "SELECT DISTINCT f FROM input")
    # 0.0 (id 3) precedes -0.0 (id 4): the surviving value is the first seen.
    assert result.column("f") == [1.5, 2.5, 0.0, None]
    surviving = result.column("f")[2]
    assert surviving == 0.0
    # First-occurrence retention: it is literally positive zero.
    assert str(surviving) == "0.0"


def test_distinct_float_signed_zero_keeps_negative_when_seen_first(tmp_path):
    schema = Schema([ColumnSchema("f", "float64")])
    p = tmp_path / "z.caef"
    write_file(p, Table(schema, {"f": [-0.0, 0.0]}))
    result = query_file(p, "SELECT DISTINCT f FROM input")
    assert result.row_count == 1
    assert str(result.column("f")[0]) == "-0.0"


def test_distinct_bool_and_utf8_semantics(path):
    result = query_file(path, "SELECT DISTINCT flag FROM input")
    assert result.column("flag") == [True, False]
    result = query_file(path, "SELECT DISTINCT s FROM input WHERE id = 1 OR id = 4")
    assert result.column("s") == ["a"]


def test_distinct_numeric_columns_do_not_collide_across_types(tmp_path):
    schema = Schema(
        [ColumnSchema("a", "int64"), ColumnSchema("b", "float64")]
    )
    p = tmp_path / "m.caef"
    # (1, 1.0) and (1, 2.0) share the int part but differ in the float column.
    write_file(p, Table(schema, {"a": [1, 1, 2], "b": [1.0, 2.0, 1.0]}))
    result = query_file(p, "SELECT DISTINCT a, b FROM input")
    assert rows(result) == [[1, 1.0], [1, 2.0], [2, 1.0]]


def test_distinct_scalar_expression_with_alias(path):
    result = query_file(path, "SELECT DISTINCT n + 1 AS x FROM input ORDER BY x NULLS LAST")
    assert result.column_names == ("x",)
    assert result.column("x") == [11, 31, None]
    assert [c.type for c in result.schema.columns] == ["int64"]


def test_distinct_float_expression_and_case(path):
    result = query_file(
        path,
        "SELECT DISTINCT f * 2 AS d FROM input WHERE f IS NOT NULL ORDER BY d",
    )
    assert result.column("d") == [0.0, 3.0, 5.0]
    result = query_file(
        path,
        "SELECT DISTINCT CASE WHEN flag THEN s ELSE 'z' END AS t FROM input ORDER BY t NULLS LAST"
    )
    # True rows yield a (ids 1/4) or NULL (ids 3/7); False rows yield 'z'.
    assert result.column("t") == ["a", "z", None]


def test_distinct_applies_after_where(path):
    result = query_file(path, "SELECT DISTINCT n FROM input WHERE id >= 4")
    # id 4 -> 10, id 5 -> NULL, id 6 -> 30, id 7 -> 10
    assert result.column("n") == [10, None, 30]


def test_distinct_keyword_is_case_insensitive(path):
    a = query_file(path, "select distinct s from input")
    b = query_file(path, "SeLeCt DiStInCt s FrOm input")
    assert a.column("s") == b.column("s") == ["a", "b", None, "c"]


# ---------------------------------------------------------------------------
# ORDER BY / LIMIT interaction
# ---------------------------------------------------------------------------


def test_distinct_order_by_projected_column(path):
    result = query_file(path, "SELECT DISTINCT n FROM input ORDER BY n NULLS FIRST")
    assert result.column("n") == [None, 10, 30]
    result = query_file(path, "SELECT DISTINCT n FROM input ORDER BY n DESC NULLS FIRST")
    assert result.column("n") == [None, 30, 10]
    result = query_file(path, "SELECT DISTINCT f FROM input ORDER BY f DESC")
    # NULL stays last regardless of DESC without an explicit NULLS clause.
    assert result.column("f") == [2.5, 1.5, 0.0, None]


def test_distinct_order_by_alias(path):
    result = query_file(
        path, "SELECT DISTINCT n + 10 AS big FROM input ORDER BY big DESC"
    )
    assert result.column("big") == [40, 20, None]


def test_distinct_order_by_sorts_after_deduplication(path):
    # id 2 and id 5 both carry (NULL, False/..); ordering the distinct (s) set
    # must show the four distinct values sorted, not the pre-dedup stream.
    result = query_file(path, "SELECT DISTINCT s FROM input ORDER BY s NULLS LAST")
    assert result.column("s") == ["a", "b", "c", None]


def test_distinct_limit(path):
    result = query_file(path, "SELECT DISTINCT s FROM input LIMIT 2")
    assert result.column("s") == ["a", "b"]


def test_distinct_order_by_then_limit(path):
    result = query_file(
        path, "SELECT DISTINCT s FROM input ORDER BY s LIMIT 2"
    )
    assert result.column("s") == ["a", "b"]
    result = query_file(
        path, "SELECT DISTINCT n FROM input ORDER BY n DESC NULLS LAST LIMIT 1"
    )
    assert result.column("n") == [30]


def test_distinct_limit_zero_and_empty_result_keep_columns(path):
    result = query_file(path, "SELECT DISTINCT s FROM input LIMIT 0")
    assert result.row_count == 0
    assert result.column_names == ("s",)
    assert [c.type for c in result.schema.columns] == ["utf8"]
    result = query_file(path, "SELECT DISTINCT n, s FROM input WHERE id > 100")
    assert result.row_count == 0
    assert result.column_names == ("n", "s")


def test_distinct_order_by_null_placement_matches_plain_semantics(path):
    asc = query_file(path, "SELECT DISTINCT n FROM input ORDER BY n")
    assert asc.column("n") == [10, 30, None]
    first = query_file(path, "SELECT DISTINCT n FROM input ORDER BY n ASC NULLS FIRST")
    assert first.column("n") == [None, 10, 30]


# ---------------------------------------------------------------------------
# Expression evaluation timing: only WHERE-passing rows, errors still raised
# ---------------------------------------------------------------------------


def test_distinct_expression_division_by_zero_raises(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT 1 / (id - 1) AS x FROM input")


def test_distinct_expression_error_filtered_out_does_not_raise(path):
    # id = 1 is the only dividing-by-zero row; excluding it is safe.
    result = query_file(
        path,
        "SELECT DISTINCT 1 / (id - 1) AS x FROM input WHERE id > 1 ORDER BY x",
    )
    assert result.row_count == 6


def test_distinct_expression_error_survives_limit(path):
    # LIMIT must be applied after projection: a tiny LIMIT cannot suppress a
    # division-by-zero on an earlier surviving row.
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT 1 / (id - 1) AS x FROM input LIMIT 1")


def test_distinct_expression_int64_overflow_raises(path):
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "SELECT DISTINCT id * 9223372036854775807 AS x FROM input WHERE id = 2",
        )
    # id = 1 is within range.
    result = query_file(
        path,
        "SELECT DISTINCT id * 9223372036854775807 AS x FROM input WHERE id = 1",
    )
    assert result.column("x") == [2**63 - 1]


def test_distinct_non_finite_float_raises(tmp_path):
    schema = Schema([ColumnSchema("a", "float64"), ColumnSchema("b", "float64")])
    p = tmp_path / "big.caef"
    write_file(p, Table(schema, {"a": [1e308], "b": [1e308]}))
    with pytest.raises(QueryValidationError):
        query_file(p, "SELECT DISTINCT a * b AS x FROM input")


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def test_distinct_rejects_aggregate_group_by_having(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT COUNT(*) FROM input")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s, COUNT(*) FROM input GROUP BY s")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s FROM input GROUP BY s")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s FROM input HAVING COUNT(*) > 0")


def test_distinct_order_by_requires_projected_name(path):
    with pytest.raises(QueryValidationError):
        # n exists in the schema but is not projected.
        query_file(path, "SELECT DISTINCT s FROM input ORDER BY n")
    with pytest.raises(QueryValidationError):
        # Completely unknown name.
        query_file(path, "SELECT DISTINCT s FROM input ORDER BY nope")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s FROM input ORDER BY COUNT(*)")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s FROM input ORDER BY s, s")


def test_distinct_syntax_errors(path):
    for sql in (
        "SELECT DISTINCT FROM input",
        "SELECT DISTINCT",
        "SELECT DISTINCT DISTINCT s FROM input",
        "SELECT s DISTINCT FROM input",
        "SELECT DISTINCT s, FROM input",
        "SELECT s FROM input DISTINCT",
        "SELECT s FROM input LIMIT DISTINCT",
        "SELECT COUNT(DISTINCT *) FROM input",
        "SELECT COUNT(DISTINCT) FROM input",
    ):
        with pytest.raises(QuerySyntaxError):
            query_file(path, sql)


def test_distinct_syntax_errors_before_file_access(tmp_path):
    missing = tmp_path / "nope.caef"
    for sql in (
        "SELECT DISTINCT FROM input",
        "SELECT DISTINCT",
        "SELECT DISTINCT DISTINCT s FROM input",
        "SELECT COUNT(DISTINCT *) FROM input",
    ):
        with pytest.raises(QuerySyntaxError):
            query_file(missing, sql)
        with pytest.raises(QuerySyntaxError):
            explain_file(missing, sql)


# ---------------------------------------------------------------------------
# Two-file joins
# ---------------------------------------------------------------------------


def test_distinct_inner_join_qualified_column(sources):
    result = query_files(
        sources, "SELECT DISTINCT l.k FROM l INNER JOIN r ON l.k = r.k ORDER BY l.k"
    )
    assert result.column_names == ("l.k",)
    assert result.column("l.k") == [10]


def test_distinct_join_star_left(sources):
    result = query_files(
        sources, "SELECT DISTINCT * FROM l LEFT JOIN r ON l.k = r.k"
    )
    assert result.column_names == (
        "l.id",
        "l.k",
        "l.name",
        "r.rid",
        "r.k",
        "r.tag",
    )
    assert rows(result) == [
        [1, 10, "a", 100, 10, "x"],
        [1, 10, "a", 200, 10, "y"],
        [2, 20, "b", None, None, None],
        [3, 10, "c", 100, 10, "x"],
        [3, 10, "c", 200, 10, "y"],
        [4, None, "d", None, None, None],
    ]


def test_distinct_join_dedups_repeated_combined_rows(sources):
    # l.id = 1 and l.id = 3 both join to (100, 10, 'x'); projecting only the
    # right-side columns makes the combined result rows identical.
    result = query_files(
        sources,
        "SELECT DISTINCT r.rid, r.k, r.tag FROM l INNER JOIN r ON l.k = r.k "
        "ORDER BY r.rid",
    )
    assert rows(result) == [
        [100, 10, "x"],
        [200, 10, "y"],
    ]


def test_distinct_join_requires_qualified_names(sources):
    with pytest.raises(QueryValidationError):
        query_files(sources, "SELECT DISTINCT k FROM l INNER JOIN r ON l.k = r.k")


def test_distinct_join_order_by_requires_projected_name(sources):
    with pytest.raises(QueryValidationError):
        query_files(
            sources,
            "SELECT DISTINCT l.k FROM l INNER JOIN r ON l.k = r.k ORDER BY l.id",
        )
    with pytest.raises(QueryValidationError):
        query_files(
            sources,
            "SELECT DISTINCT l.k FROM l INNER JOIN r ON l.k = r.k ORDER BY r.k",
        )


def test_distinct_join_aliased_expression(sources):
    result = query_files(
        sources,
        "SELECT DISTINCT l.k + 1 AS x FROM l INNER JOIN r ON l.k = r.k ORDER BY x",
    )
    assert result.column("x") == [11]


def test_distinct_join_strategies_identical(sources):
    sql = "SELECT DISTINCT r.tag FROM l INNER JOIN r ON l.k = r.k ORDER BY r.tag"
    hash_result = query_files(sources, sql, "hash")
    merge_result = query_files(sources, sql, "sort_merge")
    assert rows(hash_result) == rows(merge_result) == [["x"], ["y"]]


def test_distinct_single_table_via_query_files(sources):
    result = query_files(sources, "SELECT DISTINCT name FROM l ORDER BY name")
    assert result.column_names == ("name",)
    assert result.column("name") == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_distinct_repeated_queries_byte_stable(path):
    sql = "SELECT DISTINCT n, s, flag FROM input WHERE id > 1"
    first = query_file(path, sql)
    for _ in range(3):
        again = query_file(path, sql)
        assert rows(again) == rows(first)
        assert again.schema == first.schema


# ---------------------------------------------------------------------------
# Explain plans
# ---------------------------------------------------------------------------


def test_explain_distinct_minimal_shape(path):
    plan = explain_file(path, "SELECT DISTINCT s FROM input")
    assert operator_kinds(plan) == ["Scan", "Project", "Distinct"]
    assert find_operator(plan, "Distinct") == {"operator": "Distinct", "keys": ["s"]}
    project = find_operator(plan, "Project")
    assert project["expressions"] == [
        {"expression": {"kind": "column", "name": "s"}, "output": "s"}
    ]


def test_explain_distinct_full_stage_order(path):
    plan = explain_file(
        path,
        "SELECT DISTINCT n, s FROM input WHERE id > 1 ORDER BY n DESC NULLS FIRST LIMIT 3",
    )
    assert operator_kinds(plan) == [
        "Scan",
        "Filter",
        "Project",
        "Distinct",
        "Sort",
        "Limit",
    ]
    assert find_operator(plan, "Distinct") == {
        "operator": "Distinct",
        "keys": ["n", "s"],
    }
    assert find_operator(plan, "Sort")["keys"] == [
        {"column": "n", "direction": "DESC", "nulls": "FIRST"}
    ]
    assert find_operator(plan, "Limit") == {"operator": "Limit", "count": 3}


def test_explain_distinct_star_keys_in_schema_order(path):
    plan = explain_file(path, "SELECT DISTINCT * FROM input")
    assert find_operator(plan, "Distinct")["keys"] == list(SCHEMA.names)
    assert operator_kinds(plan) == ["Scan", "Project", "Distinct"]


def test_explain_distinct_expression_alias_key(path):
    plan = explain_file(
        path, "SELECT DISTINCT f + 1 AS up FROM input ORDER BY up"
    )
    assert operator_kinds(plan) == ["Scan", "Project", "Distinct", "Sort"]
    assert find_operator(plan, "Distinct")["keys"] == ["up"]
    assert find_operator(plan, "Sort")["keys"] == [
        {"column": "up", "direction": "ASC", "nulls": "LAST"}
    ]


def test_explain_distinct_limit_without_sort(path):
    plan = explain_file(path, "SELECT DISTINCT s FROM input LIMIT 2")
    assert operator_kinds(plan) == ["Scan", "Project", "Distinct", "Limit"]


def test_explain_distinct_join_plan(sources):
    sql = (
        "SELECT DISTINCT l.k, r.tag FROM l INNER JOIN r ON l.k = r.k "
        "WHERE l.id > 0 ORDER BY l.k LIMIT 5"
    )
    plan = explain_files(sources, sql)
    assert operator_kinds(plan) == [
        "Scan",
        "Scan",
        "Join",
        "Filter",
        "Project",
        "Distinct",
        "Sort",
        "Limit",
    ]
    assert find_operator(plan, "Distinct") == {
        "operator": "Distinct",
        "keys": ["l.k", "r.tag"],
    }


def test_explain_distinct_join_strategy_field(sources):
    plan = explain_files(
        sources,
        "SELECT DISTINCT l.k FROM l INNER JOIN r ON l.k = r.k",
        join_strategy="sort_merge",
    )
    join_op = find_operator(plan, "Join")
    assert join_op["strategy"] == "SORT_MERGE"
    assert operator_kinds(plan) == ["Scan", "Scan", "Join", "Project", "Distinct"]
    # Omitted strategy keeps the historical Join operator shape.
    implicit = explain_files(
        sources, "SELECT DISTINCT l.k FROM l INNER JOIN r ON l.k = r.k"
    )
    assert "strategy" not in find_operator(implicit, "Join")


def test_explain_distinct_output_matches_result_schema(path):
    for sql in (
        "SELECT DISTINCT * FROM input",
        "SELECT DISTINCT n, s FROM input WHERE id > 2",
        "SELECT DISTINCT f * 2 AS d, flag FROM input ORDER BY d",
        "SELECT DISTINCT n FROM input LIMIT 0",
        "SELECT DISTINCT n FROM input WHERE id > 100",
    ):
        plan = explain_file(path, sql)
        result = query_file(path, sql)
        assert plan["output"] == [
            {"name": col.name, "type": col.type, "nullable": col.nullable}
            for col in result.schema.columns
        ]


def test_explain_without_distinct_is_unchanged(path):
    plan = explain_file(path, "SELECT s FROM input WHERE id > 1 ORDER BY s LIMIT 2")
    kinds = operator_kinds(plan)
    assert "Distinct" not in kinds
    assert kinds == ["Scan", "Filter", "Sort", "Limit", "Project"]


def test_explain_distinct_is_metadata_only(path):
    raw = bytearray(path.read_bytes())
    # Perturb a data-section byte; the overall footer CRC then mismatches.
    body_end = len(raw) - 8
    raw[body_end - 1] ^= 0xFF
    corrupt = path.with_name("corrupt.caef")
    corrupt.write_bytes(bytes(raw))
    plan = explain_file(corrupt, "SELECT DISTINCT s FROM input")
    assert plan["output"][0]["name"] == "s"
    with pytest.raises(ColumnarFormatError):
        read_file(corrupt)


def test_explain_distinct_validation_errors(path):
    with pytest.raises(QueryValidationError):
        explain_file(path, "SELECT DISTINCT COUNT(*) FROM input")
    with pytest.raises(QueryValidationError):
        explain_file(path, "SELECT DISTINCT s FROM input GROUP BY s")
    with pytest.raises(QueryValidationError):
        explain_file(path, "SELECT DISTINCT s FROM input ORDER BY n")
    with pytest.raises(QueryValidationError):
        explain_file(path, "SELECT DISTINCT nope FROM input")


def test_explain_distinct_deterministic_and_keyword_insensitive(path):
    first = explain_file(path, "SELECT DISTINCT n, s FROM input WHERE id > 1 ORDER BY n")
    second = explain_file(path, "select distinct n, s from input where id > 1 order by n")
    assert first == second


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_export_distinct_csv(path, tmp_path):
    dst = tmp_path / "out.csv"
    count = export_query_file(
        path, "SELECT DISTINCT n, s FROM input ORDER BY n NULLS FIRST, s NULLS FIRST", dst
    )
    blob = dst.read_bytes()
    # Distinct (n,s) pairs: (10,a) id1, (NULL,b) id2, (10,NULL) id3,
    # (NULL,c) id5, (30,a) id6 -> five rows ordered with NULLs first.
    assert count == 5
    assert blob == (
        b"n,s\n"
        b",b\n"
        b",c\n"
        b"10,\n"
        b"10,a\n"
        b"30,a\n"
    )


def test_export_distinct_jsonl_and_row_count(path, tmp_path):
    dst = tmp_path / "out.jsonl"
    count = export_query_file(
        path, "SELECT DISTINCT flag FROM input ORDER BY flag", dst, "jsonl"
    )
    assert count == 2
    assert dst.read_bytes() == b'{"flag":false}\n{"flag":true}\n'


def test_export_distinct_zero_rows_is_header_only_csv(path, tmp_path):
    dst = tmp_path / "out.csv"
    count = export_query_file(path, "SELECT DISTINCT s FROM input WHERE id > 9 LIMIT 0", dst)
    assert count == 0
    assert dst.read_bytes() == b"s\n"


def test_export_distinct_zero_rows_jsonl_empty(path, tmp_path):
    dst = tmp_path / "out.jsonl"
    count = export_query_file(
        path, "SELECT DISTINCT s FROM input WHERE id > 9", dst, "jsonl"
    )
    assert count == 0
    assert dst.read_bytes() == b""


def test_export_distinct_join(sources, tmp_path):
    dst = tmp_path / "out.csv"
    count = export_query_files(
        sources,
        "SELECT DISTINCT r.tag FROM l INNER JOIN r ON l.k = r.k ORDER BY r.tag",
        dst,
    )
    assert count == 2
    assert dst.read_bytes() == b"r.tag\nx\ny\n"
    other = tmp_path / "m.jsonl"
    export_query_files(
        sources,
        "SELECT DISTINCT r.tag FROM l INNER JOIN r ON l.k = r.k ORDER BY r.tag",
        other,
        "jsonl",
        "sort_merge",
    )
    assert other.read_bytes() == b'{"r.tag":"x"}\n{"r.tag":"y"}\n'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_distinct_query(path, capsys):
    code = main(["query", str(path), "SELECT DISTINCT s FROM input ORDER BY s LIMIT 2"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [{"name": "s", "type": "utf8", "nullable": True}]
    assert payload["rows"] == [["a"], ["b"]]


def test_cli_distinct_query_byte_stable(path, capsys):
    outs = set()
    for sql in (
        "SELECT DISTINCT n FROM input ORDER BY n NULLS FIRST",
        "select  distinct n from input order by n nulls first",
    ):
        assert main(["query", str(path), sql]) == 0
        outs.add(capsys.readouterr().out)
    assert len(outs) == 1


def test_cli_distinct_syntax_error_exit_2(path, capsys):
    code = main(["query", str(path), "SELECT DISTINCT FROM input"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_distinct_validation_error_exit_2(path, capsys):
    code = main(["query", str(path), "SELECT DISTINCT s FROM input ORDER BY n"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_explain_distinct_single_line(path, capsys):
    code = main(["explain", str(path), "SELECT DISTINCT s FROM input ORDER BY s"])
    assert code == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    payload = json.loads(out)
    assert [op["operator"] for op in payload["operators"]] == [
        "Scan",
        "Project",
        "Distinct",
        "Sort",
    ]
    assert ", " not in out.rstrip("\n")


def test_cli_export_distinct(path, tmp_path, capsys):
    dst = tmp_path / "out.csv"
    code = main(
        ["export", str(path), "SELECT DISTINCT flag FROM input ORDER BY flag", str(dst)]
    )
    assert code == 0
    assert capsys.readouterr().out == ""
    assert dst.read_bytes() == b"flag\nfalse\ntrue\n"
