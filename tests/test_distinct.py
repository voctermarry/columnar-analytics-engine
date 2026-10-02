"""Tests for SELECT DISTINCT row deduplication (single- and two-file)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys

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
        ColumnSchema("flag_n", "bool", nullable=True),
    ]
)

DATA = {
    "id": [1, 2, 3, 4, 5, 6],
    "n": [10, None, 30, 40, None, 10],
    "f": [1.5, 2.5, None, -0.0, 10.0, 1.5],
    "s": ["a", "b", None, "a", "c", "a"],
    "flag": [True, False, True, False, True, False],
    "flag_n": [True, False, None, True, None, False],
}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "t.caef"
    write_file(p, Table(SCHEMA, DATA), compression="zlib", dictionary_encoding=["s"])
    return p


def rows_of(table):
    return [
        [table._columns[c][r] for c in range(len(table.schema.columns))]
        for r in range(table.row_count)
    ]


# ---------------------------------------------------------------------------
# Basic deduplication
# ---------------------------------------------------------------------------


def test_distinct_single_column_first_occurrence(path):
    result = query_file(path, "SELECT DISTINCT s FROM input")
    assert result.column_names == ("s",)
    # a (row 1), b (row 2), NULL (row 3), c (row 5) -- first-occurrence order.
    assert result.column("s") == ["a", "b", None, "c"]
    assert result.row_count == 4


def test_distinct_keyword_is_case_insensitive(path):
    result = query_file(path, "select DiStInCt s from input")
    assert result.column("s") == ["a", "b", None, "c"]


def test_distinct_multiple_columns_compare_full_rows(path):
    # (s, flag): (a,T),(b,F),(NULL,T),(a,F),(c,T),(a,F) -- the last (a,F)
    # equals row 4 only as a pair and must collapse.
    result = query_file(path, "SELECT DISTINCT s, flag FROM input")
    assert rows_of(result) == [
        ["a", True],
        ["b", False],
        [None, True],
        ["a", False],
        ["c", True],
    ]


def test_distinct_star_deduplicates_complete_rows(path):
    result = query_file(path, "SELECT DISTINCT * FROM input")
    # Rows 1 and 6 differ (id 1 vs 6, f 1.5 duplicated but the rest differs),
    # so the six-row fixture has no identical full rows.
    assert result.row_count == 6
    assert result.column_names == SCHEMA.names


def test_distinct_star_collapses_identical_rows(tmp_path):
    schema = Schema(
        [ColumnSchema("a", "int64"), ColumnSchema("b", "utf8", nullable=True)]
    )
    p = tmp_path / "dup.caef"
    write_file(
        p,
        Table(
            schema,
            {"a": [1, 2, 1, 3, 1], "b": ["x", "y", "x", None, "x"]},
        ),
    )
    result = query_file(p, "SELECT DISTINCT * FROM input")
    assert rows_of(result) == [[1, "x"], [2, "y"], [3, None]]


def test_distinct_column_projection_subset(path):
    result = query_file(path, "SELECT DISTINCT n, f FROM input")
    # (10,1.5),(NULL,2.5),(30,NULL),(40,-0.0),(NULL,10.0),(10,1.5 dup)
    assert rows_of(result) == [
        [10, 1.5],
        [None, 2.5],
        [30, None],
        [40, -0.0],
        [None, 10.0],
    ]


def test_distinct_int64_and_bool_equality(path):
    assert query_file(path, "SELECT DISTINCT n FROM input").column("n") == [10, None, 30, 40]
    assert query_file(path, "SELECT DISTINCT flag FROM input").column("flag") == [True, False]
    assert query_file(path, "SELECT DISTINCT flag_n FROM input").column("flag_n") == [
        True,
        False,
        None,
    ]


def test_distinct_nulls_form_one_group(tmp_path):
    schema = Schema([ColumnSchema("s", "utf8", nullable=True)])
    p = tmp_path / "nulls.caef"
    write_file(p, Table(schema, {"s": [None, None, "a", None]}))
    result = query_file(p, "SELECT DISTINCT s FROM input")
    assert result.column("s") == [None, "a"]


def test_distinct_float_signed_zero_is_equal(tmp_path):
    schema = Schema([ColumnSchema("f", "float64")])
    p = tmp_path / "zeros.caef"
    write_file(p, Table(schema, {"f": [0.0, -0.0, 0.0, -0.0]}))
    result = query_file(p, "SELECT DISTINCT f FROM input")
    assert result.column("f") == [0.0]
    assert result.row_count == 1


def test_distinct_applied_after_where(path):
    # s among rows with id >= 4: a(row4), c(row5), a(row6) -> a, c.
    result = query_file(path, "SELECT DISTINCT s FROM input WHERE id >= 4")
    assert result.column("s") == ["a", "c"]
    # WHERE keeps file row order: id 1 (n=10), 2 (NULL), 5 (NULL) survive,
    # so the first-occurrence order is 10 then NULL.
    result = query_file(
        path, "SELECT DISTINCT n FROM input WHERE id = 5 OR id = 2 OR id = 1"
    )
    assert result.column("n") == [10, None]


def test_distinct_on_empty_filter_keeps_columns(path):
    result = query_file(path, "SELECT DISTINCT s, n FROM input WHERE id > 1000")
    assert result.column_names == ("s", "n")
    assert result.row_count == 0
    assert result.columns == {"s": [], "n": []}


def test_distinct_limit_zero_keeps_columns(path):
    result = query_file(path, "SELECT DISTINCT s, f FROM input LIMIT 0")
    assert result.column_names == ("s", "f")
    assert result.row_count == 0
    assert result.columns == {"s": [], "f": []}


def test_distinct_on_empty_file(tmp_path):
    p = tmp_path / "empty.caef"
    write_file(p, Table(SCHEMA, {name: [] for name in SCHEMA.names}))
    result = query_file(p, "SELECT DISTINCT s FROM input")
    assert result.column_names == ("s",)
    assert result.row_count == 0


def test_distinct_with_scalar_expression_alias(path):
    # n + 100 (NULL stays NULL): rows give 110, NULL, 130, 140, NULL, 110.
    result = query_file(path, "SELECT DISTINCT n + 100 AS x FROM input")
    assert result.column("x") == [110, None, 130, 140]
    assert result.schema.columns == (ColumnSchema("x", "int64", nullable=True),)


def test_distinct_with_case_expression_alias(path):
    result = query_file(
        path,
        "SELECT DISTINCT CASE WHEN flag THEN 'yes' ELSE 'no' END AS tag FROM input",
    )
    assert result.column("tag") == ["yes", "no"]


def test_distinct_expression_evaluated_only_for_where_survivors(path):
    # No row passes WHERE: the division is never evaluated.
    result = query_file(
        path,
        "SELECT DISTINCT 1 / (id - 1) AS x FROM input WHERE id > 1000",
    )
    assert result.column("x") == []


def test_distinct_expression_error_on_surviving_row(path):
    with pytest.raises(QueryValidationError):
        # Row 1 reaches the projection (id - 1 = 0).
        query_file(path, "SELECT DISTINCT 1 / (id - 1) AS x FROM input WHERE id >= 1")
    with pytest.raises(QueryValidationError):
        # DISTINCT projects every surviving row before LIMIT, so LIMIT 0
        # does not spare the division by zero (unlike the non-distinct path).
        query_file(
            path,
            "SELECT DISTINCT 1 / (id - 1) AS x FROM input WHERE id >= 1 LIMIT 0",
        )


def test_distinct_int64_overflow_raises(tmp_path):
    schema = Schema([ColumnSchema("a", "int64")])
    p = tmp_path / "big.caef"
    write_file(p, Table(schema, {"a": [2**63 - 1]}))
    with pytest.raises(QueryValidationError):
        query_file(p, "SELECT DISTINCT a + 1 AS x FROM input")


def test_distinct_non_finite_float_raises(tmp_path):
    schema = Schema([ColumnSchema("d", "float64")])
    p = tmp_path / "huge.caef"
    write_file(p, Table(schema, {"d": [1.7976931348623157e308]}))
    with pytest.raises(QueryValidationError):
        query_file(p, "SELECT DISTINCT d * d AS x FROM input")


# ---------------------------------------------------------------------------
# ORDER BY / LIMIT over the deduplicated rows
# ---------------------------------------------------------------------------


def test_distinct_order_by_projected_column(path):
    result = query_file(path, "SELECT DISTINCT n FROM input ORDER BY n")
    # Distinct n: 10, NULL, 30, 40; NULLs last by default.
    assert result.column("n") == [10, 30, 40, None]
    result = query_file(path, "SELECT DISTINCT n FROM input ORDER BY n DESC")
    assert result.column("n") == [40, 30, 10, None]
    result = query_file(
        path, "SELECT DISTINCT n FROM input ORDER BY n DESC NULLS FIRST"
    )
    assert result.column("n") == [None, 40, 30, 10]


def test_distinct_order_by_alias(path):
    result = query_file(
        path,
        "SELECT DISTINCT n + 100 AS x FROM input ORDER BY x DESC NULLS FIRST",
    )
    assert result.column("x") == [None, 140, 130, 110]


def test_distinct_order_by_keeps_first_occurrence_ties(tmp_path):
    schema = Schema(
        [ColumnSchema("flag", "bool"), ColumnSchema("s", "utf8", nullable=True)]
    )
    p = tmp_path / "ties.caef"
    # Distinct pairs: (T,a),(F,b),(F,a),(T,c) in that first-occurrence order.
    write_file(
        p,
        Table(
            schema,
            {
                "flag": [True, False, True, False, True],
                "s": ["a", "b", "a", "a", "c"],
            },
        ),
    )
    result = query_file(p, "SELECT DISTINCT flag, s FROM input ORDER BY flag")
    assert rows_of(result) == [[False, "b"], [False, "a"], [True, "a"], [True, "c"]]
    result = query_file(p, "SELECT DISTINCT flag, s FROM input ORDER BY flag DESC")
    assert rows_of(result) == [[True, "a"], [True, "c"], [False, "b"], [False, "a"]]


def test_distinct_order_by_multiple_keys(path):
    result = query_file(
        path, "SELECT DISTINCT s, n FROM input ORDER BY s DESC NULLS FIRST, n"
    )
    # Distinct (s,n): (a,10),(b,NULL),(NULL,30),(a,40),(c,NULL),(a,10 dup)
    # -> (a,10),(b,NULL),(NULL,30),(a,40),(c,NULL); sorted s DESC, NULL first.
    assert rows_of(result) == [
        [None, 30],
        ["c", None],
        ["b", None],
        ["a", 10],
        ["a", 40],
    ]


def test_distinct_limit_with_and_without_order(path):
    assert query_file(path, "SELECT DISTINCT s FROM input LIMIT 2").column("s") == [
        "a",
        "b",
    ]
    result = query_file(
        path, "SELECT DISTINCT s FROM input ORDER BY s DESC LIMIT 2"
    )
    assert result.column("s") == ["c", "b"]


def test_distinct_order_by_unprojected_column_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s FROM input ORDER BY id")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s FROM input ORDER BY f, s")


def test_distinct_order_by_unknown_name_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s FROM input ORDER BY missing")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT n + 1 AS x FROM input ORDER BY nope")


def test_distinct_order_by_duplicate_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s FROM input ORDER BY s, s")


def test_distinct_order_by_aggregate_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s FROM input ORDER BY COUNT(*)")


# ---------------------------------------------------------------------------
# Conflicts and syntax errors
# ---------------------------------------------------------------------------


def test_distinct_with_group_by_having_or_aggregate_rejected(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s, COUNT(*) FROM input")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s, COUNT(*) FROM input GROUP BY s")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s FROM input GROUP BY s")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s FROM input HAVING COUNT(*) > 0")
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT s FROM input GROUP BY s HAVING COUNT(*) > 0")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT DISTINCT FROM input",
        "SELECT DISTINCT",
        "SELECT DISTINCT DISTINCT s FROM input",
        "SELECT s DISTINCT FROM input",
        "SELECT DISTINCT , s FROM input",
    ],
)
def test_distinct_syntax_errors(path, sql):
    with pytest.raises(QuerySyntaxError):
        query_file(path, sql)


def test_count_distinct_argument_not_supported(path):
    # COUNT(DISTINCT ...) is not part of this iteration's grammar.
    with pytest.raises(QuerySyntaxError):
        query_file(path, "SELECT COUNT(DISTINCT s) FROM input")


def test_distinct_syntax_checked_before_file(tmp_path):
    missing = tmp_path / "nope.caef"
    with pytest.raises(QuerySyntaxError):
        query_file(missing, "SELECT DISTINCT FROM input")
    with pytest.raises(QuerySyntaxError):
        query_file(missing, "SELECT DISTINCT DISTINCT s FROM input")
    # A valid DISTINCT statement against a missing file still reaches the file.
    with pytest.raises(FileNotFoundError):
        query_file(missing, "SELECT DISTINCT s FROM input")


def test_distinct_validation_uses_existing_unknown_column(path):
    with pytest.raises(QueryValidationError):
        query_file(path, "SELECT DISTINCT missing FROM input")


# ---------------------------------------------------------------------------
# Explain plans
# ---------------------------------------------------------------------------


def operator_kinds(plan):
    return [op["operator"] for op in plan["operators"]]


def find_operator(plan, name):
    for op in plan["operators"]:
        if op["operator"] == name:
            return op
    raise KeyError(name)


def test_explain_distinct_minimal_shape(path):
    plan = explain_file(path, "SELECT DISTINCT s FROM input")
    assert operator_kinds(plan) == ["Scan", "Project", "Distinct"]
    distinct = find_operator(plan, "Distinct")
    assert distinct == {"operator": "Distinct", "keys": ["s"]}
    project = find_operator(plan, "Project")
    assert project["expressions"] == [
        {"expression": {"kind": "column", "name": "s"}, "output": "s"}
    ]
    assert plan["output"] == [
        {"name": "s", "type": "utf8", "nullable": True}
    ]


def test_explain_distinct_full_stage_order(path):
    plan = explain_file(
        path,
        "SELECT DISTINCT s, n + 1 AS x FROM input WHERE id > 1 "
        "ORDER BY s DESC NULLS FIRST, x LIMIT 3",
    )
    assert operator_kinds(plan) == [
        "Scan",
        "Filter",
        "Project",
        "Distinct",
        "Sort",
        "Limit",
    ]
    assert find_operator(plan, "Distinct")["keys"] == ["s", "x"]
    assert find_operator(plan, "Sort")["keys"] == [
        {"column": "s", "direction": "DESC", "nulls": "FIRST"},
        {"column": "x", "direction": "ASC", "nulls": "LAST"},
    ]
    assert find_operator(plan, "Limit") == {"operator": "Limit", "count": 3}


def test_explain_distinct_star_keys_follow_result_order(path):
    plan = explain_file(path, "SELECT DISTINCT * FROM input")
    assert find_operator(plan, "Distinct")["keys"] == list(SCHEMA.names)


def test_explain_distinct_output_matches_query_schema(path):
    sql = "SELECT DISTINCT s, n * 2 AS d FROM input WHERE f IS NOT NULL ORDER BY s"
    plan = explain_file(path, sql)
    result = query_file(path, sql)
    assert plan["output"] == [
        {"name": col.name, "type": col.type, "nullable": col.nullable}
        for col in result.schema.columns
    ]


def test_explain_without_distinct_plan_unchanged(path):
    plain = explain_file(path, "SELECT s FROM input WHERE id > 1 ORDER BY s LIMIT 2")
    assert operator_kinds(plain) == ["Scan", "Filter", "Sort", "Limit", "Project"]
    assert not any(op["operator"] == "Distinct" for op in plain["operators"])


# ---------------------------------------------------------------------------
# Two-file queries
# ---------------------------------------------------------------------------


RIGHT_SCHEMA = Schema(
    [
        ColumnSchema("rid", "int64"),
        ColumnSchema("k", "int64"),
        ColumnSchema("t", "utf8", nullable=True),
    ]
)
RIGHT_DATA = {"rid": [100, 200, 300, 400], "k": [1, 1, 2, 2], "t": ["a", "a", "b", None]}


@pytest.fixture()
def sources(tmp_path, path):
    right = tmp_path / "r.caef"
    write_file(right, Table(RIGHT_SCHEMA, RIGHT_DATA))
    return {"l": path, "r": right}


def test_distinct_join_deduplicates_qualified_rows(sources):
    # l (id, s): rows 1(a),2(b),3(NULL),4(a),5(c),6(a) joined on l.id = r.k:
    # k=1 matches rid 100/200 twice per left row 1; k=2 matches 300/400 per
    # left row 2. Projecting r.t collapses the duplicated matches.
    result = query_files(
        sources,
        "SELECT DISTINCT r.t FROM l INNER JOIN r ON l.id = r.k ORDER BY r.t",
    )
    assert result.column("r.t") == ["a", "b", None]


def test_distinct_join_star(sources):
    result = query_files(
        sources, "SELECT DISTINCT * FROM l INNER JOIN r ON l.id = r.k"
    )
    # Left rows 1 and 2 each match two right rows: four distinct full rows.
    assert result.row_count == 4
    assert result.column_names == tuple(f"l.{n}" for n in SCHEMA.names) + tuple(
        f"r.{n}" for n in RIGHT_SCHEMA.names
    )


def test_distinct_left_join(sources):
    result = query_files(
        sources,
        "SELECT DISTINCT l.s, r.t FROM l LEFT JOIN r ON l.id = r.k "
        "ORDER BY l.s NULLS FIRST, r.t",
    )
    # Matched rows: id 1 (s=a) -> r.t a; id 2 (s=b) -> r.t b and NULL.
    # Unmatched left rows (s NULL/a/c, including s=a rows 4 and 6) carry
    # NULL on the right.  Ties keep each pair's first-occurrence order.
    assert rows_of(result) == [
        [None, None],
        ["a", "a"],
        ["a", None],
        ["b", "b"],
        ["b", None],
        ["c", None],
    ]


def test_distinct_join_order_by_projected_qualified_column(sources):
    result = query_files(
        sources,
        "SELECT DISTINCT l.s FROM l INNER JOIN r ON l.id = r.k ORDER BY l.s DESC",
    )
    assert result.column("l.s") == ["b", "a"]


def test_distinct_join_order_by_unprojected_rejected(sources):
    with pytest.raises(QueryValidationError):
        query_files(
            sources,
            "SELECT DISTINCT l.s FROM l INNER JOIN r ON l.id = r.k ORDER BY r.t",
        )
    with pytest.raises(QueryValidationError):
        query_files(
            sources,
            "SELECT DISTINCT l.s FROM l INNER JOIN r ON l.id = r.k ORDER BY l.id",
        )


def test_distinct_join_explain(sources):
    plan = explain_files(
        sources,
        "SELECT DISTINCT l.s, r.t FROM l INNER JOIN r ON l.id = r.k "
        "WHERE l.flag = TRUE ORDER BY l.s LIMIT 2",
    )
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
    assert find_operator(plan, "Distinct")["keys"] == ["l.s", "r.t"]
    # Both join strategies produce identical deduplicated results.
    sql = (
        "SELECT DISTINCT l.s, r.t FROM l INNER JOIN r ON l.id = r.k "
        "ORDER BY l.s NULLS FIRST, r.t NULLS FIRST"
    )
    hash_result = query_files(sources, sql, "hash")
    merge_result = query_files(sources, sql, "sort_merge")
    assert rows_of(hash_result) == rows_of(merge_result)
    merge_plan = explain_files(sources, sql, "sort_merge")
    assert find_operator(merge_plan, "Join")["strategy"] == "SORT_MERGE"
    assert operator_kinds(merge_plan) == [
        "Scan",
        "Scan",
        "Join",
        "Project",
        "Distinct",
        "Sort",
    ]


# ---------------------------------------------------------------------------
# Exports and CLI reuse the deduplicated result
# ---------------------------------------------------------------------------


def test_export_distinct_csv_and_row_count(path, tmp_path):
    dst = tmp_path / "out.csv"
    count = export_query_file(path, "SELECT DISTINCT s FROM input ORDER BY s", dst)
    assert count == 4
    # NULL sorts last and renders as an empty field; every line ends in LF.
    assert dst.read_bytes() == b"s\na\nb\nc\n\n"


def test_export_distinct_jsonl(path, tmp_path):
    dst = tmp_path / "out.jsonl"
    count = export_query_file(
        path, "SELECT DISTINCT n FROM input ORDER BY n DESC NULLS FIRST LIMIT 2", dst,
        format="jsonl",
    )
    assert count == 2
    lines = dst.read_bytes().decode("utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"n": None}, {"n": 40}]


def test_export_files_distinct(sources, tmp_path):
    dst = tmp_path / "out.csv"
    count = export_query_files(
        sources,
        "SELECT DISTINCT r.t FROM l INNER JOIN r ON l.id = r.k ORDER BY r.t",
        dst,
    )
    assert count == 3
    assert dst.read_bytes() == b"r.t\na\nb\n\n"


def test_cli_distinct_query(path, capsys):
    code = main(["query", str(path), "SELECT DISTINCT flag FROM input ORDER BY flag"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [{"name": "flag", "type": "bool", "nullable": False}]
    assert payload["rows"] == [[False], [True]]


def test_cli_distinct_validation_error_exit_2(path, capsys):
    code = main(
        ["query", str(path), "SELECT DISTINCT s FROM input ORDER BY id"]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_distinct_explain(path, capsys):
    code = main(["explain", str(path), "SELECT DISTINCT s FROM input LIMIT 1"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [op["operator"] for op in payload["operators"]] == [
        "Scan",
        "Project",
        "Distinct",
        "Limit",
    ]


def test_cli_distinct_byte_stable(path):
    sql = "SELECT DISTINCT s, n FROM input WHERE id >= 2 ORDER BY s NULLS FIRST"
    outs = set()
    for _ in range(3):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert main(["query", str(path), sql]) == 0
        outs.add(buf.getvalue())
    assert len(outs) == 1


def test_cli_distinct_subprocess(path):
    env = dict(
        os.environ,
        PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "columnar_analytics.cli",
            "query",
            str(path),
            "SELECT DISTINCT s FROM input WHERE s IS NOT NULL ORDER BY s",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["rows"] == [["a"], ["b"], ["c"]]
