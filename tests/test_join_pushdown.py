"""Tests for row-group statistics pushdown over all-INNER join chains."""

from __future__ import annotations

import json
import struct

import pytest

from columnar_analytics import (
    ColumnSchema,
    ColumnarFormatError,
    Schema,
    Table,
    explain_files,
    export_query_files,
    query_files,
    write_file,
    write_partitioned_file,
)
from columnar_analytics.cli import main


A_SCHEMA = Schema(
    [
        ColumnSchema("aid", "int64"),
        ColumnSchema("ka", "int64", nullable=True),
        ColumnSchema("va", "utf8", nullable=True),
    ]
)
A_DATA = {
    "aid": [1, 2, 3, 4, 5, 6],
    "ka": [10, None, 30, 40, None, 60],
    "va": ["a", "b", None, "c", "a", "d"],
}

B_SCHEMA = Schema(
    [
        ColumnSchema("bid", "int64"),
        ColumnSchema("kb", "int64", nullable=True),
        ColumnSchema("jb", "int64"),
    ]
)
B_DATA = {
    "bid": [10, 20, 30, 40],
    "kb": [10, 10, 30, 40],
    "jb": [1, 2, 3, 4],
}

C_SCHEMA = Schema(
    [
        ColumnSchema("cid", "int64"),
        ColumnSchema("jc", "int64", nullable=True),
        ColumnSchema("tc", "utf8", nullable=True),
    ]
)
C_DATA = {
    "cid": [7, 8, 9, 11],
    "jc": [2, 2, None, 4],
    "tc": ["x", "y", None, "z"],
}

# a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc yields
# (a1,b20,c7), (a1,b20,c8), (a4,b40,c11).
INNER_CHAIN = (
    "SELECT * FROM a "
    "INNER JOIN b ON a.ka = b.kb "
    "INNER JOIN c ON b.jb = c.jc"
)


@pytest.fixture()
def paths(tmp_path):
    made = {}
    rows = (
        ("a", A_SCHEMA, A_DATA, 2),
        ("b", B_SCHEMA, B_DATA, 2),
        ("c", C_SCHEMA, C_DATA, 2),
    )
    for name, schema, data, size in rows:
        v1 = tmp_path / f"{name}v1.caef"
        v2 = tmp_path / f"{name}v2.caef"
        write_file(v1, Table(schema, data), compression="zlib")
        write_partitioned_file(v2, Table(schema, data), size, compression="zlib")
        made[name] = (v1, v2)
    out = {}
    for version_label, index in (("v1", 0), ("v2", 1)):
        out[version_label] = {k: v[index] for k, v in made.items()}
    return out


def _corrupt_block(path, group_index, column_name):
    """Flip one byte of a (group, column) block; the header stays intact."""
    blob = bytearray(path.read_bytes())
    header_len = struct.unpack("<I", blob[5:9])[0]
    header = json.loads(bytes(blob[9 : 9 + header_len]).decode())
    col_index = [c["name"] for c in header["columns"]].index(column_name)
    block = header["row_groups"][group_index]["columns"][col_index]
    blob[9 + header_len + block["offset"]] ^= 0xFF
    path.write_bytes(blob)


def scans(plan):
    return {op["source"]: op for op in plan["operators"] if op["operator"] == "Scan"}


def joined_rows(result):
    return [
        [result._columns[c][i] for c in range(len(result.schema.columns))]
        for i in range(result.row_count)
    ]


# ---------------------------------------------------------------------------
# Result equivalence: v1, v2, mixed versions, every join strategy
# ---------------------------------------------------------------------------


PUSHED_WHERE_QUERIES = [
    # leaf on the FROM source only
    "SELECT a.aid, c.tc FROM a INNER JOIN b ON a.ka = b.kb "
    "INNER JOIN c ON b.jb = c.jc WHERE a.ka > 20",
    # leaves on two different sources, AND-combined per source
    "SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc "
    "WHERE a.ka IS NOT NULL AND c.tc = 'x'",
    # literal on the left is equivalent
    "SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc "
    "WHERE 4 <= b.jb",
    # IS NULL leaf
    "SELECT a.aid FROM a INNER JOIN b ON a.ka = b.kb "
    "INNER JOIN c ON b.jb = c.jc WHERE b.kb IS NULL",
    # OR at the top prunes nothing, sibling AND leaf still prunes c
    "SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc "
    "WHERE (a.va = 'q' OR b.jb = 9) AND c.tc IS NOT NULL",
    # NOT subtree never pushed, the level sibling on b is
    "SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc "
    "WHERE NOT (a.va = 'a') AND b.jb >= 2",
    # arithmetic operand never pushed; the c leaf next to it is
    "SELECT a.aid, c.tc FROM a INNER JOIN b ON a.ka = b.kb "
    "INNER JOIN c ON b.jb = c.jc WHERE a.aid + 0 > 1 AND c.cid > 7",
    # column-to-column (cross-source) comparison never pushed
    "SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc "
    "WHERE a.aid = b.jb AND b.jb = 4",
    # every group of a excluded
    "SELECT COUNT(*) FROM a INNER JOIN b ON a.ka = b.kb "
    "INNER JOIN c ON b.jb = c.jc WHERE a.va = 'z'",
    # aggregate / DISTINCT / ORDER / LIMIT over a pruned chain
    "SELECT c.tc, COUNT(*), SUM(a.aid) FROM a INNER JOIN b ON a.ka = b.kb "
    "INNER JOIN c ON b.jb = c.jc WHERE c.tc IS NOT NULL GROUP BY c.tc ORDER BY c.tc",
    "SELECT DISTINCT c.tc FROM a INNER JOIN b ON a.ka = b.kb "
    "INNER JOIN c ON b.jb = c.jc WHERE b.jb > 1 ORDER BY c.tc LIMIT 2",
]


@pytest.mark.parametrize("sql", PUSHED_WHERE_QUERIES)
@pytest.mark.parametrize("strategy", [None, "hash", "sort_merge"])
def test_version_and_strategy_equivalence(paths, sql, strategy):
    expected = query_files(paths["v1"], sql, strategy)
    variants = [
        paths["v2"],
        {"a": paths["v1"]["a"], "b": paths["v2"]["b"], "c": paths["v2"]["c"]},
        {"a": paths["v2"]["a"], "b": paths["v1"]["b"], "c": paths["v2"]["c"]},
        {"a": paths["v2"]["a"], "b": paths["v2"]["b"], "c": paths["v1"]["c"]},
    ]
    for sources in variants:
        result = query_files(sources, sql, strategy)
        assert result.schema == expected.schema
        assert joined_rows(result) == joined_rows(expected)


def test_inner_chain_rows_with_pruning(paths):
    sql = (
        "SELECT a.aid, b.bid, c.cid FROM a "
        "INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc "
        "WHERE a.ka IS NOT NULL AND c.tc = 'x'"
    )
    assert joined_rows(query_files(paths["v2"], sql)) == [[1, 20, 7]]


def test_all_groups_excluded_chain_semantics(paths):
    sql = (
        "SELECT a.aid FROM a INNER JOIN b ON a.ka = b.kb "
        "INNER JOIN c ON b.jb = c.jc WHERE a.va = 'z'"
    )
    assert query_files(paths["v2"], sql).row_count == 0
    count = (
        "SELECT COUNT(*) FROM a INNER JOIN b ON a.ka = b.kb "
        "INNER JOIN c ON b.jb = c.jc WHERE a.va = 'z'"
    )
    assert query_files(paths["v2"], count).column("COUNT(*)") == [0]


# ---------------------------------------------------------------------------
# Explain plan: per-source fields, order and content
# ---------------------------------------------------------------------------


def test_explain_inner_chain_scan_fields(paths):
    sql = (
        "SELECT a.aid, c.tc FROM a "
        "INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc "
        "WHERE a.ka > 20 AND c.tc = 'x'"
    )
    plan = explain_files(paths["v2"], sql)
    got = scans(plan)
    # a groups: g0 {10,None} excluded by ka > 20, g1 {30,40} and g2 {None,60} kept.
    assert list(got["a"].keys()) == [
        "operator",
        "source",
        "required_columns",
        "row_groups_total",
        "row_groups_selected",
        "pushed_condition",
    ]
    assert got["a"]["row_groups_total"] == 3
    assert got["a"]["row_groups_selected"] == 2
    assert got["a"]["pushed_condition"] == {
        "kind": "comparison",
        "operator": ">",
        "operands": [
            {"kind": "column", "name": "a.ka"},
            {"kind": "literal", "type": "int64", "value": 20},
        ],
    }
    # b has no leaf but still reports total == selected and a null tree;
    # its ON keys remain required columns.
    assert got["b"]["required_columns"] == ["kb", "jb"]
    assert got["b"]["row_groups_total"] == 2
    assert got["b"]["row_groups_selected"] == 2
    assert got["b"]["pushed_condition"] is None
    # c g1 holds tc {None,'z'} and cannot contain 'x', so only g0 survives.
    assert got["c"]["row_groups_total"] == 2
    assert got["c"]["row_groups_selected"] == 1
    assert got["c"]["pushed_condition"]["operands"][0]["name"] == "c.tc"


def test_explain_multiple_leaves_same_source_combine_in_order(paths):
    sql = (
        "SELECT a.aid FROM a INNER JOIN b ON a.ka = b.kb "
        "INNER JOIN c ON b.jb = c.jc "
        "WHERE c.cid > 7 AND c.tc IS NOT NULL"
    )
    plan = explain_files(paths["v2"], sql)
    pushed = scans(plan)["c"]["pushed_condition"]
    assert pushed["kind"] == "logic"
    assert pushed["operator"] == "AND"
    assert [op["kind"] for op in pushed["operands"]] == ["comparison", "is_null"]
    assert pushed["operands"][1]["operator"] == "IS NOT NULL"


def test_explain_literal_left_keeps_sql_shape(paths):
    sql = (
        "SELECT a.aid FROM a INNER JOIN b ON a.ka = b.kb "
        "INNER JOIN c ON b.jb = c.jc WHERE 'x' = c.tc"
    )
    pushed = scans(explain_files(paths["v2"], sql))["c"]["pushed_condition"]
    assert pushed["operator"] == "="
    assert pushed["operands"][0] == {"kind": "literal", "type": "utf8", "value": "x"}
    assert pushed["operands"][1] == {"kind": "column", "name": "c.tc"}


def test_explain_or_not_arith_cross_source_handling(paths):
    sql = (
        "SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc "
        "WHERE (a.va = 'q' OR b.jb = 9) AND NOT (c.tc = 'x') "
        "AND a.aid + 0 > 1 AND b.jb = a.aid AND c.cid > 7"
    )
    got = scans(explain_files(paths["v2"], sql))
    # Only the plain top-AND leaves on c (cid > 7) are pushed; everything
    # under OR / NOT / arithmetic and the cross-source comparison is not.
    assert got["a"]["pushed_condition"] is None
    assert got["b"]["pushed_condition"] is None
    assert got["c"]["pushed_condition"]["operands"][0]["name"] == "c.cid"
    assert got["c"]["row_groups_selected"] == 2


def test_explain_v1_scan_unchanged_in_inner_chain(paths):
    plan = explain_files(paths["v1"], "SELECT * FROM " + INNER_CHAIN.split(" FROM ", 1)[1])
    for scan_op in scans(plan).values():
        assert "row_groups_total" not in scan_op
        assert "row_groups_selected" not in scan_op
        assert "pushed_condition" not in scan_op


def test_explain_mixed_chain_only_v2_scans_annotated(paths):
    sources = {"a": paths["v1"]["a"], "b": paths["v2"]["b"], "c": paths["v2"]["c"]}
    sql = (
        "SELECT * FROM a INNER JOIN b ON a.ka = b.kb "
        "INNER JOIN c ON b.jb = c.jc WHERE b.jb >= 4"
    )
    got = scans(explain_files(sources, sql))
    assert "row_groups_total" not in got["a"]
    # g0 of b (jb 1..2) is excluded; g1 (3..4) survives.
    assert got["b"]["row_groups_total"] == 2
    assert got["b"]["row_groups_selected"] == 1
    assert got["c"]["row_groups_selected"] == 2
    assert got["c"]["pushed_condition"] is None


@pytest.mark.parametrize(
    "join_sql",
    [
        "SELECT * FROM a LEFT JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc",
        "SELECT * FROM a INNER JOIN b ON a.ka = b.kb LEFT JOIN c ON b.jb = c.jc",
        "SELECT * FROM a INNER JOIN b ON a.ka = b.kb RIGHT JOIN c ON b.jb = c.jc",
        "SELECT * FROM a INNER JOIN b ON a.ka = b.kb FULL OUTER JOIN c ON b.jb = c.jc",
    ],
)
def test_explain_outer_chain_scans_unchanged(paths, join_sql):
    sql = join_sql + " WHERE a.ka > 20 AND c.tc = 'x'"
    plan = explain_files(paths["v2"], sql)
    for scan_op in scans(plan).values():
        assert set(scan_op) == {"operator", "source", "required_columns"}


@pytest.mark.parametrize(
    "join_sql",
    [
        "SELECT * FROM a LEFT JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc",
        "SELECT * FROM a INNER JOIN b ON a.ka = b.kb RIGHT JOIN c ON b.jb = c.jc",
        "SELECT * FROM a INNER JOIN b ON a.ka = b.kb FULL OUTER JOIN c ON b.jb = c.jc",
    ],
)
@pytest.mark.parametrize("strategy", [None, "hash", "sort_merge"])
def test_outer_chain_results_match_v1(paths, join_sql, strategy):
    sql = join_sql + " WHERE a.ka > 20"
    expected = query_files(paths["v1"], sql, strategy)
    result = query_files(paths["v2"], sql, strategy)
    assert result.schema == expected.schema
    assert joined_rows(result) == joined_rows(expected)


def test_plan_json_byte_stable(paths):
    sql = (
        "SELECT a.aid, c.tc FROM a INNER JOIN b ON a.ka = b.kb "
        "INNER JOIN c ON b.jb = c.jc WHERE a.ka > 20 AND c.tc = 'x'"
    )
    first = json.dumps(explain_files(paths["v2"], sql), ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False)
    second = json.dumps(explain_files(paths["v2"], sql), ensure_ascii=False,
                        separators=(",", ":"), allow_nan=False)
    assert first == second


# ---------------------------------------------------------------------------
# Selected / excluded groups: corruption isolation and read consistency
# ---------------------------------------------------------------------------


def test_excluded_group_blocks_never_read(paths):
    c_v2 = paths["v2"]["c"]
    # c g1 (tc {None,'z'}) is excluded by tc = 'x'; corrupting its join-key
    # block must not matter.
    _corrupt_block(c_v2, 1, "jc")
    sql = (
        "SELECT a.aid, c.tc FROM a INNER JOIN b ON a.ka = b.kb "
        "INNER JOIN c ON b.jb = c.jc WHERE c.tc = 'x'"
    )
    result = query_files(paths["v2"], sql)
    assert joined_rows(result) == [[1, "x"]]
    # A selected group whose block is corrupt raises the format error.
    _corrupt_block(c_v2, 0, "jc")
    with pytest.raises(ColumnarFormatError):
        query_files(paths["v2"], sql)


def test_explain_counts_match_groups_read(paths):
    # Corrupting one join-key block per source at a time reveals exactly
    # which groups execution reads: selected groups' key blocks raise, the
    # excluded group's corrupt key block is never touched.
    # a.ka IS NULL can hit only g0 (10,NULL) / g2 (NULL,60), never g1;
    # b.jb >= 4 can hit only g1 (jb 3,4), never g0 (jb 1,2);
    # c.tc = 'z' can hit only g1 (tc None,z), never g0 (tc x,y).
    sql = (
        "SELECT * FROM a INNER JOIN b ON a.ka = b.kb "
        "INNER JOIN c ON b.jb = c.jc "
        "WHERE a.ka IS NULL AND b.jb >= 4 AND c.tc = 'z'"
    )
    plan = scans(explain_files(paths["v2"], sql))
    pristine = {
        "a": (A_SCHEMA, A_DATA, "ka", 3, {0, 2}),
        "b": (B_SCHEMA, B_DATA, "kb", 2, {1}),
        "c": (C_SCHEMA, C_DATA, "jc", 2, {1}),
    }

    def rewrite_all():
        write_partitioned_file(paths["v2"]["a"], Table(A_SCHEMA, A_DATA), 2, compression="zlib")
        write_partitioned_file(paths["v2"]["b"], Table(B_SCHEMA, B_DATA), 2, compression="zlib")
        write_partitioned_file(paths["v2"]["c"], Table(C_SCHEMA, C_DATA), 2, compression="zlib")

    for source, (schema, data, key_col, total, expected_read) in pristine.items():
        path = paths["v2"][source]
        observed = set()
        for group in range(total):
            rewrite_all()
            _corrupt_block(path, group, key_col)
            try:
                query_files(paths["v2"], sql)
            except ColumnarFormatError:
                observed.add(group)
        assert observed == expected_read
        assert plan[source]["row_groups_total"] == total
        assert plan[source]["row_groups_selected"] == len(expected_read)
    rewrite_all()


# ---------------------------------------------------------------------------
# Exports and CLI stay byte-identical / reflect the plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("format", ["csv", "jsonl"])
def test_export_bytes_identical_across_versions(paths, tmp_path, format):
    sql = (
        "SELECT a.aid, c.tc FROM a INNER JOIN b ON a.ka = b.kb "
        "INNER JOIN c ON b.jb = c.jc WHERE a.ka > 20 AND c.tc IS NOT NULL "
        "ORDER BY a.aid, c.tc"
    )
    d1 = tmp_path / f"v1.{format}"
    d2 = tmp_path / f"v2.{format}"
    assert export_query_files(paths["v1"], sql, d1, format) == export_query_files(
        paths["v2"], sql, d2, format
    )
    assert d1.read_bytes() == d2.read_bytes()


def test_cli_explain_files_reports_pushdown(paths, capsys):
    sources_json = json.dumps({k: str(v) for k, v in paths["v2"].items()})
    sql = (
        "SELECT * FROM a INNER JOIN b ON a.ka = b.kb "
        "INNER JOIN c ON b.jb = c.jc WHERE b.jb >= 4"
    )
    assert main(["explain-files", sources_json, sql]) == 0
    plan = json.loads(capsys.readouterr().out)
    b_scan = scans(plan)["b"]
    assert b_scan["row_groups_total"] == 2
    assert b_scan["row_groups_selected"] == 1
