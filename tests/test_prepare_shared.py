"""Regression tests for the shared query-preparation chain.

The query, explain and export entries all share one preparation pass
(parsing, source selection, schema binding, required-column collection and
v2 row-group selection).  These tests pin the consequences:

* v1 and v2 sources return identical results for the same statement;
* a JOIN-less ``query_files`` and the equivalent single-file query agree on
  binding, projection pruning and v2 row-group selection;
* an all-INNER chain attributes and pushes leaves per source;
* a chain containing an OUTER join still prunes columns but never prunes row
  groups; and
* the columns and row groups execution decodes are exactly the ones the
  explain plan's Scan operators describe (proved by corrupting blocks: a
  block execution never reads stays harmless, a block it must read raises).
"""

from __future__ import annotations

import json
import struct

import pytest

from columnar_analytics import (
    ColumnSchema,
    ColumnarFormatError,
    Schema,
    Table,
    explain_file,
    explain_files,
    query_file,
    query_files,
    write_file,
    write_partitioned_file,
)
from columnar_analytics.prepare import prepare_mapped, prepare_single

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


def _write_pair(tmp_path, name, schema, data, group_size):
    v1 = tmp_path / f"{name}v1.caef"
    v2 = tmp_path / f"{name}v2.caef"
    write_file(v1, Table(schema, data), compression="zlib")
    write_partitioned_file(v2, Table(schema, data), group_size, compression="zlib")
    return v1, v2


@pytest.fixture()
def sources(tmp_path):
    a1, a2 = _write_pair(tmp_path, "a", A_SCHEMA, A_DATA, 2)
    b1, b2 = _write_pair(tmp_path, "b", B_SCHEMA, B_DATA, 2)
    c1, c2 = _write_pair(tmp_path, "c", C_SCHEMA, C_DATA, 2)
    return {
        "v1": {"a": a1, "b": b1, "c": c1},
        "v2": {"a": a2, "b": b2, "c": c2},
    }


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


def rows(result):
    return [
        [result._columns[c][i] for c in range(len(result.schema.columns))]
        for i in range(result.row_count)
    ]


# ---------------------------------------------------------------------------
# v1 / v2 result equivalence (single file and chains)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("join_strategy", [None, "hash", "sort_merge"])
def test_v1_and_v2_return_identical_results(sources, join_strategy):
    sql = (
        "SELECT a.aid, b.bid FROM a "
        "INNER JOIN b ON a.ka = b.kb "
        "WHERE a.aid > 1 ORDER BY a.aid, b.bid"
    )
    r1 = query_files(sources["v1"], sql, join_strategy)
    r2 = query_files(sources["v2"], sql, join_strategy)
    assert rows(r1) == rows(r2)


def test_single_v1_full_read_and_projection(sources):
    # The single-file grammar over a v1 file reads the file in full but the
    # result only carries the projected column.
    result = query_file(sources["v1"]["a"], "SELECT aid FROM input WHERE ka IS NOT NULL")
    assert result.column_names == ("aid",)
    assert result.column("aid") == [1, 3, 4, 6]


# ---------------------------------------------------------------------------
# JOIN-less mapped query is equivalent to the single-file query
# ---------------------------------------------------------------------------


def test_no_join_mapping_matches_single_file_binding_and_pruning(sources):
    single_sql = "SELECT aid FROM input WHERE ka > 35 ORDER BY aid"
    mapped_sql = "SELECT a.aid FROM a WHERE a.ka > 35 ORDER BY a.aid"

    single = explain_file(sources["v2"]["a"], single_sql)
    mapped = explain_files(sources["v2"], mapped_sql)

    s_scan, m_scan = scans(single)["input"], scans(mapped)["a"]
    assert m_scan["required_columns"] == s_scan["required_columns"] == ["aid", "ka"]
    assert m_scan["row_groups_total"] == s_scan["row_groups_total"]
    assert m_scan["row_groups_selected"] == s_scan["row_groups_selected"]
    # The pushed tree is identical except the single-file column is bare.
    assert m_scan["pushed_condition"] == s_scan["pushed_condition"]

    assert rows(query_files(sources["v2"], mapped_sql)) == rows(
        query_file(sources["v2"]["a"], single_sql)
    )


def test_no_join_mapping_v1_matches_single_file(sources):
    sql_single = "SELECT aid, va FROM input WHERE ka = 10"
    sql_mapped = "SELECT a.aid, a.va FROM a WHERE a.ka = 10"
    assert rows(query_files(sources["v1"], sql_mapped)) == rows(
        query_file(sources["v1"]["a"], sql_single)
    )
    scan = scans(explain_files(sources["v1"], sql_mapped))["a"]
    # v1 scans keep the historical three-field shape (no pushdown fields).
    assert set(scan) == {"operator", "source", "required_columns"}
    assert scan["required_columns"] == ["aid", "ka", "va"]


# ---------------------------------------------------------------------------
# Execution reads exactly the columns and groups the plan describes
# ---------------------------------------------------------------------------


def test_executed_columns_and_groups_match_plan_single_v2(sources):
    path = sources["v2"]["a"]
    sql = "SELECT aid FROM input WHERE ka > 35"

    before = prepare_single(path, sql)
    plan = explain_file(path, sql)
    scan = scans(plan)["input"]
    assert scan["required_columns"] == ["aid", "ka"]
    assert scan["row_groups_selected"] < scan["row_groups_total"]
    kept = set(before.group_selection["input"])
    excluded = set(range(scan["row_groups_total"])) - kept

    # A block of an unread column is never decoded, even in a kept group.
    _corrupt_block(path, 0, "va")
    assert query_file(path, sql).row_count == 2
    # An excluded group's required-column block is never decoded either.
    _corrupt_block(path, sorted(excluded)[0], "aid")
    assert query_file(path, sql).row_count == 2
    # Corrupting a required block in a group the plan keeps must surface.
    _corrupt_block(path, sorted(kept)[0], "ka")
    with pytest.raises(ColumnarFormatError):
        query_file(path, sql)


def test_executed_columns_and_groups_match_plan_inner_chain(sources):
    # Push a leaf to b only; a is read via the join key, c never appears in
    # WHERE but its join key jc must be read.
    sql = (
        "SELECT a.aid, b.bid, c.cid FROM a "
        "INNER JOIN b ON a.ka = b.kb "
        "INNER JOIN c ON b.jb = c.jc "
        "WHERE b.jb = 4"
    )
    prepared = prepare_mapped(sources["v2"], sql, "hash")
    plan = explain_files(sources["v2"], sql, "hash")
    plan_scans = scans(plan)

    # required_columns include each table's join keys but no unused columns.
    assert prepared.required_columns["a"] == ["aid", "ka"]
    assert prepared.required_columns["b"] == ["bid", "kb", "jb"]
    assert prepared.required_columns["c"] == ["cid", "jc"]
    assert plan_scans["a"]["required_columns"] == ["aid", "ka"]
    assert plan_scans["b"]["required_columns"] == ["bid", "kb", "jb"]
    assert plan_scans["c"]["required_columns"] == ["cid", "jc"]
    assert plan_scans["b"]["row_groups_selected"] < plan_scans["b"]["row_groups_total"]

    expected = query_files(sources["v1"], sql, "hash")

    # Unreferenced columns are safe to corrupt for every source.
    _corrupt_block(sources["v2"]["a"], 0, "va")
    _corrupt_block(sources["v2"]["c"], 0, "tc")
    assert rows(query_files(sources["v2"], sql, "hash")) == rows(expected)

    # An excluded b group's required block is never read.
    total = plan_scans["b"]["row_groups_total"]
    kept = set(prepared.group_selection["b"])
    excluded = set(range(total)) - kept
    _corrupt_block(sources["v2"]["b"], sorted(excluded)[0], "kb")
    assert rows(query_files(sources["v2"], sql, "hash")) == rows(expected)

    # ... while a kept b group's required block must be verified.
    _corrupt_block(sources["v2"]["b"], sorted(kept)[0], "jb")
    with pytest.raises(ColumnarFormatError):
        query_files(sources["v2"], sql, "hash")


def test_outer_join_chain_prunes_columns_but_never_row_groups(sources):
    outer_sqls = (
        "SELECT a.aid, b.bid FROM a "
        "LEFT JOIN b ON a.ka = b.kb WHERE a.aid > 1",
        "SELECT a.aid, b.bid FROM a "
        "FULL OUTER JOIN b ON a.ka = b.kb",
        "SELECT a.aid, b.bid FROM a "
        "RIGHT JOIN b ON a.ka = b.kb WHERE b.bid > 0",
    )
    for sql in outer_sqls:
        plan = explain_files(sources["v2"], sql, "sort_merge")
        for scan in scans(plan).values():
            # No row-group statistics are reported under an OUTER step.
            assert "row_groups_total" not in scan
            assert "pushed_condition" not in scan

    # Column pruning still applies: corrupting an unread column is harmless.
    sql = "SELECT a.aid, b.bid FROM a LEFT JOIN b ON a.ka = b.kb"
    _corrupt_block(sources["v2"]["a"], 0, "va")
    _corrupt_block(sources["v2"]["b"], 0, "jb")
    assert rows(query_files(sources["v2"], sql, "sort_merge")) == rows(
        query_files(sources["v1"], sql, "sort_merge")
    )

    # Every row group is read: a required block of a group an INNER-only
    # pushdown would have excluded still has to verify.
    filtered = "SELECT a.aid FROM a LEFT JOIN b ON a.ka = b.kb WHERE a.aid > 5"
    prepared = prepare_mapped(sources["v2"], filtered, "hash")
    assert prepared.group_selection == {}  # no pruning decided for any source
    _corrupt_block(sources["v2"]["a"], 0, "ka")  # group 0 fails a.aid > 5
    with pytest.raises(ColumnarFormatError):
        query_files(sources["v2"], filtered, "hash")


def test_strategy_plan_markers_match_execution_path(sources):
    sql = "SELECT a.aid, b.bid FROM a INNER JOIN b ON a.ka = b.kb"
    for strategy, label in ((None, None), ("hash", "HASH"), ("sort_merge", "SORT_MERGE")):
        plan = explain_files(sources["v1"], sql, strategy)
        joins = [op for op in plan["operators"] if op["operator"] == "Join"]
        if label is None:
            assert all("strategy" not in op for op in joins)
        else:
            assert [op["strategy"] for op in joins] == [label]


def test_unreferenced_source_is_never_opened(tmp_path, sources):
    # A source not named by the statement is never touched, so a path that
    # does not exist is accepted as long as it stays unreferenced.
    missing = tmp_path / "never-opened.caef"
    mapped = dict(sources["v1"])
    mapped["unused"] = missing
    plan = explain_files(mapped, "SELECT a.aid FROM a")
    assert [s["name"] for s in plan["sources"]] == ["a"]
    assert query_files(mapped, "SELECT a.aid FROM a").column("aid") == A_DATA["aid"]
