"""Regression tests for the shared query-preparation pipeline.

The four plan/query entries (``query_file`` / ``query_files`` /
``explain_file`` / ``explain_files``) and the export entries share one
preparation stage -- parsing, source selection, metadata binding,
required-column collection and v2 row-group selection.  These tests pin
the guarantees of that shared stage without relying on any internal
module:

* a JOIN-less ``query_files`` statement binds, trims columns and selects
  v2 row groups exactly like the equivalent ``query_file`` statement;
* the columns and row groups execution reads match the explain Scan of
  the v1 file, the v2 file, an all-INNER chain and a chain with an OUTER
  step (which never prunes);
* query, explain and export classify and order errors the same way
  (syntax / sources / strategy before any file access, then binding
  validation, then metadata, with OSError preserved);
* explain reads only referenced metadata -- never data blocks -- and
  unreferenced sources are never opened.
"""

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
    export_query_files,
    query_file,
    query_files,
    write_file,
    write_partitioned_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


S_SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("s", "utf8", nullable=True),
    ]
)
# Grouped with group size 3 -> groups [0:3], [3:6], [6:9], [9:10].
S_DATA = {
    "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "n": [10, None, 30, 40, None, 60, 70, 80, None, 100],
    "s": ["a", "b", None, "c", None, "a", "b", "c", None, "a"],
}

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
        ColumnSchema("vb", "utf8", nullable=True),
    ]
)
B_DATA = {
    "bid": [10, 20, 30, 40],
    "kb": [10, 10, 30, 40],
    "vb": ["x", "y", "z", "w"],
}


@pytest.fixture()
def single_files(tmp_path):
    v1 = tmp_path / "single_v1.caef"
    v2 = tmp_path / "single_v2.caef"
    write_file(v1, Table(S_SCHEMA, S_DATA), compression="zlib")
    write_partitioned_file(v2, Table(S_SCHEMA, S_DATA), 3, compression="zlib")
    return v1, v2


@pytest.fixture()
def chain_files(tmp_path):
    made = {}
    for name, schema, data in (("a", A_SCHEMA, A_DATA), ("b", B_SCHEMA, B_DATA)):
        v1 = tmp_path / f"{name}_v1.caef"
        v2 = tmp_path / f"{name}_v2.caef"
        write_file(v1, Table(schema, data), compression="zlib")
        write_partitioned_file(v2, Table(schema, data), 2, compression="zlib")
        made[name] = (v1, v2)
    return {
        "v1": {k: v[0] for k, v in made.items()},
        "v2": {k: v[1] for k, v in made.items()},
    }


def _corrupt_block(path, group_index, column_name):
    """Flip one byte of a (group, column) block; the v2 header stays intact."""
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
# JOIN-less mapped query == single-file query
# ---------------------------------------------------------------------------


EQUIVALENT_SQL = [
    "SELECT id, n FROM t WHERE n > 50",
    "SELECT id FROM t WHERE n < 50 ORDER BY id DESC LIMIT 2",
    "SELECT s, COUNT(*), SUM(n) FROM t GROUP BY s ORDER BY s",
    "SELECT COUNT(*) FROM t WHERE n > 1000",
    "SELECT DISTINCT s FROM t WHERE s IS NOT NULL ORDER BY s",
    "SELECT id, n * 2 AS dbl FROM t WHERE id >= 4 ORDER BY dbl, id",
]


@pytest.mark.parametrize("sql", EQUIVALENT_SQL)
@pytest.mark.parametrize("version_label", ["v1", "v2"])
def test_no_join_mapped_query_matches_single_file(single_files, sql, version_label):
    path = single_files[0] if version_label == "v1" else single_files[1]
    expected = query_file(path, sql.replace(" FROM t", " FROM input"))
    actual = query_files({"t": path}, sql)
    assert actual.schema == expected.schema
    assert rows(actual) == rows(expected)


@pytest.mark.parametrize("sql", EQUIVALENT_SQL)
def test_no_join_explain_matches_single_file_plan(single_files, sql):
    _v1, v2 = single_files
    single = explain_file(v2, sql.replace(" FROM t", " FROM input"))
    mapped = explain_files({"t": v2}, sql)
    # Only the source name differs ("input" vs the mapping key "t"); every
    # operator, required column, row-group count and pushed tree is equal.
    assert single["sources"][0]["name"] == "input"
    assert mapped["sources"][0]["name"] == "t"
    for plan in (single, mapped):
        plan["sources"][0]["name"] = "t"
        for op in plan["operators"]:
            if op.get("operator") == "Scan":
                op["source"] = "t"
    assert mapped == single


def test_no_join_mapped_and_single_select_same_row_groups(single_files):
    _v1, v2 = single_files
    sql_t = "SELECT id, n FROM t WHERE n > 50"
    sql_in = "SELECT id, n FROM input WHERE n > 50"
    mapped_scan = scans(explain_files({"t": v2}, sql_t))["t"]
    single_scan = scans(explain_file(v2, sql_in))["input"]
    assert mapped_scan["required_columns"] == ["id", "n"]
    assert mapped_scan["required_columns"] == single_scan["required_columns"]
    assert mapped_scan["row_groups_total"] == single_scan["row_groups_total"] == 4
    assert mapped_scan["row_groups_selected"] == single_scan["row_groups_selected"]
    assert mapped_scan["pushed_condition"] == single_scan["pushed_condition"]


# ---------------------------------------------------------------------------
# v1 / v2 scans: the plan's read range is what execution reads
# ---------------------------------------------------------------------------


def test_v1_scan_shape_and_full_read(single_files):
    v1, _v2 = single_files
    plan = explain_file(v1, "SELECT id FROM input WHERE n > 50")
    assert scans(plan)["input"] == {
        "operator": "Scan",
        "source": "input",
        "required_columns": ["id", "n"],
    }


def test_v2_required_columns_govern_decoded_blocks(single_files):
    _v1, v2 = single_files
    plan = explain_file(v2, "SELECT id FROM input WHERE n > 50")
    required = scans(plan)["input"]["required_columns"]
    assert required == ["id", "n"]
    # "s" is not required: corrupting every "s" block leaves the query fine.
    for group in range(4):
        _corrupt_block(v2, group, "s")
    assert query_file(v2, "SELECT id FROM input WHERE n > 50").column("id") == [
        6,
        7,
        8,
        10,
    ]
    # A required column's selected-group block is decoded and verified.
    _corrupt_block(v2, 1, "n")
    with pytest.raises(ColumnarFormatError):
        query_file(v2, "SELECT id FROM input WHERE n > 50")


def test_v2_executed_row_groups_match_plan(single_files):
    _v1, v2 = single_files
    sql = "SELECT id, n FROM input WHERE n > 50"
    selected = scans(explain_file(v2, sql))["input"]["row_groups_selected"]
    # n maxima per group: g0 30 (excluded), g1 60, g2 80, g3 100 (kept).
    assert selected == 3
    read_groups = set()
    for group in range(4):
        _corrupt_block(v2, group, "n")
        try:
            query_file(v2, sql)
        except ColumnarFormatError:
            read_groups.add(group)
        _corrupt_block(v2, group, "n")  # flip the same byte back
    assert read_groups == {1, 2, 3}


def test_all_groups_excluded_reads_nothing_and_keeps_semantics(single_files):
    _v1, v2 = single_files
    sql = "SELECT COUNT(*) FROM input WHERE s = 'z'"
    scan = scans(explain_file(v2, sql))["input"]
    assert scan["row_groups_selected"] == 0
    # Every "s" block can be corrupt: no group is selected, none is decoded.
    for group in range(4):
        _corrupt_block(v2, group, "s")
    assert query_file(v2, sql).column("COUNT(*)") == [0]
    v1 = single_files[0]
    assert query_file(v1, sql).column("COUNT(*)") == [0]


# ---------------------------------------------------------------------------
# All-INNER chain: per-source columns and groups match between plan and reads
# ---------------------------------------------------------------------------


INNER_CHAIN = "SELECT * FROM a INNER JOIN b ON a.ka = b.kb"


def test_inner_chain_required_columns_and_group_fields(chain_files):
    sources = chain_files["v2"]
    sql = (
        "SELECT a.aid, b.vb FROM a INNER JOIN b ON a.ka = b.kb "
        "WHERE a.ka > 20 AND b.vb IS NOT NULL"
    )
    got = scans(explain_files(sources, sql))
    assert got["a"]["required_columns"] == ["aid", "ka"]
    assert got["b"]["required_columns"] == ["kb", "vb"]
    # a g0 holds ka {10,None}, excluded by ka > 20; g1 {30,40}, g2 {None,60} kept.
    assert (got["a"]["row_groups_total"], got["a"]["row_groups_selected"]) == (3, 2)
    # vb is nullable but never NULL in the data: both b groups stay selected.
    assert (got["b"]["row_groups_total"], got["b"]["row_groups_selected"]) == (2, 2)
    assert got["b"]["pushed_condition"] is not None


def test_inner_chain_executed_groups_match_plan(chain_files):
    sources = chain_files["v2"]
    sql = (
        "SELECT a.aid, b.bid FROM a INNER JOIN b ON a.ka = b.kb "
        "WHERE a.ka > 20"
    )
    plan = scans(explain_files(sources, sql))
    expected = {
        "a": (3, {1, 2}, "ka"),  # g0 excluded, g1/g2 read
        "b": (2, {0, 1}, "kb"),  # no b leaf, every group read
    }
    for key, (total, wanted_groups, key_col) in expected.items():
        observed = set()
        for group in range(total):
            _corrupt_block(sources[key], group, key_col)
            try:
                query_files(sources, sql)
            except ColumnarFormatError:
                observed.add(group)
            _corrupt_block(sources[key], group, key_col)
        assert observed == wanted_groups
        assert plan[key]["row_groups_total"] == total
        assert plan[key]["row_groups_selected"] == len(wanted_groups)


@pytest.mark.parametrize("strategy", [None, "hash", "sort_merge"])
def test_inner_chain_v1_v2_equivalence(chain_files, strategy):
    sql = INNER_CHAIN + " WHERE a.va IS NOT NULL ORDER BY a.aid, b.bid"
    expected = query_files(chain_files["v1"], sql, strategy)
    actual = query_files(chain_files["v2"], sql, strategy)
    assert actual.schema == expected.schema
    assert rows(actual) == rows(expected)


# ---------------------------------------------------------------------------
# OUTER chain: no row-group pruning anywhere, every group is read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "join_sql",
    [
        "SELECT * FROM a LEFT JOIN b ON a.ka = b.kb",
        "SELECT * FROM a RIGHT JOIN b ON a.ka = b.kb",
        "SELECT * FROM a FULL OUTER JOIN b ON a.ka = b.kb",
    ],
)
def test_outer_chain_scans_keep_historical_shape(chain_files, join_sql):
    plan = explain_files(chain_files["v2"], join_sql + " WHERE a.ka > 20")
    for op in scans(plan).values():
        assert set(op) == {"operator", "source", "required_columns"}


@pytest.mark.parametrize(
    "join_sql",
    [
        "SELECT * FROM a LEFT JOIN b ON a.ka = b.kb",
        "SELECT * FROM a RIGHT JOIN b ON a.ka = b.kb",
        "SELECT * FROM a FULL OUTER JOIN b ON a.ka = b.kb",
    ],
)
@pytest.mark.parametrize("strategy", [None, "hash", "sort_merge"])
def test_outer_chain_reads_every_group_and_matches_v1(
    chain_files, join_sql, strategy
):
    sql = join_sql + " WHERE a.ka > 20"
    v2 = chain_files["v2"]
    plan = scans(explain_files(v2, sql))
    # The plan promises no pruning; prove execution really touches every
    # join-key block of every group (a has 3 groups, b has 2).
    for key, total, key_col in (("a", 3, "ka"), ("b", 2, "kb")):
        for group in range(total):
            _corrupt_block(v2[key], group, key_col)
            with pytest.raises(ColumnarFormatError):
                query_files(v2, sql, strategy)
            _corrupt_block(v2[key], group, key_col)
        assert "row_groups_total" not in plan[key]
    expected = query_files(chain_files["v1"], sql, strategy)
    actual = query_files(v2, sql, strategy)
    assert actual.schema == expected.schema
    assert rows(actual) == rows(expected)


# ---------------------------------------------------------------------------
# Explain stays metadata-only; unreferenced sources are never opened
# ---------------------------------------------------------------------------


def test_explain_never_decodes_data_blocks(single_files):
    _v1, v2 = single_files
    # Corrupt a selected group's required-column block: explain still works
    # because it only reads metadata, while execution reports the error.
    _corrupt_block(v2, 1, "n")
    plan = explain_file(v2, "SELECT id, n FROM input WHERE n > 50")
    assert scans(plan)["input"]["row_groups_selected"] == 3
    with pytest.raises(ColumnarFormatError):
        query_file(v2, "SELECT id, n FROM input WHERE n > 50")
    plan2 = explain_files({"t": v2}, "SELECT id, n FROM t WHERE n > 50")
    assert scans(plan2)["t"]["row_groups_selected"] == 3


def test_unreferenced_sources_are_never_opened(single_files, tmp_path):
    v1, v2 = single_files
    ghost = tmp_path / "ghost.caef"  # never created
    sources = {"t": v2, "ghost": ghost}
    plan = explain_files(sources, "SELECT id FROM t WHERE n > 50")
    assert [s["name"] for s in plan["sources"]] == ["t"]
    assert query_files(sources, "SELECT id FROM t WHERE n > 50").column("id") == [
        6,
        7,
        8,
        10,
    ]
    dst = tmp_path / "out.csv"
    export_query_files(sources, "SELECT id FROM t WHERE n > 50", dst)
    assert dst.read_text(encoding="utf-8").splitlines()[0] == "id"


# ---------------------------------------------------------------------------
# Unified error classification and ordering across every entry point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda p, dst: query_file(p, "SELCT id FROM input"),
        lambda p, dst: explain_file(p, "SELCT id FROM input"),
        lambda p, dst: export_query_file(p, "SELCT id FROM input", dst),
        lambda p, dst: query_files({"t": p}, "SELCT id FROM t"),
        lambda p, dst: explain_files({"t": p}, "SELCT id FROM t"),
        lambda p, dst: export_query_files({"t": p}, "SELCT id FROM t", dst),
    ],
)
def test_syntax_errors_precede_file_access(tmp_path, call):
    missing = tmp_path / "does-not-exist.caef"
    with pytest.raises(QuerySyntaxError):
        call(missing, tmp_path / "out.csv")


@pytest.mark.parametrize(
    "call",
    [
        lambda sources, sql, strat: query_files(sources, sql, strat),
        lambda sources, sql, strat: explain_files(sources, sql, strat),
    ],
)
def test_sources_and_strategy_value_errors_precede_access(tmp_path, call):
    missing = tmp_path / "missing.caef"
    good_sql = "SELECT id FROM t"
    for bad_sources in (None, [], {}, {"": missing}, {"t": 123}):
        with pytest.raises(ValueError):
            call(bad_sources, good_sql, None)
    with pytest.raises(ValueError):
        call({"t": missing}, good_sql, "nope")
    # A syntax error is still a syntax error with otherwise valid arguments.
    with pytest.raises(QuerySyntaxError):
        call({"t": missing}, "SELCT id FROM t", None)


def test_binding_validation_errors_consistent(single_files, chain_files):
    _v1, v2 = single_files
    missing_col = "SELECT nope FROM input"
    type_conflict = "SELECT id FROM input WHERE s = 1"
    for sql in (missing_col, type_conflict):
        with pytest.raises(QueryValidationError):
            query_file(v2, sql)
        with pytest.raises(QueryValidationError):
            explain_file(v2, sql)
    sources = chain_files["v2"]
    # Unqualified reference inside a join statement.
    with pytest.raises(QueryValidationError):
        query_files(sources, "SELECT aid FROM a INNER JOIN b ON a.ka = b.kb")
    with pytest.raises(QueryValidationError):
        explain_files(sources, "SELECT aid FROM a INNER JOIN b ON a.ka = b.kb")
    # Unknown table.
    with pytest.raises(QueryValidationError):
        explain_files(sources, "SELECT z.aid FROM z INNER JOIN b ON z.ka = b.kb")
    # Incompatible join key types.
    with pytest.raises(QueryValidationError):
        query_files(
            sources, "SELECT a.aid FROM a INNER JOIN b ON a.va = b.kb"
        )


def test_metadata_errors_and_os_error_consistent(tmp_path):
    bad = tmp_path / "bad.caef"
    bad.write_bytes(b"garbage")
    missing = tmp_path / "nope.caef"
    for entry in (
        lambda: query_file(bad, "SELECT * FROM input"),
        lambda: explain_file(bad, "SELECT * FROM input"),
        lambda: export_query_file(bad, "SELECT * FROM input", tmp_path / "x.csv"),
        lambda: query_files({"t": bad}, "SELECT * FROM t"),
        lambda: explain_files({"t": bad}, "SELECT * FROM t"),
    ):
        with pytest.raises(ColumnarFormatError):
            entry()
    for entry in (
        lambda: query_file(missing, "SELECT * FROM input"),
        lambda: explain_file(missing, "SELECT * FROM input"),
        lambda: query_files({"t": missing}, "SELECT * FROM t"),
        lambda: explain_files({"t": missing}, "SELECT * FROM t"),
    ):
        with pytest.raises(OSError):
            entry()


def test_export_format_and_overlap_value_errors_unchanged(single_files, tmp_path):
    v1, _v2 = single_files
    with pytest.raises(ValueError):
        export_query_file(v1, "SELECT id FROM input", tmp_path / "x.bin", format="bin")
    with pytest.raises(ValueError):
        export_query_file(v1, "SELECT id FROM input", v1)


def test_export_bytes_stable_across_preparation_paths(single_files, tmp_path):
    v1, v2 = single_files
    sql_in = "SELECT id, n FROM input WHERE n > 50 ORDER BY id"
    sql_t = "SELECT id, n FROM t WHERE n > 50 ORDER BY id"
    d1 = tmp_path / "single_v1.csv"
    d2 = tmp_path / "single_v2.csv"
    d3 = tmp_path / "mapped_v2.csv"
    d4 = tmp_path / "single_v2.jsonl"
    d5 = tmp_path / "mapped_v2.jsonl"
    export_query_file(v1, sql_in, d1)
    export_query_file(v2, sql_in, d2)
    export_query_files({"t": v2}, sql_t, d3)
    export_query_file(v2, sql_in, d4, format="jsonl")
    export_query_files({"t": v2}, sql_t, d5, format="jsonl")
    assert d1.read_bytes() == d2.read_bytes() == d3.read_bytes()
    assert d4.read_bytes() == d5.read_bytes()
