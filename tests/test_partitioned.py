"""Tests for CAEF v2 row-group files, projection and statistics pushdown."""

from __future__ import annotations

import json
import os
import struct
import zlib

import pytest

from columnar_analytics import (
    FORMAT_VERSION,
    FORMAT_VERSION_V2,
    ColumnSchema,
    ColumnarFormatError,
    Schema,
    Table,
    inspect_file,
    inspect_row_groups,
    read_file,
    write_file,
    write_partitioned_file,
)
from columnar_analytics.export import export_query_file, export_query_files
from columnar_analytics.query import (
    QueryValidationError,
    explain_file,
    explain_files,
    query_file,
    query_files,
)

SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("cat", "utf8", nullable=True),
        ColumnSchema("ratio", "float64", nullable=True),
        ColumnSchema("ok", "bool"),
    ]
)

N = 12
ROWS = {
    "id": list(range(N)),
    "cat": ["a" if i < 6 else ("b" if i < 9 else None) for i in range(N)],
    "ratio": [None if i % 4 == 3 else float(i) + 0.25 for i in range(N)],
    "ok": [i % 2 == 0 for i in range(N)],
}


def make_table():
    return Table(SCHEMA, ROWS)


@pytest.fixture()
def paths(tmp_path):
    v1 = tmp_path / "v1.caef"
    v2 = tmp_path / "v2.caef"
    write_file(v1, make_table(), compression="zlib", dictionary_encoding=["cat"])
    write_partitioned_file(
        v2, make_table(), 3, compression="zlib", dictionary_encoding=["cat"]
    )
    return v1, v2


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compression", ["none", "zlib"])
@pytest.mark.parametrize("dictionary", [[], ["cat"]])
def test_partitioned_roundtrip(tmp_path, compression, dictionary):
    path = tmp_path / "p.caef"
    write_partitioned_file(
        path, make_table(), 3, compression=compression, dictionary_encoding=dictionary
    )
    restored = read_file(path)
    assert restored.schema == SCHEMA
    assert restored.row_count == N
    assert restored.columns == ROWS
    assert open(path, "rb").read()[4] == FORMAT_VERSION_V2


def test_partitioned_deterministic_bytes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for opts in (
        {"compression": "none"},
        {"compression": "zlib"},
        {"compression": "none", "dictionary_encoding": ["cat"]},
        {"compression": "zlib", "dictionary_encoding": ["cat"]},
    ):
        write_partitioned_file(a, make_table(), 3, **opts)
        write_partitioned_file(b, make_table(), 3, **opts)
        assert a.read_bytes() == b.read_bytes()


def test_row_group_counts(tmp_path):
    path = tmp_path / "p.caef"
    write_partitioned_file(path, make_table(), 5)
    groups = inspect_row_groups(path)["row_groups"]
    assert [g["row_count"] for g in groups] == [5, 5, 2]


def test_group_size_larger_than_rows_makes_one_group(tmp_path):
    path = tmp_path / "p.caef"
    write_partitioned_file(path, make_table(), 1000)
    assert [g["row_count"] for g in inspect_row_groups(path)["row_groups"]] == [N]


def test_empty_table_has_no_row_groups(tmp_path):
    schema = Schema([ColumnSchema("a", "int64", nullable=True)])
    path = tmp_path / "empty.caef"
    write_partitioned_file(path, Table(schema, {"a": []}), 4)
    assert read_file(path).column("a") == []
    assert inspect_row_groups(path)["row_groups"] == []
    assert inspect_file(path)["row_count"] == 0


@pytest.mark.parametrize("bad", [0, -1, -100, 1.5, "3", True, False, 3.0])
def test_invalid_row_group_size(tmp_path, bad):
    path = tmp_path / "bad.caef"
    with pytest.raises(ValueError):
        write_partitioned_file(path, make_table(), bad)
    assert not path.exists()


def test_bad_encoding_options_leave_no_file(tmp_path):
    path = tmp_path / "bad.caef"
    with pytest.raises(ValueError):
        write_partitioned_file(path, make_table(), 3, compression="gzip")
    with pytest.raises(ValueError):
        write_partitioned_file(path, make_table(), 3, dictionary_encoding=["id"])
    with pytest.raises(ValueError):
        write_partitioned_file(path, make_table(), 3, dictionary_encoding=["cat", "cat"])
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_failed_partitioned_write_preserves_existing(tmp_path):
    path = tmp_path / "keep.caef"
    write_partitioned_file(path, make_table(), 3)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_partitioned_file(path, make_table(), 0)
    assert path.read_bytes() == original


# ---------------------------------------------------------------------------
# Projection / row-group subset reads
# ---------------------------------------------------------------------------


def test_projection_subset_and_order(paths):
    _v1, v2 = paths
    table = read_file(v2, columns=["cat", "id"])
    assert table.column_names == ("cat", "id")
    assert table.column("cat") == ROWS["cat"]
    assert table.column("id") == ROWS["id"]


def test_projection_unknown_column(paths):
    _v1, v2 = paths
    with pytest.raises(KeyError):
        read_file(v2, columns=["nope"])


def test_projection_duplicate_column(paths):
    _v1, v2 = paths
    with pytest.raises(ValueError):
        read_file(v2, columns=["id", "id"])


def test_row_group_subset(paths):
    _v1, v2 = paths
    # groups of 3: 0..2, 3..5, 6..8, 9..11
    assert read_file(v2, row_groups=[0, 2]).column("id") == [0, 1, 2, 6, 7, 8]
    assert read_file(v2, row_groups=[3]).column("id") == [9, 10, 11]
    assert read_file(v2, row_groups=[]).column("id") == []


def test_row_group_subset_with_projection(paths):
    _v1, v2 = paths
    table = read_file(v2, columns=["ratio"], row_groups=[1])
    assert table.column_names == ("ratio",)
    assert table.column("ratio") == ROWS["ratio"][3:6]


@pytest.mark.parametrize("bad", [-1, 4, 99])
def test_row_group_index_out_of_range(paths, bad):
    _v1, v2 = paths
    with pytest.raises(ValueError):
        read_file(v2, row_groups=[bad])


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_inspect_v2_aggregates_stats(paths):
    _v1, v2 = paths
    meta = inspect_file(v2)
    assert list(meta) == ["format_version", "row_count", "columns"]
    assert meta["format_version"] == FORMAT_VERSION_V2
    assert meta["row_count"] == N
    by_name = {c["name"]: c for c in meta["columns"]}
    assert by_name["id"]["null_count"] == 0
    assert by_name["id"]["min"] == 0
    assert by_name["id"]["max"] == N - 1
    assert by_name["id"]["row_count"] == N
    assert by_name["cat"]["null_count"] == 3
    assert by_name["cat"]["min"] == "a"
    assert by_name["cat"]["max"] == "b"
    assert by_name["ratio"]["null_count"] == 3
    assert by_name["ratio"]["min"] == pytest.approx(0.25)
    # NULLs land at i = 3, 7, 11, so the largest present ratio is 10.25.
    assert by_name["ratio"]["max"] == pytest.approx(10.25)
    assert by_name["ok"]["min"] is False and by_name["ok"]["max"] is True


def test_inspect_row_groups_structure(paths):
    _v1, v2 = paths
    info = inspect_row_groups(v2)
    assert list(info) == ["format_version", "row_count", "row_groups"]
    assert info["format_version"] == FORMAT_VERSION_V2
    assert info["row_count"] == N
    groups = info["row_groups"]
    assert [g["row_count"] for g in groups] == [3, 3, 3, 3]
    for group in groups:
        assert list(group) == ["row_count", "columns"]
        assert [c["name"] for c in group["columns"]] == [
            "id",
            "cat",
            "ratio",
            "ok",
        ]
        first = group["columns"][0]
        assert set(first) == {"name", "type", "nullable", "null_count", "min", "max"}
    # group 3: id 9..11, cat all NULL
    last = groups[3]["columns"]
    by_name = {c["name"]: c for c in last}
    assert by_name["id"]["min"] == 9 and by_name["id"]["max"] == 11
    assert by_name["cat"]["null_count"] == 3
    assert by_name["cat"]["min"] is None and by_name["cat"]["max"] is None


def test_inspect_row_groups_v1_empty(paths):
    v1, _v2 = paths
    info = inspect_row_groups(v1)
    assert info["format_version"] == FORMAT_VERSION == 1
    assert info["row_count"] == N
    assert info["row_groups"] == []


def test_v1_file_still_version_1(paths):
    v1, _v2 = paths
    assert inspect_file(v1)["format_version"] == FORMAT_VERSION == 1
    assert v1.read_bytes()[4] == 1


# ---------------------------------------------------------------------------
# Query parity between v1 and v2
# ---------------------------------------------------------------------------


def _rows(table):
    return [
        tuple(table._columns[c][r] for c in range(len(table.schema.columns)))
        for r in range(table.row_count)
    ]


PARITY_QUERIES = [
    "SELECT * FROM input",
    "SELECT id, cat FROM input WHERE id >= 4 ORDER BY id",
    "SELECT id FROM input WHERE cat = 'b'",
    "SELECT id FROM input WHERE id >= 5 AND id < 11",
    "SELECT id FROM input WHERE ratio IS NULL",
    "SELECT id FROM input WHERE ratio IS NOT NULL ORDER BY id",
    "SELECT id FROM input WHERE ok = TRUE ORDER BY id",
    "SELECT id FROM input WHERE (id > 100 OR cat = 'a') ORDER BY id",
    "SELECT id FROM input WHERE NOT (id < 6) ORDER BY id",
    "SELECT id FROM input WHERE id = id ORDER BY id",
    "SELECT id FROM input WHERE id + 0 > 8 ORDER BY id",
    "SELECT id, CASE WHEN id < 3 THEN 1 ELSE 0 END AS small FROM input ORDER BY id",
    "SELECT cat, COUNT(*) FROM input GROUP BY cat ORDER BY cat",
    "SELECT COUNT(*) FROM input",
    "SELECT COUNT(*) FROM input WHERE id > 100",
    "SELECT COUNT(*) FROM input WHERE id >= 0",
    "SELECT SUM(id), AVG(ratio), MIN(cat), MAX(id) FROM input",
    "SELECT cat, COUNT(DISTINCT ok) FROM input GROUP BY cat ORDER BY cat",
    "SELECT MAX(id) FROM input HAVING MAX(id) > 100",
    "SELECT MAX(id) FROM input HAVING MAX(id) > 5",
    "SELECT id FROM input WHERE cat = 'b' ORDER BY id DESC LIMIT 2",
    "SELECT DISTINCT cat FROM input ORDER BY cat",
    "SELECT id + 1 AS next FROM input WHERE id < 3 ORDER BY next",
]


def test_query_parity(paths):
    v1, v2 = paths
    for sql in PARITY_QUERIES:
        t1 = query_file(v1, sql)
        t2 = query_file(v2, sql)
        assert t1.schema == t2.schema, sql
        assert _rows(t1) == _rows(t2), sql


def test_stable_row_order_without_order_by(paths):
    v1, v2 = paths
    sql = "SELECT id FROM input WHERE ratio IS NOT NULL"
    assert _rows(query_file(v1, sql)) == _rows(query_file(v2, sql))


def test_repeated_execution_byte_stable(paths):
    _v1, v2 = paths
    first = _rows(query_file(v2, "SELECT id, cat FROM input WHERE id > 1 ORDER BY id"))
    second = _rows(query_file(v2, "SELECT id, cat FROM input WHERE id > 1 ORDER BY id"))
    assert first == second


# ---------------------------------------------------------------------------
# Statistics pushdown semantics
# ---------------------------------------------------------------------------


def test_all_groups_excluded_plain(paths):
    _v1, v2 = paths
    result = query_file(v2, "SELECT id FROM input WHERE id > 999")
    assert result.row_count == 0
    assert result.column_names == ("id",)
    # same answer as v1
    assert _rows(result) == _rows(query_file(paths[0], "SELECT id FROM input WHERE id > 999"))


def test_all_groups_excluded_aggregates(paths):
    _v1, v2 = paths
    assert query_file(v2, "SELECT COUNT(*) FROM input WHERE id > 999").column("COUNT(*)") == [0]
    assert query_file(v2, "SELECT SUM(id) FROM input WHERE id > 999").column("SUM(id)") == [None]
    assert query_file(v2, "SELECT MAX(cat) FROM input WHERE id < -5").column("MAX(cat)") == [None]
    # HAVING still applies to the single global aggregate row.
    assert (
        query_file(
            v2, "SELECT COUNT(*) FROM input WHERE id > 999 HAVING COUNT(*) > 0"
        ).row_count
        == 0
    )
    assert (
        query_file(v2, "SELECT COUNT(*) FROM input HAVING COUNT(*) > 5").row_count == 1
    )


def test_group_exclusion_by_type(tmp_path):
    schema = Schema(
        [
            ColumnSchema("n", "int64"),
            ColumnSchema("f", "float64"),
            ColumnSchema("s", "utf8"),
            ColumnSchema("b", "bool"),
        ]
    )
    table = Table(
        schema,
        {
            "n": [1, 2, 10, 11, 20, 21],
            "f": [1.0, 2.0, 10.0, 11.0, 20.0, 21.0],
            "s": ["a", "b", "m", "n", "x", "y"],
            "b": [True, True, False, False, True, False],
        },
    )
    path = tmp_path / "p.caef"
    write_partitioned_file(path, table, 2)  # 3 groups of 2

    def selected(sql):
        scan = explain_file(path, sql)["operators"][0]
        return scan["row_groups_selected"]

    # Groups: n=[1,2] | [10,11] | [20,21].
    assert selected("SELECT n FROM input WHERE n >= 15") == 1
    assert selected("SELECT n FROM input WHERE n < 5") == 1
    assert selected("SELECT n FROM input WHERE n = 10") == 1
    assert selected("SELECT n FROM input WHERE n = 3") == 0
    assert selected("SELECT n FROM input WHERE n != 1") == 3
    assert selected("SELECT n FROM input WHERE f >= 15") == 1
    assert selected("SELECT n FROM input WHERE s >= 'o'") == 1  # only x,y group
    assert selected("SELECT n FROM input WHERE s = 'z'") == 0
    assert selected("SELECT n FROM input WHERE b = TRUE") == 2  # groups 0 and 2
    assert selected("SELECT n FROM input WHERE b = FALSE") == 2  # groups 1 and 2


def test_is_null_pushdown(tmp_path):
    schema = Schema([ColumnSchema("s", "utf8", nullable=True)])
    table = Table(
        schema, {"s": ["a", "b", None, None, "c", None]}  # nulls in group 1 and 2
    )
    path = tmp_path / "p.caef"
    write_partitioned_file(path, table, 2)
    isnull = explain_file(path, "SELECT s FROM input WHERE s IS NULL")["operators"][0]
    notnull = explain_file(path, "SELECT s FROM input WHERE s IS NOT NULL")["operators"][0]
    # group 0 has no NULLs; groups 1 (both null) and 2 (one null) survive IS NULL.
    assert isnull["row_groups_selected"] == 2
    # group 1 is entirely NULL and is excluded by IS NOT NULL.
    assert notnull["row_groups_selected"] == 2


def test_or_not_case_arith_column_comparison_not_pushed(tmp_path):
    path = tmp_path / "p.caef"
    write_partitioned_file(path, make_table(), 3)

    def scan(sql):
        return explain_file(path, sql)["operators"][0]

    # OR / NOT / CASE / arithmetic / column-to-column comparisons cannot be proven:
    # every group stays selected even when the predicate excludes every row.
    assert scan("SELECT id FROM input WHERE id > 999 OR id < -999")["row_groups_selected"] == 4
    assert scan("SELECT id FROM input WHERE NOT (id >= 0)")["row_groups_selected"] == 4
    assert scan("SELECT id FROM input WHERE CASE WHEN id > 999 THEN TRUE ELSE FALSE END")[
        "row_groups_selected"
    ] == 4
    assert scan("SELECT id FROM input WHERE id + 0 > 999")["row_groups_selected"] == 4
    # A pushed conjunct decides the group set even when another conjunct is
    # unpushable: id = id stays row-level, but id > 999 alone proves the
    # whole AND impossible for every group.
    assert scan("SELECT id FROM input WHERE id = id AND id > 999")["row_groups_selected"] == 0
    # The unpushable column-to-column comparison does not appear in the plan,
    # and a query that is only id = id pushes nothing at all.
    scan_eq = scan("SELECT id FROM input WHERE id = id")
    assert scan_eq["pushed_condition"] is None
    assert scan_eq["row_groups_selected"] == 4


def test_pushed_condition_shape(tmp_path):
    path = tmp_path / "p.caef"
    write_partitioned_file(path, make_table(), 3)
    scan = explain_file(path, "SELECT id FROM input WHERE id > 2 AND cat = 'a'")["operators"][0]
    condition = scan["pushed_condition"]
    assert condition["kind"] == "logic"
    assert condition["operator"] == "AND"
    leaves = condition["operands"]
    assert all(node["kind"] == "comparison" for node in leaves)
    # No WHERE -> no pushed condition.
    assert explain_file(path, "SELECT id FROM input")["operators"][0]["pushed_condition"] is None


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------


def test_explain_v1_scan_unchanged(paths):
    v1, _v2 = paths
    scan = explain_file(v1, "SELECT id FROM input WHERE id > 1")["operators"][0]
    assert scan == {"operator": "Scan", "source": "input", "required_columns": ["id"]}


def test_explain_v2_scan_fields(paths):
    _v1, v2 = paths
    scan = explain_file(v2, "SELECT id FROM input WHERE id > 7")["operators"][0]
    assert scan["required_columns"] == ["id"]
    assert scan["row_groups_total"] == 4
    # id > 7 matches only rows 8..11 -> groups 2 and 3.
    assert scan["row_groups_selected"] == 2
    assert scan["pushed_condition"]["kind"] == "comparison"


def test_explain_v2_no_where(paths):
    _v1, v2 = paths
    scan = explain_file(v2, "SELECT cat, ok FROM input")["operators"][0]
    assert scan["row_groups_total"] == scan["row_groups_selected"] == 4
    assert scan["pushed_condition"] is None


# ---------------------------------------------------------------------------
# Joins: cross-version parity and projection
# ---------------------------------------------------------------------------


def _join_files(tmp_path, left, right):
    paths = {}
    for name, table in (("l", left), ("r", right)):
        v1 = tmp_path / f"{name}_v1.caef"
        v2 = tmp_path / f"{name}_v2.caef"
        write_file(v1, table)
        write_partitioned_file(v2, table, 2)
        paths[name] = (v1, v2)
    return paths


JOIN_SQLS = [
    "SELECT l.id, r.v FROM l INNER JOIN r ON l.id = r.id ORDER BY l.id, r.v",
    "SELECT l.id, r.v FROM l LEFT JOIN r ON r.id = l.id ORDER BY l.id, r.v",
    "SELECT l.id, r.v FROM l RIGHT JOIN r ON l.id = r.id ORDER BY r.id, l.id",
    "SELECT l.id FROM l FULL OUTER JOIN r ON l.id = r.id ORDER BY l.id, r.id",
    "SELECT COUNT(*) FROM l INNER JOIN r ON l.id = r.id",
]


@pytest.mark.parametrize("strategy", [None, "hash", "sort_merge"])
def test_join_cross_version_parity(tmp_path, strategy):
    left_schema = Schema(
        [ColumnSchema("id", "int64"), ColumnSchema("lv", "utf8", nullable=True)]
    )
    left = Table(left_schema, {"id": [1, 2, 3, 4], "lv": ["a", "b", None, "d"]})
    right_schema = Schema(
        [ColumnSchema("id", "int64"), ColumnSchema("v", "int64", nullable=True)]
    )
    right = Table(right_schema, {"id": [2, 3, 3, 5], "v": [20, 30, 31, 50]})
    files = _join_files(tmp_path, left, right)
    for lv in (0, 1):
        for rv in (0, 1):
            sources = {"l": files["l"][lv], "r": files["r"][rv]}
            for sql in JOIN_SQLS:
                result = query_files(sources, sql, strategy)
                expected = query_files(
                    {"l": files["l"][0], "r": files["r"][0]}, sql, strategy
                )
                assert result.schema == expected.schema
                assert _rows(result) == _rows(expected), (lv, rv, sql)


def test_join_explain_scans_select_all_groups(tmp_path):
    left_schema = Schema([ColumnSchema("id", "int64")])
    left = Table(left_schema, {"id": [1, 2, 3, 4]})
    right_schema = Schema([ColumnSchema("id", "int64"), ColumnSchema("v", "int64")])
    right = Table(right_schema, {"id": [2, 3, 3, 5], "v": [1, 2, 3, 4]})
    files = _join_files(tmp_path, left, right)
    plan = explain_files(
        {"l": files["l"][1], "r": files["r"][1]},
        "SELECT l.id FROM l INNER JOIN r ON l.id = r.id WHERE r.v > 1000",
    )
    scans = [op for op in plan["operators"] if op["operator"] == "Scan"]
    # No stats pushdown through a join: both sources keep all groups even though
    # the WHERE excludes every joined row.
    for scan in scans:
        assert scan["row_groups_total"] == scan["row_groups_selected"] == 2
        assert scan["pushed_condition"] is None


def test_joinless_query_files_uses_pushdown(tmp_path):
    path = tmp_path / "p.caef"
    write_partitioned_file(path, make_table(), 3)
    result = query_files({"input": path}, "SELECT id FROM input WHERE id > 9")
    assert result.column("id") == [10, 11]
    plan = explain_files({"input": path}, "SELECT id FROM input WHERE id > 9")
    scan = plan["operators"][0]
    assert scan["row_groups_total"] == 4 and scan["row_groups_selected"] == 1


# ---------------------------------------------------------------------------
# Export parity
# ---------------------------------------------------------------------------


def test_export_parity(tmp_path, paths):
    v1, v2 = paths
    sql = "SELECT id, cat, ratio FROM input WHERE ratio IS NOT NULL ORDER BY id"
    for fmt in ("csv", "jsonl"):
        d1 = tmp_path / f"v1.{fmt}"
        d2 = tmp_path / f"v2.{fmt}"
        n1 = export_query_file(v1, sql, d1, fmt)
        n2 = export_query_file(v2, sql, d2, fmt)
        assert n1 == n2
        assert d1.read_bytes() == d2.read_bytes()


def test_export_join_cross_version(tmp_path):
    left_schema = Schema([ColumnSchema("id", "int64")])
    left = Table(left_schema, {"id": [1, 2, 3]})
    right_schema = Schema([ColumnSchema("id", "int64"), ColumnSchema("v", "utf8")])
    right = Table(right_schema, {"id": [1, 2, 2], "v": ["a", "b", "c"]})
    files = _join_files(tmp_path, left, right)
    sql = "SELECT l.id, r.v FROM l INNER JOIN r ON l.id = r.id ORDER BY l.id, r.v"
    d1 = tmp_path / "j1.jsonl"
    d2 = tmp_path / "j2.jsonl"
    export_query_files({"l": files["l"][0], "r": files["r"][0]}, sql, d1, "jsonl")
    export_query_files({"l": files["l"][1], "r": files["r"][1]}, sql, d2, "jsonl")
    assert d1.read_bytes() == d2.read_bytes()


# ---------------------------------------------------------------------------
# Corruption: independently checksummed chunks
# ---------------------------------------------------------------------------


def _rebuild(path, dst, mutate_column, mutate_group, byte_offset):
    """Copy ``path`` flipping one chunk byte, fixing only the outer CRCs."""
    blob = bytearray(path.read_bytes())
    header_length = struct.unpack("<I", blob[5:9])[0]
    header = json.loads(bytes(blob[9 : 9 + header_length]).decode())
    data_start = 9 + header_length
    chunk = header["columns"][mutate_column]["chunks"][mutate_group]
    blob[data_start + chunk["offset"] + byte_offset] ^= 0xFF
    data_section = bytes(blob[data_start:-8])
    header["data_crc32"] = zlib.crc32(data_section) & 0xFFFFFFFF
    header_bytes = json.dumps(
        header, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    prefix = blob[:5] + struct.pack("<I", len(header_bytes))
    covered = prefix + header_bytes + data_section
    dst.write_bytes(
        covered
        + struct.pack("<I", zlib.crc32(covered) & 0xFFFFFFFF)
        + b"END1"
    )


def test_corrupted_chunk_only_fails_when_read(paths, tmp_path):
    _v1, v2 = paths
    bad = tmp_path / "bad.caef"
    # Corrupt the id chunk of group 0.
    _rebuild(v2, bad, 0, 0, 4)
    # Excluding group 0 never reads the chunk; the query still succeeds.
    assert query_file(bad, "SELECT id FROM input WHERE id >= 3").column("id") == list(
        range(3, N)
    )
    # Projecting another column never touches the id chunk either.
    assert query_file(bad, "SELECT cat FROM input").row_count == N
    # Reading the chunk, full file or a scan selecting group 0, must fail.
    with pytest.raises(ColumnarFormatError):
        read_file(bad)
    with pytest.raises(ColumnarFormatError):
        query_file(bad, "SELECT id FROM input")
    with pytest.raises(ColumnarFormatError):
        read_file(bad, row_groups=[0])


def test_unreferenced_column_chunk_never_decoded(paths, tmp_path):
    _v1, v2 = paths
    bad = tmp_path / "bad.caef"
    # Corrupt every chunk of the cat column.
    for g in range(4):
        _rebuild(v2 if g == 0 else bad, bad, 1, g, 2)
    assert query_file(bad, "SELECT id FROM input").column("id") == ROWS["id"]
    assert query_file(bad, "SELECT COUNT(*) FROM input").column("COUNT(*)") == [N]
    with pytest.raises(ColumnarFormatError):
        query_file(bad, "SELECT cat FROM input")


def test_chunk_stats_mismatch(paths, tmp_path):
    _v1, v2 = paths
    blob = bytearray(v2.read_bytes())
    header_length = struct.unpack("<I", blob[5:9])[0]
    header = json.loads(bytes(blob[9 : 9 + header_length]).decode())
    # Lie about the id min of group 0 (actual ids 0,1,2): the header stays
    # internally consistent (min <= max) but contradicts the decoded values.
    header["columns"][0]["chunks"][0]["min"] = 1
    data_start = 9 + header_length
    data_section = bytes(blob[data_start:-8])
    header_bytes = json.dumps(
        header, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    prefix = blob[:5] + struct.pack("<I", len(header_bytes))
    covered = prefix + header_bytes + data_section
    bad = tmp_path / "lied.caef"
    bad.write_bytes(
        covered
        + struct.pack("<I", zlib.crc32(covered) & 0xFFFFFFFF)
        + b"END1"
    )
    # Metadata-only reads do not validate value-level stats ...
    assert inspect_file(bad)["row_count"] == N
    # ... but decoding the group must surface the stats inconsistency.
    with pytest.raises(ColumnarFormatError):
        query_file(bad, "SELECT id FROM input")


def test_declared_size_mismatch(paths):
    _v1, v2 = paths
    blob = bytearray(v2.read_bytes())
    v2.write_bytes(bytes(blob[:-1]))  # truncated
    with pytest.raises(ColumnarFormatError):
        read_file(v2)
    with pytest.raises(ColumnarFormatError):
        inspect_file(v2)
    with pytest.raises(ColumnarFormatError):
        inspect_row_groups(v2)


def test_os_error_preserved(tmp_path):
    missing = tmp_path / "nope.caef"
    with pytest.raises(FileNotFoundError):
        query_file(missing, "SELECT * FROM input")
    with pytest.raises(FileNotFoundError):
        inspect_row_groups(missing)
