"""Tests for the row-group-partitioned CAEF v2 format."""

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
    export_query_file,
    inspect_file,
    inspect_row_groups,
    query_file,
    query_files,
    read_file,
    write_file,
    write_partitioned_file,
)

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
    "id": list(range(1, 11)),
    "n": [10, None, 30, 40, None, 60, 70, 80, 90, 100],
    "f": [1.5, 2.5, None, -0.5, 10.0, 3.25, 7.5, None, 9.5, 11.5],
    "s": ["a", "b", None, "a", "c", "d", "e", None, "f", "g"],
    "flag": [True, False, True, False, True, False, True, False, True, False],
}


def make_table():
    return Table(SCHEMA, DATA)


@pytest.fixture()
def v1_path(tmp_path):
    path = tmp_path / "t1.caef"
    write_file(path, make_table(), compression="zlib", dictionary_encoding=["s"])
    return path


@pytest.fixture()
def v2_path(tmp_path):
    path = tmp_path / "t2.caef"
    write_partitioned_file(
        path, make_table(), 3, compression="zlib", dictionary_encoding=["s"]
    )
    return path


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compression", ["none", "zlib"])
@pytest.mark.parametrize("dictionary", [[], ["s"]])
@pytest.mark.parametrize("size", [1, 3, 4, 10, 100])
def test_roundtrip(tmp_path, compression, dictionary, size):
    path = tmp_path / "t.caef"
    write_partitioned_file(
        path, make_table(), size, compression=compression, dictionary_encoding=dictionary
    )
    table = read_file(path)
    assert table.schema == SCHEMA
    assert table.row_count == 10
    for name in SCHEMA.names:
        assert table.column(name) == DATA[name]


def test_deterministic_bytes(tmp_path):
    p1, p2 = tmp_path / "a", tmp_path / "b"
    for opts in (
        {"compression": "none"},
        {"compression": "zlib"},
        {"compression": "none", "dictionary_encoding": ["s"]},
        {"compression": "zlib", "dictionary_encoding": ["s"]},
    ):
        write_partitioned_file(p1, make_table(), 3, **opts)
        write_partitioned_file(p2, make_table(), 3, **opts)
        assert p1.read_bytes() == p2.read_bytes()


def test_empty_table_roundtrip(tmp_path):
    schema = Schema([ColumnSchema("a", "int64"), ColumnSchema("b", "utf8")])
    path = tmp_path / "empty.caef"
    write_partitioned_file(path, Table(schema, {"a": [], "b": []}), 4)
    restored = read_file(path)
    assert restored.row_count == 0
    assert inspect_row_groups(path) == []


@pytest.mark.parametrize("bad", [0, -1, -100, 2.5, "3", True, False, None])
def test_invalid_row_group_size(tmp_path, bad):
    target = tmp_path / "x.caef"
    with pytest.raises(ValueError):
        write_partitioned_file(target, make_table(), bad)
    assert not target.exists()


def test_invalid_options_leave_no_file(tmp_path):
    target = tmp_path / "x.caef"
    with pytest.raises(ValueError):
        write_partitioned_file(target, make_table(), 3, compression="gzip")
    with pytest.raises(ValueError):
        write_partitioned_file(target, make_table(), 3, dictionary_encoding=["id"])
    with pytest.raises(ValueError):
        write_partitioned_file(target, make_table(), 3, dictionary_encoding=["nope"])
    with pytest.raises(ValueError):
        write_partitioned_file(target, make_table(), 3, dictionary_encoding=["s", "s"])
    assert not target.exists()


def test_failed_write_preserves_existing(v2_path):
    original = v2_path.read_bytes()
    with pytest.raises(ValueError):
        write_partitioned_file(v2_path, make_table(), 0)
    assert v2_path.read_bytes() == original


def test_non_table_rejected(tmp_path):
    with pytest.raises(TypeError):
        write_partitioned_file(tmp_path / "x.caef", object(), 3)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_inspect_v2(v2_path):
    meta = inspect_file(v2_path)
    assert list(meta) == ["format_version", "row_count", "columns"]
    assert meta["format_version"] == 2
    assert meta["row_count"] == 10
    by_name = {c["name"]: c for c in meta["columns"]}
    assert by_name["n"]["null_count"] == 2
    assert by_name["n"]["min"] == 10
    assert by_name["n"]["max"] == 100
    assert by_name["s"]["min"] == "a"
    assert by_name["s"]["max"] == "g"
    assert by_name["flag"]["min"] is False
    assert by_name["flag"]["max"] is True


def test_inspect_row_groups(v2_path, v1_path):
    groups = inspect_row_groups(v2_path)
    assert [g["row_count"] for g in groups] == [3, 3, 3, 1]
    first = {c["name"]: c for c in groups[0]["columns"]}
    assert [c["name"] for c in groups[0]["columns"]] == list(SCHEMA.names)
    assert first["n"] == {"name": "n", "null_count": 1, "min": 10, "max": 30}
    assert first["s"] == {"name": "s", "null_count": 1, "min": "a", "max": "b"}
    last = {c["name"]: c for c in groups[3]["columns"]}
    assert last["id"] == {"name": "id", "null_count": 0, "min": 10, "max": 10}
    # v1 files have no row groups
    assert inspect_row_groups(v1_path) == []


def test_inspect_v1_unchanged(v1_path):
    meta = inspect_file(v1_path)
    assert meta["format_version"] == 1
    assert meta["row_count"] == 10


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_projection(v2_path):
    table = read_file(v2_path, columns=["s", "id"])
    assert table.column_names == ("s", "id")
    assert table.column("s") == DATA["s"]
    assert table.column("id") == DATA["id"]
    with pytest.raises(KeyError):
        read_file(v2_path, columns=["nope"])
    with pytest.raises(ValueError):
        read_file(v2_path, columns=["id", "id"])


def _corrupt_block(path, group_index, column_name):
    """Flip one byte of a (group, column) block; header stays intact."""
    blob = bytearray(path.read_bytes())
    header_len = struct.unpack("<I", blob[5:9])[0]
    header = json.loads(bytes(blob[9 : 9 + header_len]).decode())
    col_index = [c["name"] for c in header["columns"]].index(column_name)
    block = header["row_groups"][group_index]["columns"][col_index]
    blob[9 + header_len + block["offset"]] ^= 0xFF
    path.write_bytes(blob)


def test_unreferenced_block_not_decoded(v2_path):
    _corrupt_block(v2_path, 0, "f")
    # Queries and projections that never touch "f" still succeed.
    assert query_file(v2_path, "SELECT id FROM input WHERE n > 30").row_count == 6
    assert read_file(v2_path, columns=["id", "n"]).column("id") == DATA["id"]
    # Reading the corrupted column reports the block checksum mismatch.
    with pytest.raises(ColumnarFormatError):
        read_file(v2_path)
    with pytest.raises(ColumnarFormatError):
        read_file(v2_path, columns=["f"])
    with pytest.raises(ColumnarFormatError):
        query_file(v2_path, "SELECT f FROM input")
    # Metadata-only entry points never touch the block bytes.
    assert inspect_file(v2_path)["row_count"] == 10
    assert len(inspect_row_groups(v2_path)) == 4


def test_excluded_row_group_not_decoded(v2_path):
    _corrupt_block(v2_path, 0, "n")  # group 0 holds n in [10, None, 30]
    result = query_file(v2_path, "SELECT id FROM input WHERE n > 50")
    assert result.column("id") == [6, 7, 8, 9, 10]
    with pytest.raises(ColumnarFormatError):
        query_file(v2_path, "SELECT id FROM input WHERE n < 50")


def test_corruption_detection(tmp_path):
    path = tmp_path / "t.caef"
    write_partitioned_file(path, make_table(), 3)
    blob = path.read_bytes()

    path.write_bytes(b"X" + blob[1:])
    with pytest.raises(ColumnarFormatError):
        read_file(path)
    with pytest.raises(ColumnarFormatError):
        inspect_row_groups(path)

    path.write_bytes(blob + b"\x00")
    with pytest.raises(ColumnarFormatError):
        read_file(path)

    path.write_bytes(blob[:-1])
    with pytest.raises(ColumnarFormatError):
        read_file(path)

    path.write_bytes(blob[:-4] + b"XXXX")
    with pytest.raises(ColumnarFormatError):
        read_file(path)

    with pytest.raises(FileNotFoundError):
        read_file(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        inspect_row_groups(tmp_path / "missing")


def test_tampered_stats_rejected(tmp_path):
    path = tmp_path / "t.caef"
    write_partitioned_file(path, make_table(), 3)
    blob = bytearray(path.read_bytes())
    header_len = struct.unpack("<I", blob[5:9])[0]
    header = json.loads(bytes(blob[9 : 9 + header_len]).decode())
    header["row_groups"][0]["columns"][1]["null_count"] = 0  # n has one NULL there
    new_header = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    path.write_bytes(
        blob[:5] + struct.pack("<I", len(new_header)) + new_header + blob[9 + header_len :]
    )
    with pytest.raises(ColumnarFormatError):
        read_file(path)


# ---------------------------------------------------------------------------
# Query equivalence and pushdown
# ---------------------------------------------------------------------------

QUERIES = [
    "SELECT * FROM input",
    "SELECT id, s FROM input WHERE n > 30",
    "SELECT id FROM input WHERE n IS NULL",
    "SELECT id FROM input WHERE n IS NOT NULL AND f < 5.0",
    "SELECT id FROM input WHERE s = 'a' OR n = 70",
    "SELECT id FROM input WHERE NOT flag",
    "SELECT id FROM input WHERE id + 1 > 5",
    "SELECT id FROM input WHERE n > id",
    "SELECT COUNT(*) FROM input",
    "SELECT COUNT(*) FROM input WHERE n > 1000",
    "SELECT SUM(n), AVG(f), MIN(s), MAX(id) FROM input",
    "SELECT flag, COUNT(*), SUM(n) FROM input GROUP BY flag HAVING COUNT(*) > 1",
    "SELECT s, COUNT(DISTINCT n) FROM input GROUP BY s ORDER BY s DESC LIMIT 3",
    "SELECT DISTINCT flag FROM input",
    "SELECT id, n * 2 AS dbl FROM input WHERE id >= 2 AND id <= 4 ORDER BY id DESC LIMIT 2",
    "SELECT COUNT(*) FROM input WHERE n > 1000 HAVING COUNT(*) = 0",
    "SELECT COUNT(*) FROM input WHERE n > 1000 HAVING COUNT(*) = 1",
    "SELECT id FROM input WHERE f >= 2.5 AND f <= 10.0 AND s != 'x' AND flag = TRUE",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_query_equivalence(v1_path, v2_path, sql):
    expected = query_file(v1_path, sql)
    actual = query_file(v2_path, sql)
    assert actual.schema == expected.schema
    assert actual.columns == expected.columns


def test_all_groups_excluded_semantics(v2_path):
    assert query_file(v2_path, "SELECT COUNT(*) FROM input WHERE n > 1000").column(
        "COUNT(*)"
    ) == [0]
    assert query_file(v2_path, "SELECT SUM(n) FROM input WHERE n > 1000").column(
        "SUM(n)"
    ) == [None]
    assert query_file(v2_path, "SELECT id FROM input WHERE n > 1000").row_count == 0
    grouped = query_file(
        v2_path, "SELECT flag, COUNT(*) FROM input GROUP BY flag HAVING flag = TRUE"
    )
    assert grouped.column("flag") == [True]


def test_explain_scan_pushdown_fields(v2_path, v1_path):
    plan = explain_file(v2_path, "SELECT id FROM input WHERE n > 30 AND s IS NOT NULL")
    scan = plan["operators"][0]
    assert scan["operator"] == "Scan"
    assert scan["row_groups_total"] == 4
    assert scan["row_groups_selected"] == 3
    assert scan["pushed_condition"]["kind"] == "logic"
    # The Filter operator still carries the full condition.
    kinds = [op["operator"] for op in plan["operators"]]
    assert "Filter" in kinds

    plan = explain_file(v2_path, "SELECT id FROM input")
    scan = plan["operators"][0]
    assert scan["row_groups_total"] == 4
    assert scan["row_groups_selected"] == 4
    assert scan["pushed_condition"] is None

    # OR conditions are not pushed down.
    plan = explain_file(v2_path, "SELECT id FROM input WHERE n > 30 OR id = 1")
    scan = plan["operators"][0]
    assert scan["row_groups_selected"] == 4
    assert scan["pushed_condition"] is None

    # v1 plans keep the historical Scan shape.
    plan = explain_file(v1_path, "SELECT id FROM input WHERE n > 30")
    scan = plan["operators"][0]
    assert scan == {
        "operator": "Scan",
        "source": "input",
        "required_columns": ["id", "n"],
    }


def test_explain_files_no_join_pushdown(v2_path):
    plan = explain_files({"t": v2_path}, "SELECT id FROM t WHERE n < 40")
    scan = plan["operators"][0]
    assert scan["row_groups_total"] == 4
    assert scan["row_groups_selected"] == 1
    result = query_files({"t": v2_path}, "SELECT id FROM t WHERE n < 40")
    assert result.column("id") == [1, 3]


def test_query_files_no_join_equivalence(v1_path, v2_path):
    sql = "SELECT id FROM t WHERE n >= 40 AND s IS NOT NULL ORDER BY id DESC LIMIT 3"
    assert query_files({"t": v1_path}, sql).columns == query_files(
        {"t": v2_path}, sql
    ).columns


# ---------------------------------------------------------------------------
# Joins and exports
# ---------------------------------------------------------------------------

RIGHT_SCHEMA = Schema(
    [
        ColumnSchema("rid", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("tag", "utf8"),
    ]
)
RIGHT_DATA = {"rid": [100, 200, 300], "k": [1, 2, 100], "tag": ["x", "y", "z"]}

JOIN_QUERIES = [
    "SELECT l.id, r.tag FROM l INNER JOIN r ON l.id = r.k",
    "SELECT l.id, r.tag FROM l LEFT JOIN r ON l.id = r.k WHERE l.n > 5",
    "SELECT l.id, r.tag FROM l RIGHT JOIN r ON l.id = r.k",
    "SELECT l.id, r.tag FROM l FULL OUTER JOIN r ON l.id = r.k",
    "SELECT r.tag, COUNT(*) FROM l INNER JOIN r ON l.id = r.k GROUP BY r.tag ORDER BY r.tag",
    "SELECT l.id, r.tag FROM l INNER JOIN r ON l.id = r.k WHERE l.n > 5 AND r.rid >= 150",
    "SELECT l.id, r.tag FROM l INNER JOIN r ON l.id = r.k WHERE l.n > 5 OR r.rid >= 150",
    "SELECT l.id, r.tag FROM l INNER JOIN r ON l.id = r.k WHERE l.n IS NOT NULL AND r.rid < 250 ORDER BY l.id DESC LIMIT 1",
    "SELECT DISTINCT r.tag FROM l INNER JOIN r ON l.id = r.k WHERE l.id <= 10 AND r.rid != 300",
]


@pytest.fixture()
def join_files(tmp_path, v1_path, v2_path):
    r1 = tmp_path / "r1.caef"
    r2 = tmp_path / "r2.caef"
    write_file(r1, Table(RIGHT_SCHEMA, RIGHT_DATA))
    write_partitioned_file(r2, Table(RIGHT_SCHEMA, RIGHT_DATA), 2, compression="zlib")
    return {"v1": (v1_path, r1), "v2": (v2_path, r2)}


@pytest.mark.parametrize("sql", JOIN_QUERIES)
@pytest.mark.parametrize("strategy", [None, "hash", "sort_merge"])
def test_join_equivalence(join_files, sql, strategy):
    left_v1, right_v1 = join_files["v1"]
    left_v2, right_v2 = join_files["v2"]
    expected = query_files({"l": left_v1, "r": right_v1}, sql, strategy)
    for left, right in (
        (left_v2, right_v2),
        (left_v1, right_v2),
        (left_v2, right_v1),
    ):
        actual = query_files({"l": left, "r": right}, sql, strategy)
        assert actual.schema == expected.schema
        assert actual.columns == expected.columns


def test_join_explain_inner_chain_pushdown_fields(join_files):
    left_v2, right_v2 = join_files["v2"]
    plan = explain_files(
        {"l": left_v2, "r": right_v2},
        "SELECT l.id, r.tag FROM l INNER JOIN r ON l.id = r.k "
        "WHERE l.n > 50 AND r.rid > 250",
    )
    scans = {op["source"]: op for op in plan["operators"] if op["operator"] == "Scan"}
    # l.n > 50 excludes the first left group (n max 30 there).
    assert scans["l"]["row_groups_total"] == 4
    assert scans["l"]["row_groups_selected"] == 3
    assert scans["l"]["pushed_condition"] == {
        "kind": "comparison",
        "operator": ">",
        "operands": [
            {"kind": "column", "name": "l.n"},
            {"kind": "literal", "type": "int64", "value": 50},
        ],
    }
    # r.rid > 250 excludes the first right group (rid max 200 there).
    assert scans["r"]["row_groups_total"] == 2
    assert scans["r"]["row_groups_selected"] == 1
    assert scans["r"]["pushed_condition"] == {
        "kind": "comparison",
        "operator": ">",
        "operands": [
            {"kind": "column", "name": "r.rid"},
            {"kind": "literal", "type": "int64", "value": 250},
        ],
    }
    # The Filter operator still carries the full condition.
    assert "Filter" in [op["operator"] for op in plan["operators"]]


def test_join_explain_pushdown_field_order_and_combination(join_files):
    left_v2, right_v2 = join_files["v2"]
    plan = explain_files(
        {"l": left_v2, "r": right_v2},
        "SELECT l.id FROM l INNER JOIN r ON l.id = r.k "
        "WHERE l.n > 50 AND l.s IS NOT NULL AND r.rid > 1000",
    )
    scans = {op["source"]: op for op in plan["operators"] if op["operator"] == "Scan"}
    # The pushdown fields follow required_columns in a fixed order.
    assert list(scans["l"]) == [
        "operator",
        "source",
        "required_columns",
        "row_groups_total",
        "row_groups_selected",
        "pushed_condition",
    ]
    # Same-source leaves combine as AND in SQL order, keeping qualified names.
    pushed = scans["l"]["pushed_condition"]
    assert pushed["kind"] == "logic"
    assert pushed["operator"] == "AND"
    left_leaf, right_leaf = pushed["operands"]
    assert left_leaf["operands"][0] == {"kind": "column", "name": "l.n"}
    assert right_leaf["kind"] == "is_null"
    assert right_leaf["operator"] == "IS NOT NULL"
    assert right_leaf["operands"][0] == {"kind": "column", "name": "l.s"}
    # r.rid > 1000 excludes every right group.
    assert scans["r"]["row_groups_selected"] == 0
    # The same plan is byte-stable.
    assert json.dumps(plan) == json.dumps(
        explain_files(
            {"l": left_v2, "r": right_v2},
            "SELECT l.id FROM l INNER JOIN r ON l.id = r.k "
            "WHERE l.n > 50 AND l.s IS NOT NULL AND r.rid > 1000",
        )
    )


def test_join_explain_no_qualifying_leaf(join_files):
    left_v2, right_v2 = join_files["v2"]
    # No WHERE at all: every group selected, null pushed conditions.
    plan = explain_files(
        {"l": left_v2, "r": right_v2}, "SELECT l.id FROM l INNER JOIN r ON l.id = r.k"
    )
    scans = {op["source"]: op for op in plan["operators"] if op["operator"] == "Scan"}
    assert scans["l"]["row_groups_total"] == 4
    assert scans["l"]["row_groups_selected"] == 4
    assert scans["l"]["pushed_condition"] is None
    assert scans["r"]["row_groups_total"] == 2
    assert scans["r"]["row_groups_selected"] == 2
    assert scans["r"]["pushed_condition"] is None

    # OR / cross-source comparisons do not participate but do not block the
    # qualifying sibling leaves either.
    plan = explain_files(
        {"l": left_v2, "r": right_v2},
        "SELECT l.id FROM l INNER JOIN r ON l.id = r.k "
        "WHERE (l.n > 1000 OR r.rid = 100) AND l.id = r.rid AND r.rid > 250",
    )
    scans = {op["source"]: op for op in plan["operators"] if op["operator"] == "Scan"}
    assert scans["l"]["row_groups_selected"] == 4
    assert scans["l"]["pushed_condition"] is None
    assert scans["r"]["row_groups_selected"] == 1
    assert scans["r"]["pushed_condition"]["operator"] == ">"


def test_join_explain_mixed_versions(join_files):
    (left_v1, _), (_, right_v2) = join_files["v1"], join_files["v2"]
    plan = explain_files(
        {"l": left_v1, "r": right_v2},
        "SELECT l.id, r.tag FROM l INNER JOIN r ON l.id = r.k WHERE r.rid > 250",
    )
    scans = {op["source"]: op for op in plan["operators"] if op["operator"] == "Scan"}
    # v1 scans keep the historical shape even in an all-INNER chain.
    assert scans["l"] == {
        "operator": "Scan",
        "source": "l",
        "required_columns": ["id"],
    }
    assert scans["r"]["row_groups_total"] == 2
    assert scans["r"]["row_groups_selected"] == 1
    assert scans["r"]["pushed_condition"] is not None


def test_join_explain_keeps_scan_shape(join_files):
    left_v2, right_v2 = join_files["v2"]
    # A chain containing any outer join applies no row-group pushdown.
    for sql in (
        "SELECT l.id FROM l LEFT JOIN r ON l.id = r.k WHERE l.n > 50",
        "SELECT l.id FROM l RIGHT JOIN r ON l.id = r.k WHERE l.n > 50",
        "SELECT l.id FROM l FULL OUTER JOIN r ON l.id = r.k WHERE l.n > 50",
        "SELECT l.id FROM l INNER JOIN r ON l.id = r.k "
        "LEFT JOIN r2 ON l.id = r2.k WHERE l.n > 50",
    ):
        sources = {"l": left_v2, "r": right_v2}
        if "r2" in sql:
            sources["r2"] = right_v2
        plan = explain_files(sources, sql)
        for op in plan["operators"]:
            if op["operator"] == "Scan":
                assert "row_groups_total" not in op
                assert "row_groups_selected" not in op
                assert "pushed_condition" not in op


def test_join_reads_only_referenced_columns(join_files):
    left_v2, right_v2 = join_files["v2"]
    # Corrupt an unreferenced column of the v2 right table; the join query
    # must not decode it.
    _corrupt_block(right_v2, 0, "rid")
    sql = "SELECT l.id, r.tag FROM l INNER JOIN r ON l.id = r.k"
    result = query_files({"l": left_v2, "r": right_v2}, sql)
    assert result.column("r.tag") == ["x", "y"]
    with pytest.raises(ColumnarFormatError):
        query_files({"l": left_v2, "r": right_v2}, "SELECT r.rid FROM l INNER JOIN r ON l.id = r.k")


@pytest.mark.parametrize("strategy", [None, "hash", "sort_merge"])
def test_join_excluded_row_group_not_decoded(join_files, strategy):
    left_v2, right_v2 = join_files["v2"]
    # Group 2 of the left table holds n in [70, 80, 90]; "l.n < 50"
    # excludes it, so its corrupted block is never read.
    _corrupt_block(left_v2, 2, "n")
    sql = "SELECT l.id, r.tag FROM l INNER JOIN r ON l.id = r.k WHERE l.n < 50"
    result = query_files({"l": left_v2, "r": right_v2}, sql, strategy)
    assert result.column("l.id") == [1]
    assert result.column("r.tag") == ["x"]
    # A condition that keeps the corrupted group surfaces the block error.
    with pytest.raises(ColumnarFormatError):
        query_files(
            {"l": left_v2, "r": right_v2},
            "SELECT l.id FROM l INNER JOIN r ON l.id = r.k WHERE l.n > 5",
            strategy,
        )
    # An outer join in the chain disables the pushdown: every group is read.
    with pytest.raises(ColumnarFormatError):
        query_files(
            {"l": left_v2, "r": right_v2},
            "SELECT l.id FROM l LEFT JOIN r ON l.id = r.k WHERE l.n < 50",
            strategy,
        )


@pytest.mark.parametrize("strategy", [None, "hash", "sort_merge"])
def test_join_pushdown_on_each_source(join_files, strategy):
    left_v2, right_v2 = join_files["v2"]
    # Both sources get a pushed leaf; the excluded groups on either side
    # are corrupted and must stay unread.
    _corrupt_block(left_v2, 0, "n")  # n in [10, None, 30]
    _corrupt_block(right_v2, 0, "rid")  # rid in [100, 200]
    sql = (
        "SELECT l.id, r.tag FROM l INNER JOIN r ON l.id = r.k "
        "WHERE l.n > 50 AND r.rid > 250"
    )
    result = query_files({"l": left_v2, "r": right_v2}, sql, strategy)
    assert result.row_count == 0
    # The plan reports the same selected groups the query read.
    plan = explain_files({"l": left_v2, "r": right_v2}, sql, strategy)
    scans = {op["source"]: op for op in plan["operators"] if op["operator"] == "Scan"}
    assert scans["l"]["row_groups_selected"] == 3
    assert scans["r"]["row_groups_selected"] == 1


def test_join_pushdown_mixed_versions(join_files):
    (left_v1, _), (_, right_v2) = join_files["v1"], join_files["v2"]
    # Only the v2 source is pruned; the corrupt excluded block stays unread.
    _corrupt_block(right_v2, 0, "rid")
    sql = (
        "SELECT l.id, r.tag FROM l INNER JOIN r ON l.id = r.k "
        "WHERE l.n < 50 AND r.rid > 250"
    )
    result = query_files({"l": left_v1, "r": right_v2}, sql)
    assert result.row_count == 0
    # The same statement against the all-v1 pair agrees.
    left_v1b, right_v1 = join_files["v1"]
    expected = query_files({"l": left_v1b, "r": right_v1}, sql)
    assert result.columns == expected.columns


@pytest.mark.parametrize("format", ["csv", "jsonl"])
def test_export_equivalence(v1_path, v2_path, tmp_path, format):
    d1 = tmp_path / f"e1.{format}"
    d2 = tmp_path / f"e2.{format}"
    sql = "SELECT id, s FROM input WHERE n >= 40"
    assert export_query_file(v1_path, sql, d1, format) == export_query_file(
        v2_path, sql, d2, format
    )
    assert d1.read_bytes() == d2.read_bytes()
