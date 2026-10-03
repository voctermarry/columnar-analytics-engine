"""Tests for deterministic chained equi-joins across three or more tables."""

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
    explain_files,
    export_query_files,
    query_file,
    query_files,
    write_file,
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
    "aid": [1, 2, 3, 4],
    "ka": [10, 20, 10, None],
    "va": ["a", "b", "c", "d"],
}

B_SCHEMA = Schema(
    [
        ColumnSchema("bid", "int64"),
        ColumnSchema("kb", "int64", nullable=True),
        ColumnSchema("jb", "int64"),
    ]
)
B_DATA = {
    "bid": [100, 200, 300],
    "kb": [10, 10, 30],
    "jb": [1, 2, 3],
}

C_SCHEMA = Schema(
    [
        ColumnSchema("cid", "int64"),
        ColumnSchema("jc", "int64", nullable=True),
        ColumnSchema("tc", "utf8"),
    ]
)
C_DATA = {
    "cid": [7, 8, 9],
    "jc": [2, 2, None],
    "tc": ["x", "y", "z"],
}


@pytest.fixture()
def paths(tmp_path):
    a = tmp_path / "a.caef"
    b = tmp_path / "b.caef"
    c = tmp_path / "c.caef"
    write_file(a, Table(A_SCHEMA, A_DATA))
    write_file(b, Table(B_SCHEMA, B_DATA))
    write_file(c, Table(C_SCHEMA, C_DATA))
    return {"a": a, "b": b, "c": c}


def joined_rows(result):
    return [
        [result._columns[col][i] for col in range(len(result.schema.columns))]
        for i in range(result.row_count)
    ]


def schema_dicts(table):
    return [
        {"name": col.name, "type": col.type, "nullable": col.nullable}
        for col in table.schema.columns
    ]


INNER_CHAIN = (
    "SELECT * FROM a "
    "INNER JOIN b ON a.ka = b.kb "
    "INNER JOIN c ON b.jb = c.jc"
)

# a-b matches: a1/a3 (ka 10) -> b100(j1),b200(j2); the b100 legs find no c
# match (jc values are 2,2,NULL), so only the b200 legs survive the second
# INNER, each expanding over c7/c8 in c-file order.
INNER_CHAIN_ROWS = [
    [1, 10, "a", 200, 10, 2, 7, 2, "x"],
    [1, 10, "a", 200, 10, 2, 8, 2, "y"],
    [3, 10, "c", 200, 10, 2, 7, 2, "x"],
    [3, 10, "c", 200, 10, 2, 8, 2, "y"],
]


# ---------------------------------------------------------------------------
# Star expansion, ordering and strategies
# ---------------------------------------------------------------------------


def test_inner_chain_star_order_and_rows(paths):
    result = query_files(paths, INNER_CHAIN)
    assert result.column_names == (
        "a.aid", "a.ka", "a.va",
        "b.bid", "b.kb", "b.jb",
        "c.cid", "c.jc", "c.tc",
    )
    assert joined_rows(result) == INNER_CHAIN_ROWS
    # Both INNER steps preserve the files' declared nullability.
    assert [col.nullable for col in result.schema.columns] == [
        False, True, True,
        False, True, False,
        False, True, False,
    ]


@pytest.mark.parametrize("strategy", [None, "hash", "sort_merge"])
def test_inner_chain_strategies_identical(paths, strategy):
    result = query_files(paths, INNER_CHAIN, strategy)
    assert joined_rows(result) == INNER_CHAIN_ROWS
    assert schema_dicts(result) == schema_dicts(query_files(paths, INNER_CHAIN))


def test_on_sides_exchangeable_at_every_step(paths):
    swapped = (
        "SELECT * FROM a "
        "INNER JOIN b ON b.kb = a.ka "
        "INNER JOIN c ON c.jc = b.jb"
    )
    normal = query_files(paths, INNER_CHAIN)
    result = query_files(paths, swapped)
    assert result.column_names == normal.column_names
    assert joined_rows(result) == joined_rows(normal)


def test_new_table_may_connect_to_any_earlier_table(paths):
    # The second step joins c directly to the FROM table a, skipping b.
    sql = (
        "SELECT a.aid, b.bid, c.cid FROM a "
        "INNER JOIN b ON a.aid = b.jb "
        "INNER JOIN c ON c.cid = a.aid"
    )
    result = query_files(paths, sql)
    # a-b: a1-b100(j1), a2-b200(j2), a3-b300(j3); c joins a.aid: c7/c8
    # have no aid partner, only c9=aid? cids are 7,8,9 so none match -> empty.
    assert result.row_count == 0
    sql2 = (
        "SELECT a.aid, b.bid, c.cid FROM a "
        "INNER JOIN b ON a.aid = b.jb "
        "INNER JOIN c ON a.aid = c.tc"  # wrong type caught at the second step
    )
    with pytest.raises(QueryValidationError):
        query_files(paths, sql2)


# ---------------------------------------------------------------------------
# Outer joins across the chain, per-step nullability
# ---------------------------------------------------------------------------


def test_left_chain_rows_and_nullability(paths):
    sql = (
        "SELECT * FROM a "
        "LEFT JOIN b ON a.ka = b.kb "
        "LEFT JOIN c ON b.jb = c.jc"
    )
    result = query_files(paths, sql)
    assert joined_rows(result) == [
        [1, 10, "a", 100, 10, 1, None, None, None],
        [1, 10, "a", 200, 10, 2, 7, 2, "x"],
        [1, 10, "a", 200, 10, 2, 8, 2, "y"],
        [2, 20, "b", None, None, None, None, None, None],
        [3, 10, "c", 100, 10, 1, None, None, None],
        [3, 10, "c", 200, 10, 2, 7, 2, "x"],
        [3, 10, "c", 200, 10, 2, 8, 2, "y"],
        [4, None, "d", None, None, None, None, None, None],
    ]
    # a keeps file nullability; every LEFT step makes the newly added table's
    # result columns nullable.
    assert [col.nullable for col in result.schema.columns] == [
        False, True, True,
        True, True, True,
        True, True, True,
    ]


def test_inner_then_right_pads_whole_intermediate(paths):
    sql = (
        "SELECT * FROM a "
        "INNER JOIN b ON a.ka = b.kb "
        "RIGHT JOIN c ON b.jb = c.jc"
    )
    result = query_files(paths, sql)
    # New-file (c) order; c7/c8 match the b200 legs expanding in current
    # intermediate row order (a1 before a3), c9's NULL key pads the whole
    # intermediate side.
    assert joined_rows(result) == [
        [1, 10, "a", 200, 10, 2, 7, 2, "x"],
        [3, 10, "c", 200, 10, 2, 7, 2, "x"],
        [1, 10, "a", 200, 10, 2, 8, 2, "y"],
        [3, 10, "c", 200, 10, 2, 8, 2, "y"],
        [None, None, None, None, None, None, 9, None, "z"],
    ]
    # The RIGHT step forces every earlier table's columns nullable; c keeps
    # its file-declared nullability.
    assert [col.nullable for col in result.schema.columns] == [
        True, True, True,
        True, True, True,
        False, True, False,
    ]


def test_full_chain_appends_unmatched_new_rows(paths):
    sql = (
        "SELECT * FROM a "
        "FULL OUTER JOIN b ON a.ka = b.kb "
        "FULL OUTER JOIN c ON b.jb = c.jc"
    )
    result = query_files(paths, sql)
    assert joined_rows(result) == [
        [1, 10, "a", 100, 10, 1, None, None, None],
        [1, 10, "a", 200, 10, 2, 7, 2, "x"],
        [1, 10, "a", 200, 10, 2, 8, 2, "y"],
        [2, 20, "b", None, None, None, None, None, None],
        [3, 10, "c", 100, 10, 1, None, None, None],
        [3, 10, "c", 200, 10, 2, 7, 2, "x"],
        [3, 10, "c", 200, 10, 2, 8, 2, "y"],
        [4, None, "d", None, None, None, None, None, None],
        [None, None, None, 300, 30, 3, None, None, None],
        [None, None, None, None, None, None, 9, None, "z"],
    ]
    assert all(col.nullable for col in result.schema.columns)


def test_mixed_kind_chain_strategies_and_repeats_byte_stable(paths):
    sql = (
        "SELECT a.aid, b.bid, c.cid FROM a "
        "LEFT JOIN b ON b.kb = a.ka "
        "INNER JOIN c ON c.jc = b.jb"
    )
    default = joined_rows(query_files(paths, sql))
    assert default == [
        [1, 200, 7],
        [1, 200, 8],
        [3, 200, 7],
        [3, 200, 8],
    ]
    for strategy in ("hash", "sort_merge"):
        for _ in range(2):
            assert joined_rows(query_files(paths, sql, strategy)) == default


def test_four_table_chain_smoke(tmp_path):
    def make(name, schema, data):
        path = tmp_path / name
        write_file(path, Table(schema, data))
        return path

    t1 = make("t1.caef", Schema([ColumnSchema("k", "int64")]), {"k": [1]})
    t2 = make("t2.caef", Schema([ColumnSchema("k", "int64"), ColumnSchema("j", "int64")]),
              {"k": [1], "j": [2]})
    t3 = make("t3.caef", Schema([ColumnSchema("j", "int64"), ColumnSchema("m", "int64")]),
              {"j": [2], "m": [3]})
    t4 = make("t4.caef", Schema([ColumnSchema("m", "int64"), ColumnSchema("z", "utf8")]),
              {"m": [3], "z": ["deep"]})
    sources = {"t1": t1, "t2": t2, "t3": t3, "t4": t4}
    sql = (
        "SELECT t4.z FROM t1 "
        "INNER JOIN t2 ON t1.k = t2.k "
        "INNER JOIN t3 ON t2.j = t3.j "
        "INNER JOIN t4 ON t3.m = t4.m"
    )
    for strategy in (None, "hash", "sort_merge"):
        result = query_files(sources, sql, strategy)
        assert result.column("t4.z") == ["deep"]


# ---------------------------------------------------------------------------
# Post-join clauses run after every step
# ---------------------------------------------------------------------------


def test_where_group_having_distinct_sort_limit_after_chain(paths):
    grouped = query_files(
        paths,
        "SELECT a.va, COUNT(*), SUM(c.cid) FROM a "
        "INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc "
        "GROUP BY a.va HAVING COUNT(*) >= 2 ORDER BY a.va",
    )
    assert grouped.column_names == ("a.va", "COUNT(*)", "SUM(c.cid)")
    assert joined_rows(grouped) == [["a", 2, 15], ["c", 2, 15]]

    distinct = query_files(
        paths,
        "SELECT DISTINCT b.jb, c.jc FROM a "
        "INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc "
        "ORDER BY b.jb LIMIT 1",
    )
    assert joined_rows(distinct) == [[2, 2]]

    filtered = query_files(
        paths,
        "SELECT a.aid, c.cid FROM a "
        "INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc "
        "WHERE c.tc = 'y' ORDER BY a.aid DESC LIMIT 1",
    )
    assert joined_rows(filtered) == [[3, 8]]


def test_scalar_and_case_expressions_over_chain(paths):
    result = query_files(
        paths,
        "SELECT a.aid + c.cid AS s, "
        "CASE WHEN b.jb = c.jc THEN 'match' ELSE 'no' END AS hit "
        "FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc "
        "ORDER BY s, hit",
    )
    assert result.column_names == ("s", "hit")
    assert joined_rows(result) == [
        [8, "match"], [9, "match"], [10, "match"], [11, "match"]
    ]
    cols = {col.name: col for col in result.schema.columns}
    assert cols["s"].type == "int64"
    assert cols["s"].nullable is False


# ---------------------------------------------------------------------------
# Explain
# ---------------------------------------------------------------------------


def join_ops(plan):
    return [op for op in plan["operators"] if op["operator"] == "Join"]


def test_explain_chain_sources_scans_and_joins_in_order(paths):
    plan = explain_files(paths, INNER_CHAIN)
    assert [s["name"] for s in plan["sources"]] == ["a", "b", "c"]
    kinds = [op["operator"] for op in plan["operators"]]
    assert kinds == ["Scan", "Scan", "Scan", "Join", "Join", "Project"]
    scans = [op for op in plan["operators"] if op["operator"] == "Scan"]
    # SELECT * projects every schema column in FROM/JOIN order.
    assert scans == [
        {"operator": "Scan", "source": "a", "required_columns": ["aid", "ka", "va"]},
        {"operator": "Scan", "source": "b", "required_columns": ["bid", "kb", "jb"]},
        {"operator": "Scan", "source": "c", "required_columns": ["cid", "jc", "tc"]},
    ]
    assert join_ops(plan) == [
        {
            "operator": "Join",
            "type": "INNER",
            "left": {"table": "a", "column": "ka"},
            "right": {"table": "b", "column": "kb"},
        },
        {
            "operator": "Join",
            "type": "INNER",
            "left": {"table": "b", "column": "jb"},
            "right": {"table": "c", "column": "jc"},
        },
    ]


def test_explain_normalizes_on_direction_and_carries_strategy(paths):
    sql = (
        "SELECT a.aid FROM a "
        "INNER JOIN b ON b.kb = a.ka "
        "LEFT JOIN c ON c.jc = b.jb"
    )
    plain = explain_files(paths, sql)
    for op in join_ops(plain):
        assert "strategy" not in op
    assert join_ops(plain)[0]["left"] == {"table": "a", "column": "ka"}
    assert join_ops(plain)[0]["right"] == {"table": "b", "column": "kb"}
    assert join_ops(plain)[1]["left"] == {"table": "b", "column": "jb"}
    assert join_ops(plain)[1]["right"] == {"table": "c", "column": "jc"}
    for strategy, label in (("hash", "HASH"), ("sort_merge", "SORT_MERGE")):
        planned = explain_files(paths, sql, strategy)
        assert [op.get("strategy") for op in join_ops(planned)] == [label, label]


def test_explain_required_columns_attributed_per_source(paths):
    plan = explain_files(
        paths,
        "SELECT a.va, c.tc, COUNT(*) FROM a "
        "INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc "
        "WHERE a.aid > 0 GROUP BY a.va, c.tc ORDER BY a.va",
    )
    scans = {op["source"]: op["required_columns"]
             for op in plan["operators"] if op["operator"] == "Scan"}
    assert scans == {
        "a": ["aid", "ka", "va"],
        "b": ["kb", "jb"],
        "c": ["jc", "tc"],
    }


def test_explain_chain_output_matches_query_schema(paths):
    sql = (
        "SELECT * FROM a "
        "LEFT JOIN b ON a.ka = b.kb "
        "FULL OUTER JOIN c ON b.jb = c.jc"
    )
    plan = explain_files(paths, sql)
    result = query_files(paths, sql)
    assert plan["output"] == schema_dicts(result)


def test_explain_unreferenced_sources_never_opened(paths, tmp_path):
    sources = dict(paths)
    sources["ghost"] = tmp_path / "does-not-exist.caef"
    plan = explain_files(sources, INNER_CHAIN)
    assert [s["name"] for s in plan["sources"]] == ["a", "b", "c"]
    result = query_files(sources, INNER_CHAIN)
    assert result.row_count == 4


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


SYNTAX_ERRORS = [
    # broken join keyword in a later step
    "SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER c ON b.jb = c.jc",
    "SELECT * FROM a INNER JOIN b ON a.ka = b.kb LEFT c ON b.jb = c.jc",
    # missing ON / incomplete ON in a later step
    "SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c",
    "SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb",
    # non-equality / compound ON in a later step
    "SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb > c.jc",
    "SELECT * FROM a INNER JOIN b ON a.ka = b.kb "
    "INNER JOIN c ON b.jb = c.jc AND a.aid = 1",
    # FULL without OUTER in a later step
    "SELECT * FROM a INNER JOIN b ON a.ka = b.kb FULL JOIN c ON b.jb = c.jc",
    # alias in a later step
    "SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c x ON b.jb = c.jc",
]


@pytest.mark.parametrize("sql", SYNTAX_ERRORS)
def test_chain_syntax_errors_before_file_access(sql):
    missing = {k: f"/nonexistent/{k}.caef" for k in ("a", "b", "c")}
    with pytest.raises(QuerySyntaxError):
        query_files(missing, sql)
    with pytest.raises(QuerySyntaxError):
        explain_files(missing, sql)


VALIDATION_CASES = [
    # unknown table introduced at a later step
    ("SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN nope ON b.jb = nope.j",
     None),
    # repeated table
    ("SELECT * FROM a INNER JOIN b ON a.ka = b.kb LEFT JOIN b ON b.jb = b.kb", None),
    # ON never references the new table
    ("SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON a.aid = b.bid", None),
    # both ON sides name the new table
    ("SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON c.jc = c.cid", None),
    # unknown key column on the earlier side / new side
    ("SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.missing = c.jc",
     None),
    ("SELECT * FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.missing",
     None),
    # incompatible key types at the second step (int64 vs utf8)
    ("SELECT * FROM a INNER JOIN b ON a.va = b.kb INNER JOIN c ON b.jb = c.jc", None),
    # unqualified column elsewhere in a join query
    ("SELECT aid FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc", None),
    # unknown table qualifier
    ("SELECT z.aid FROM a INNER JOIN b ON a.ka = b.kb INNER JOIN c ON b.jb = c.jc",
     None),
]


@pytest.mark.parametrize("sql,strategy", VALIDATION_CASES)
def test_chain_validation_errors(paths, sql, strategy):
    with pytest.raises(QueryValidationError):
        query_files(paths, sql)
    with pytest.raises(QueryValidationError):
        explain_files(paths, sql)


def test_duplicate_table_validation_before_file_access():
    missing = {k: f"/nonexistent/{k}.caef" for k in ("a", "b", "c")}
    sql = (
        "SELECT * FROM a INNER JOIN b ON a.ka = b.kb "
        "INNER JOIN c ON b.jb = c.jc INNER JOIN a ON c.jc = a.aid"
    )
    with pytest.raises(QueryValidationError):
        query_files(missing, sql)


def test_unknown_table_in_later_step_does_not_open_earlier_files(paths):
    # Parsing/resolution failures must surface before any read: b and c
    # exist, the unknown name never reaches a file open.
    sql = (
        "SELECT * FROM a INNER JOIN b ON a.ka = b.kb "
        "INNER JOIN nope ON b.jb = nope.j"
    )
    with pytest.raises(QueryValidationError):
        query_files(paths, sql)


def test_single_file_entry_rejects_chained_join(paths):
    with pytest.raises(QuerySyntaxError):
        query_file(
            paths["a"],
            "SELECT * FROM input INNER JOIN b ON input.ka = b.kb "
            "INNER JOIN c ON b.jb = c.jc",
        )


def test_malformed_later_source_is_format_error(paths, tmp_path):
    bad = tmp_path / "bad.caef"
    bad.write_bytes(b"garbage")
    sources = {"a": paths["a"], "b": paths["b"], "c": bad}
    with pytest.raises(ColumnarFormatError):
        query_files(sources, INNER_CHAIN)


# ---------------------------------------------------------------------------
# Mixed key types across steps
# ---------------------------------------------------------------------------


def test_int64_float64_mixed_key_in_chain(tmp_path):
    a = tmp_path / "a.caef"
    b = tmp_path / "b.caef"
    c = tmp_path / "c.caef"
    write_file(a, Table(Schema([ColumnSchema("k", "int64")]), {"k": [1, 2]}))
    write_file(b, Table(Schema([ColumnSchema("k", "float64"),
                                ColumnSchema("j", "int64")]),
                        {"k": [2.0, 1.0], "j": [5, 6]}))
    write_file(c, Table(Schema([ColumnSchema("j", "int64")]), {"j": [5]}))
    sources = {"a": a, "b": b, "c": c}
    sql = (
        "SELECT a.k, b.j, c.j FROM a "
        "INNER JOIN b ON a.k = b.k INNER JOIN c ON b.j = c.j"
    )
    expected = [[2, 5, 5]]
    for strategy in (None, "hash", "sort_merge"):
        assert joined_rows(query_files(sources, sql, strategy)) == expected


# ---------------------------------------------------------------------------
# Export and CLI
# ---------------------------------------------------------------------------


def test_export_chain_overlap_protection_and_bytes(paths, tmp_path):
    # The destination aliases a source only introduced at the second step.
    with pytest.raises(ValueError):
        export_query_files(paths, INNER_CHAIN, paths["c"])
    # An unreferenced source may be the destination.
    extra = tmp_path / "extra.caef"
    write_file(extra, Table(Schema([ColumnSchema("x", "int64")]), {"x": [1]}))
    outputs = {}
    sources = dict(paths)
    sources["u"] = extra
    for label, strategy in (("default", None), ("hash", "hash"),
                            ("merge", "sort_merge")):
        dst = tmp_path / f"{label}.csv"
        rows = export_query_files(sources, INNER_CHAIN, dst, "csv", strategy)
        assert rows == 4
        outputs[label] = dst.read_bytes()
    assert outputs["default"] == outputs["hash"] == outputs["merge"]
    # Repeated exports stay byte-identical.
    again = tmp_path / "again.csv"
    export_query_files(sources, INNER_CHAIN, again, "jsonl", "sort_merge")
    once_more = tmp_path / "once-more.jsonl"
    export_query_files(sources, INNER_CHAIN, once_more, "jsonl", "hash")
    assert again.read_bytes() == once_more.read_bytes()


def test_cli_chain_query_explain_export(paths, tmp_path, capsys):
    sources_json = json.dumps({k: str(v) for k, v in paths.items()})

    assert main(["query-files", sources_json, INNER_CHAIN]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [col["name"] for col in payload["columns"]] == [
        "a.aid", "a.ka", "a.va",
        "b.bid", "b.kb", "b.jb",
        "c.cid", "c.jc", "c.tc",
    ]
    assert payload["rows"] == INNER_CHAIN_ROWS

    assert main(["explain-files", sources_json, INNER_CHAIN,
                 "--join-strategy", "sort_merge"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [op["strategy"] for op in join_ops(plan)] == [
        "SORT_MERGE", "SORT_MERGE"
    ]

    dst = tmp_path / "chain.csv"
    assert main(["export-files", sources_json, INNER_CHAIN, str(dst)]) == 0
    assert capsys.readouterr().out == ""
    assert dst.read_text(encoding="utf-8").splitlines()[0] == (
        "a.aid,a.ka,a.va,b.bid,b.kb,b.jb,c.cid,c.jc,c.tc"
    )


def test_cli_chain_duplicate_table_exit_2(paths, capsys):
    sources_json = json.dumps({k: str(v) for k, v in paths.items()})
    sql = (
        "SELECT * FROM a INNER JOIN b ON a.ka = b.kb "
        "INNER JOIN b ON b.jb = b.kb"
    )
    assert main(["query-files", sources_json, sql]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err
