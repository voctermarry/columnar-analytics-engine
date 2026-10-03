"""Tests for deterministic continuous (multi-step) equi-joins.

A statement may chain any number of INNER / LEFT / RIGHT / FULL OUTER JOIN
steps after FROM; each step introduces one so-far-unused table with a
single swap-symmetric equality ON. The zero- and one-JOIN behaviour is
covered by ``test_join.py``; this module exercises the chain extension.
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
    explain_files,
    export_query_files,
    query_file,
    query_files,
    write_file,
)
from columnar_analytics.cli import main


def joined_rows(result):
    return [
        [result._columns[c][i] for c in range(len(result.schema.columns))]
        for i in range(result.row_count)
    ]


def write_src(tmp_path, name, columns, data):
    path = tmp_path / name
    write_file(path, Table(Schema([ColumnSchema(*c) for c in columns]), data))
    return path


@pytest.fixture()
def sources(tmp_path):
    # a: k joins b; b: k/t, t joins c.  NULLs and duplicate keys appear on
    # every side so the per-step NULL / multiplicity rules are exercised.
    a = write_src(
        tmp_path, "a.caef",
        [("id", "int64"), ("k", "int64", True), ("av", "utf8")],
        {"id": [1, 2, 3, 4], "k": [10, 20, 10, None], "av": ["a", "b", "c", "d"]},
    )
    b = write_src(
        tmp_path, "b.caef",
        [("bid", "int64"), ("k", "int64", True), ("t", "int64", True)],
        {"bid": [10, 11, 12], "k": [10, 10, 30], "t": [100, None, 300]},
    )
    c = write_src(
        tmp_path, "c.caef",
        [("cid", "int64"), ("t", "int64"), ("cv", "utf8")],
        {"cid": [100, 200], "t": [100, 300], "cv": ["x", "y"]},
    )
    return {"a": a, "b": b, "c": c}


# ---------------------------------------------------------------------------
# INNER chains
# ---------------------------------------------------------------------------


def test_inner_inner_star_order(sources):
    sql = "SELECT * FROM a INNER JOIN b ON a.k = b.k INNER JOIN c ON b.t = c.t"
    result = query_files(sources, sql)
    assert result.column_names == (
        "a.id", "a.k", "a.av",
        "b.bid", "b.k", "b.t",
        "c.cid", "c.t", "c.cv",
    )
    # a1/a3 each match b10 (t 100) which matches c100; b11 (NULL t) and
    # b12 (t 300, key k 30 matches no a row) are dropped by the first INNER.
    assert joined_rows(result) == [
        [1, 10, "a", 10, 10, 100, 100, 100, "x"],
        [3, 10, "c", 10, 10, 100, 100, 100, "x"],
    ]
    assert [col.nullable for col in result.schema.columns] == [
        False, True, False,
        False, True, True,
        False, False, False,
    ]


def test_inner_inner_on_sides_swappable(sources):
    sql = "SELECT a.id, b.bid, c.cid FROM a INNER JOIN b ON b.k = a.k INNER JOIN c ON c.t = b.t"
    assert joined_rows(query_files(sources, sql)) == [
        [1, 10, 100],
        [3, 10, 100],
    ]


def test_chain_on_key_can_reference_any_preceding_table(tmp_path):
    # The second step joins the new table c directly to the base a (not the
    # immediately preceding b), while step 1 matches a/b on a separate key.
    a = write_src(tmp_path, "a.caef",
                  [("x", "int64"), ("j", "int64")], {"x": [1], "j": [0]})
    b = write_src(tmp_path, "b.caef",
                  [("y", "int64"), ("j", "int64")], {"y": [7], "j": [0]})
    c = write_src(tmp_path, "c.caef", [("z", "int64")], {"z": [1]})
    src = {"a": a, "b": b, "c": c}
    result = query_files(
        src,
        "SELECT a.x, b.y, c.z FROM a INNER JOIN b ON a.j = b.j "
        "INNER JOIN c ON c.z = a.x",
    )
    assert joined_rows(result) == [[1, 7, 1]]


def test_three_way_full_combinations(tmp_path):
    a = write_src(tmp_path, "a.caef", [("k", "int64")], {"k": [1, 1]})
    b = write_src(tmp_path, "b.caef", [("k", "int64")], {"k": [1, 1]})
    c = write_src(tmp_path, "c.caef", [("k", "int64")], {"k": [1]})
    src = {"a": a, "b": b, "c": c}
    result = query_files(src, "SELECT * FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.k=c.k")
    # 2 x 2 x 1 combinations in current-order / new-file order at each step.
    assert joined_rows(result) == [[1, 1, 1]] * 4
    assert result.row_count == 4


def test_int64_float64_compatible_per_step(tmp_path):
    a = write_src(tmp_path, "a.caef", [("k", "int64")], {"k": [1, 2]})
    b = write_src(tmp_path, "b.caef", [("k", "float64")], {"k": [2.0]})
    c = write_src(tmp_path, "c.caef", [("n", "int64"), ("m", "float64")], {"n": [2], "m": [2.0]})
    src = {"a": a, "b": b, "c": c}
    result = query_files(
        src,
        "SELECT a.k, b.k, c.n FROM a INNER JOIN b ON a.k = b.k INNER JOIN c ON c.n = a.k",
    )
    assert joined_rows(result) == [[2, 2.0, 2]]


# ---------------------------------------------------------------------------
# Outer joins in a chain: ordering and nullable derivation per step
# ---------------------------------------------------------------------------


def test_left_then_inner_keeps_padded_nulls(sources):
    # LEFT first: unmatched a rows (2,4) survive with b padded NULL; the
    # following INNER against c drops those padded rows.
    sql = (
        "SELECT a.id, b.bid, c.cid FROM a LEFT JOIN b ON a.k = b.k "
        "INNER JOIN c ON b.t = c.t"
    )
    assert joined_rows(query_files(sources, sql)) == [
        [1, 10, 100],
        [3, 10, 100],
    ]


def test_inner_then_left_appends_unmatched_intermediate(sources):
    sql = (
        "SELECT a.id, b.bid, c.cid FROM a INNER JOIN b ON a.k = b.k "
        "LEFT JOIN c ON b.t = c.t"
    )
    # INNER result: (1,b10), (1,b11), (3,b10), (3,b11).  c matches t=100
    # only; b11's NULL t never matches and becomes an unmatched row.
    assert joined_rows(query_files(sources, sql)) == [
        [1, 10, 100],
        [1, 11, None],
        [3, 10, 100],
        [3, 11, None],
    ]
    result = query_files(
        sources,
        "SELECT * FROM a INNER JOIN b ON a.k = b.k LEFT JOIN c ON b.t = c.t",
    )
    # Only the c-side columns become nullable in the second step.
    nullability = {col.name: col.nullable for col in result.schema.columns}
    assert nullability["a.id"] is False
    assert nullability["b.t"] is True
    assert nullability["c.cid"] is True
    assert nullability["c.t"] is True


def test_right_join_step_orders_by_new_file(tmp_path):
    a = write_src(tmp_path, "a.caef", [("k", "int64")], {"k": [1, 2]})
    b = write_src(tmp_path, "b.caef", [("k", "int64"), ("t", "int64")], {"k": [1, 1], "t": [9, 8]})
    c = write_src(tmp_path, "c.caef", [("t", "int64")], {"t": [8, 9, 7]})
    src = {"a": a, "b": b, "c": c}
    # Step 1 INNER gives, in a order: (1,b t9),(1,b t8).  Step 2 RIGHT JOIN c
    # follows c file order 8,9,7; each match expands in current (a) order.
    result = query_files(
        src,
        "SELECT a.k, b.t, c.t FROM a INNER JOIN b ON a.k=b.k RIGHT JOIN c ON b.t=c.t",
    )
    assert joined_rows(result) == [
        [1, 8, 8],
        [1, 9, 9],
        [None, None, 7],
    ]
    nullability = {col.name: col.nullable for col in result.schema.columns}
    assert nullability["a.k"] is True
    assert nullability["b.t"] is True
    assert nullability["c.t"] is False


def test_full_then_full_nullability_accumulates(tmp_path):
    a = write_src(tmp_path, "a.caef", [("k", "int64")], {"k": [1]})
    b = write_src(tmp_path, "b.caef", [("k", "int64"), ("t", "int64")], {"k": [2], "t": [1]})
    c = write_src(tmp_path, "c.caef", [("t", "int64")], {"t": [2]})
    src = {"a": a, "b": b, "c": c}
    result = query_files(
        src, "SELECT * FROM a FULL OUTER JOIN b ON a.k=b.k FULL OUTER JOIN c ON b.t=c.t"
    )
    # First FULL (a1,b2 unmatched) then second FULL against c2:
    # current rows (a1,pad),(pad,b2 t1); neither matches c t2, so they stay
    # as unmatched intermediate rows and c2 is appended with a/b padded.
    assert joined_rows(result) == [
        [1, None, None, None],
        [None, 2, 1, None],
        [None, None, None, 2],
    ]
    assert all(col.nullable for col in result.schema.columns)


def test_where_groupby_distinct_order_limit_run_after_chain(sources):
    sql = (
        "SELECT a.k, COUNT(*), SUM(c.cid) FROM a INNER JOIN b ON a.k=b.k "
        "INNER JOIN c ON b.t=c.t WHERE c.cv = 'x' GROUP BY a.k "
        "HAVING COUNT(*) >= 1 ORDER BY a.k DESC LIMIT 5"
    )
    result = query_files(sources, sql)
    assert result.column_names == ("a.k", "COUNT(*)", "SUM(c.cid)")
    assert joined_rows(result) == [[10, 2, 200]]

    distinct = query_files(
        sources,
        "SELECT DISTINCT a.id FROM a INNER JOIN b ON a.k=b.k "
        "INNER JOIN c ON b.t=c.t ORDER BY a.id",
    )
    assert joined_rows(distinct) == [[1], [3]]


def test_scalar_expression_over_chain_columns(sources):
    result = query_files(
        sources,
        "SELECT a.id, b.t + c.cid AS s FROM a INNER JOIN b ON a.k=b.k "
        "INNER JOIN c ON b.t=c.t ORDER BY a.id",
    )
    assert result.column_names == ("a.id", "s")
    assert [row[-1] for row in joined_rows(result)] == [200, 200]


# ---------------------------------------------------------------------------
# Strategy: applies to every step, identical results, byte stability
# ---------------------------------------------------------------------------


CHAIN_SQLS = [
    "SELECT * FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.t=c.t",
    "SELECT * FROM a LEFT JOIN b ON a.k=b.k LEFT JOIN c ON b.t=c.t",
    "SELECT * FROM a INNER JOIN b ON a.k=b.k RIGHT JOIN c ON b.t=c.t",
    "SELECT a.id, b.bid, c.cid FROM a FULL OUTER JOIN b ON a.k=b.k "
    "FULL OUTER JOIN c ON b.t=c.t",
    "SELECT a.id, c.cv FROM a INNER JOIN b ON b.k=a.k INNER JOIN c ON c.t=b.t "
    "WHERE a.id > 1 ORDER BY a.id, c.cid LIMIT 2",
]


@pytest.mark.parametrize("sql", CHAIN_SQLS)
@pytest.mark.parametrize("strategy", ["hash", "sort_merge"])
def test_strategies_match_default(sources, sql, strategy):
    default = query_files(sources, sql)
    chosen = query_files(sources, sql, strategy)
    assert [
        {"name": c.name, "type": c.type, "nullable": c.nullable}
        for c in chosen.schema.columns
    ] == [
        {"name": c.name, "type": c.type, "nullable": c.nullable}
        for c in default.schema.columns
    ]
    assert joined_rows(chosen) == joined_rows(default)


@pytest.mark.parametrize("sql", CHAIN_SQLS)
def test_repeated_runs_byte_stable(sources, sql):
    first = joined_rows(query_files(sources, sql))
    for _ in range(3):
        assert joined_rows(query_files(sources, sql)) == first
        assert joined_rows(query_files(sources, sql, "sort_merge")) == first


@pytest.mark.parametrize("strategy,label", [("hash", "HASH"), ("sort_merge", "SORT_MERGE")])
def test_explain_every_join_carries_strategy(sources, strategy, label):
    sql = "SELECT a.id FROM a INNER JOIN b ON a.k=b.k LEFT JOIN c ON b.t=c.t"
    plan = explain_files(sources, sql, strategy)
    joins = [op for op in plan["operators"] if op["operator"] == "Join"]
    assert [op["strategy"] for op in joins] == [label, label]
    default = explain_files(sources, sql)
    assert all("strategy" not in op for op in default["operators"] if op["operator"] == "Join")
    # Stripping the field reproduces the default plan exactly.
    for op in joins:
        op.pop("strategy")
    assert joins == [op for op in default["operators"] if op["operator"] == "Join"]


def test_explain_chain_order_sources_scans_joins(sources):
    sql = (
        "SELECT a.id, c.cv, COUNT(*) FROM a INNER JOIN b ON b.k=a.k "
        "INNER JOIN c ON c.t=b.t WHERE a.id > 0 GROUP BY a.id, c.cv"
    )
    plan = explain_files(sources, sql)
    assert [s["name"] for s in plan["sources"]] == ["a", "b", "c"]
    kinds = [op["operator"] for op in plan["operators"]]
    assert kinds == [
        "Scan", "Scan", "Scan",
        "Join", "Join",
        "Filter", "Aggregate", "Project",
    ]
    scans = {op["source"]: op["required_columns"] for op in plan["operators"] if op["operator"] == "Scan"}
    # a: id(project/where) + k(join); b: k(join) + t(join); c: t(join) + cv(group/project)
    assert scans == {"a": ["id", "k"], "b": ["k", "t"], "c": ["t", "cv"]}
    joins = [op for op in plan["operators"] if op["operator"] == "Join"]
    assert joins == [
        {
            "operator": "Join",
            "type": "INNER",
            "left": {"table": "a", "column": "k"},
            "right": {"table": "b", "column": "k"},
        },
        {
            "operator": "Join",
            "type": "INNER",
            "left": {"table": "b", "column": "t"},
            "right": {"table": "c", "column": "t"},
        },
    ]


def test_explain_on_keys_scanned_even_when_not_projected(sources):
    plan = explain_files(sources, "SELECT COUNT(*) FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.t=c.t")
    scans = {op["source"]: op["required_columns"] for op in plan["operators"] if op["operator"] == "Scan"}
    assert scans == {"a": ["k"], "b": ["k", "t"], "c": ["t"]}


def test_explain_swapped_on_reports_oriented_keys(sources):
    plan = explain_files(sources, "SELECT a.id FROM a INNER JOIN b ON b.k=a.k INNER JOIN c ON c.t=b.t")
    joins = [op for op in plan["operators"] if op["operator"] == "Join"]
    assert joins[0]["left"] == {"table": "a", "column": "k"}
    assert joins[0]["right"] == {"table": "b", "column": "k"}
    assert joins[1]["left"] == {"table": "b", "column": "t"}
    assert joins[1]["right"] == {"table": "c", "column": "t"}


def test_explain_output_matches_query_schema(sources):
    sql = "SELECT * FROM a LEFT JOIN b ON a.k=b.k FULL OUTER JOIN c ON b.t=c.t"
    plan = explain_files(sources, sql)
    result = query_files(sources, sql)
    assert plan["output"] == [
        {"name": col.name, "type": col.type, "nullable": col.nullable}
        for col in result.schema.columns
    ]


# ---------------------------------------------------------------------------
# Unreferenced sources and same-path export protection
# ---------------------------------------------------------------------------


def test_unreferenced_source_never_opened(sources, tmp_path):
    ghost = tmp_path / "ghost.caef"
    sources = dict(sources, g=ghost)
    result = query_files(sources, "SELECT a.id FROM a INNER JOIN b ON a.k=b.k WHERE a.id=1")
    # b carries two rows with k=10 (bid 10 and 11), both match a id 1.
    assert joined_rows(result) == [[1], [1]]
    plan = explain_files(sources, "SELECT a.id FROM a INNER JOIN b ON a.k=b.k")
    assert [s["name"] for s in plan["sources"]] == ["a", "b"]


def test_export_overlap_checks_every_referenced_source(sources, tmp_path):
    dst = tmp_path / "out.csv"
    sql = "SELECT a.id FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.t=c.t"
    for key in ("a", "b", "c"):
        with pytest.raises(ValueError):
            export_query_files(sources, sql, sources[key])
    assert not dst.exists()
    # An unreferenced source is not protected.
    export_query_files(
        dict(sources, g=tmp_path / "g.caef"),
        "SELECT a.id FROM a INNER JOIN b ON a.k=b.k",
        dst,
    )
    # a k=10 (ids 1,3) each matches b's two k=10 rows (bid 10, 11).
    assert dst.read_bytes() == b"a.id\n1\n1\n3\n3\n"


def test_export_chains_byte_identical(sources, tmp_path):
    sql = "SELECT a.id, c.cv FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.t=c.t ORDER BY a.id"
    outputs = {}
    for label, strat in (("default", None), ("hash", "hash"), ("merge", "sort_merge")):
        dst = tmp_path / f"{label}.jsonl"
        export_query_files(sources, sql, dst, "jsonl", strat)
        outputs[label] = dst.read_bytes()
    assert outputs["default"] == outputs["hash"] == outputs["merge"]


# ---------------------------------------------------------------------------
# Errors: syntax (before any file access) and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM a INNER JOIN b ON a.k=b.k INNER c ON b.t=c.t",   # missing JOIN
        "SELECT * FROM a INNER JOIN b ON a.k=b.k LEFT",                 # truncated
        "SELECT * FROM a INNER JOIN b ON a.k=b.k INNER JOIN c",         # missing ON
        "SELECT * FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.t",  # incomplete ON
        "SELECT * FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.t > c.t",  # non-equality
        "SELECT * FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.t=c.t AND a.id=1",  # compound
        "SELECT * FROM a INNER JOIN b ON a.k=b.k FULL JOIN c ON b.t=c.t",  # FULL without OUTER
        "SELECT * FROM a INNER JOIN b ON a.k=b.k JOIN c ON b.t=c.t",    # bare JOIN
        "SELECT * FROM a INNER JOIN b x ON a.k=b.k INNER JOIN c ON b.t=c.t",  # alias
    ],
)
def test_chain_syntax_errors_before_file_access(sql):
    missing = {t: f"/nonexistent/{t}.caef" for t in ("a", "b", "c")}
    with pytest.raises(QuerySyntaxError):
        query_files(missing, sql)
    with pytest.raises(QuerySyntaxError):
        explain_files(missing, sql)


@pytest.mark.parametrize(
    "sql",
    [
        # duplicate table at any position
        "SELECT * FROM a INNER JOIN b ON a.k=b.k INNER JOIN b ON b.t=b.t",
        "SELECT * FROM a INNER JOIN a ON a.k=a.k",
        # ON does not connect the new table to a preceding one
        "SELECT * FROM a INNER JOIN b ON a.k=a.id INNER JOIN c ON b.t=c.t",
        "SELECT * FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON a.k=b.k",
        # ON references a table introduced only later
        "SELECT * FROM a INNER JOIN b ON a.k=c.k INNER JOIN c ON b.t=c.t",
        # unknown table / column / unqualified reference
        "SELECT * FROM a INNER JOIN z ON a.k=z.k",
        "SELECT * FROM a INNER JOIN b ON a.nope=b.k INNER JOIN c ON b.t=c.t",
        "SELECT a.id FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.t=c.t WHERE id=1",
    ],
)
def test_chain_validation_errors(sources, sql):
    with pytest.raises(QueryValidationError):
        query_files(sources, sql)
    with pytest.raises(QueryValidationError):
        explain_files(sources, sql)


def test_chain_incompatible_key_type_second_step(tmp_path):
    a = write_src(tmp_path, "a.caef", [("k", "int64")], {"k": [1]})
    b = write_src(tmp_path, "b.caef", [("k", "int64"), ("t", "int64")], {"k": [1], "t": [1]})
    c = write_src(tmp_path, "c.caef", [("t", "utf8")], {"t": ["1"]})
    with pytest.raises(QueryValidationError):
        query_files(
            {"a": a, "b": b, "c": c},
            "SELECT * FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.t=c.t",
        )


def test_chain_duplicate_result_columns(sources):
    # Same table.column projected twice is rejected exactly as with one join.
    with pytest.raises(QueryValidationError):
        query_files(
            sources,
            "SELECT a.id, a.id FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.t=c.t",
        )


def test_single_file_entry_still_rejects_chains(sources):
    with pytest.raises(QuerySyntaxError):
        query_file(sources["a"], "SELECT * FROM input INNER JOIN b ON input.k=b.k")
    with pytest.raises(QuerySyntaxError):
        query_file(
            sources["a"],
            "SELECT * FROM input INNER JOIN b ON input.k=b.k INNER JOIN c ON b.t=c.t",
        )


# ---------------------------------------------------------------------------
# File errors keep their classification
# ---------------------------------------------------------------------------


def test_corrupt_intermediate_source_format_error(sources, tmp_path):
    bad = tmp_path / "bad.caef"
    bad.write_bytes(b"garbage")
    broken = dict(sources, b=bad)
    sql = "SELECT * FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.t=c.t"
    with pytest.raises(ColumnarFormatError):
        query_files(broken, sql)
    # A corrupt source that is never referenced stays untouched.
    extra = dict(sources, g=bad)
    result = query_files(extra, "SELECT a.id FROM a")
    assert joined_rows(result) == [[1], [2], [3], [4]]


def test_missing_source_oserror(sources):
    sql = "SELECT * FROM a INNER JOIN b ON a.k=b.k"
    broken = dict(sources, b="/nonexistent/b-missing.caef")
    with pytest.raises(OSError):
        query_files(broken, sql)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_query_files_chain(sources, capsys):
    sources_json = json.dumps({k: str(v) for k, v in sources.items()})
    sql = "SELECT a.id, c.cid FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.t=c.t ORDER BY a.id"
    assert main(["query-files", sources_json, sql]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert [col["name"] for col in payload["columns"]] == ["a.id", "c.cid"]
    assert payload["rows"] == [[1, 100], [3, 100]]
    # Byte stability across repeated invocations.
    assert main(["query-files", sources_json, sql]) == 0
    assert capsys.readouterr().out == out


def test_cli_explain_files_chain_strategy(sources, capsys):
    sources_json = json.dumps({k: str(v) for k, v in sources.items()})
    sql = "SELECT a.id FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.t=c.t"
    assert main(["explain-files", sources_json, sql, "--join-strategy", "hash"]) == 0
    plan = json.loads(capsys.readouterr().out)
    joins = [op for op in plan["operators"] if op["operator"] == "Join"]
    assert [op["strategy"] for op in joins] == ["HASH", "HASH"]


def test_cli_export_files_chain(sources, tmp_path, capsys):
    sources_json = json.dumps({k: str(v) for k, v in sources.items()})
    dst = tmp_path / "out.csv"
    sql = "SELECT a.id FROM a INNER JOIN b ON a.k=b.k INNER JOIN c ON b.t=c.t ORDER BY a.id"
    assert main(["export-files", sources_json, sql, str(dst)]) == 0
    assert capsys.readouterr().out == ""
    assert dst.read_bytes() == b"a.id\n1\n3\n"
