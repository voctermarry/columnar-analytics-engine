"""Tests for the optional ``join_strategy`` argument (hash / sort_merge).

The two strategies must return identical column descriptions, values,
NULL positions and row order; only the explain plan distinguishes them,
and only when a strategy was requested explicitly.
"""

from __future__ import annotations

import json

import pytest

from columnar_analytics import (
    ColumnSchema,
    Schema,
    Table,
    explain_files,
    export_query_files,
    query_files,
    write_file,
)
from columnar_analytics.cli import main


def joined_rows(result):
    return [
        [result._columns[c][i] for c in range(len(result.schema.columns))]
        for i in range(result.row_count)
    ]


def schema_dicts(table):
    return [
        {"name": col.name, "type": col.type, "nullable": col.nullable}
        for col in table.schema.columns
    ]


@pytest.fixture()
def paths(tmp_path):
    # Duplicate keys (10 appears twice on each side), NULL keys and an
    # unmatched value on both sides exercise every join edge case.
    left = tmp_path / "left.caef"
    right = tmp_path / "right.caef"
    write_file(
        left,
        Table(
            Schema(
                [
                    ColumnSchema("id", "int64"),
                    ColumnSchema("k", "int64", nullable=True),
                    ColumnSchema("name", "utf8", nullable=True),
                ]
            ),
            {
                "id": [1, 2, 3, 4, 5],
                "k": [10, 20, 10, None, 30],
                "name": ["a", "b", "c", "d", "e"],
            },
        ),
    )
    write_file(
        right,
        Table(
            Schema(
                [
                    ColumnSchema("rid", "int64"),
                    ColumnSchema("k", "int64", nullable=True),
                    ColumnSchema("tag", "utf8", nullable=True),
                ]
            ),
            {
                "rid": [100, 200, 300, 400, 500],
                "k": [30, 10, None, 10, 20],
                "tag": ["p", "q", None, "r", "s"],
            },
        ),
    )
    return {"l": left, "r": right}


JOIN_SQL = [
    "SELECT * FROM l INNER JOIN r ON l.k = r.k",
    "SELECT * FROM l LEFT JOIN r ON l.k = r.k",
    "SELECT * FROM l RIGHT JOIN r ON l.k = r.k",
    "SELECT * FROM l FULL OUTER JOIN r ON l.k = r.k",
    "SELECT l.id, r.rid FROM l INNER JOIN r ON l.k = r.k "
    "WHERE r.tag IS NOT NULL ORDER BY l.id DESC, r.rid ASC LIMIT 3",
    "SELECT l.name, COUNT(*), SUM(r.rid) FROM l LEFT JOIN r ON l.k = r.k "
    "GROUP BY l.name HAVING COUNT(*) >= 1 ORDER BY l.name",
    "SELECT l.id, r.rid FROM l FULL OUTER JOIN r ON l.k = r.k "
    "WHERE l.id IS NULL OR r.rid IS NULL ORDER BY l.id ASC NULLS LAST",
    "SELECT l.id, r.rid, l.k + r.k AS s FROM l INNER JOIN r ON l.k = r.k "
    "WHERE l.k + r.k > 15 ORDER BY s",
]


@pytest.mark.parametrize("sql", JOIN_SQL)
@pytest.mark.parametrize("strategy", ["hash", "sort_merge"])
def test_strategies_match_default_path(paths, sql, strategy):
    default = query_files(paths, sql)
    chosen = query_files(paths, sql, strategy)
    assert schema_dicts(chosen) == schema_dicts(default)
    assert joined_rows(chosen) == joined_rows(default)


def test_strategies_match_each_other_inner_and_left(paths):
    expected_inner = [
        # Left file order; within one left row the right matches follow the
        # right file's original order (rid 200 before 400 for key 10).
        [1, 10, "a", 200, 10, "q"],
        [1, 10, "a", 400, 10, "r"],
        [2, 20, "b", 500, 20, "s"],
        [3, 10, "c", 200, 10, "q"],
        [3, 10, "c", 400, 10, "r"],
        [5, 30, "e", 100, 30, "p"],
    ]
    # The unmatched NULL-keyed left row (id 4) keeps its original position
    # between ids 3 and 5.
    expected_left = expected_inner[:5] + [
        [4, None, "d", None, None, None],
        expected_inner[5],
    ]
    for kind, expected in (("INNER", expected_inner), ("LEFT", expected_left)):
        sql = f"SELECT * FROM l {kind} JOIN r ON l.k = r.k"
        hash_rows = joined_rows(query_files(paths, sql, "hash"))
        merge_rows = joined_rows(query_files(paths, sql, "sort_merge"))
        assert hash_rows == expected
        assert merge_rows == expected


def test_strategies_match_each_other_right(paths):
    expected_right = [
        # Right-file order; the matches of one right row follow left-file
        # order (rid 200/400 on key 10 expand to left ids 1 then 3), and
        # unmatched right rows (rid 300, key NULL) pad the left side.
        [5, 30, "e", 100, 30, "p"],
        [1, 10, "a", 200, 10, "q"],
        [3, 10, "c", 200, 10, "q"],
        [None, None, None, 300, None, None],
        [1, 10, "a", 400, 10, "r"],
        [3, 10, "c", 400, 10, "r"],
        [2, 20, "b", 500, 20, "s"],
    ]
    sql = "SELECT * FROM l RIGHT JOIN r ON l.k = r.k"
    default_rows = joined_rows(query_files(paths, sql))
    hash_rows = joined_rows(query_files(paths, sql, "hash"))
    merge_rows = joined_rows(query_files(paths, sql, "sort_merge"))
    assert default_rows == expected_right
    assert hash_rows == expected_right
    assert merge_rows == expected_right


def test_strategies_match_each_other_full(paths):
    expected_inner = [
        [1, 10, "a", 200, 10, "q"],
        [1, 10, "a", 400, 10, "r"],
        [2, 20, "b", 500, 20, "s"],
        [3, 10, "c", 200, 10, "q"],
        [3, 10, "c", 400, 10, "r"],
        [5, 30, "e", 100, 30, "p"],
    ]
    # LEFT JOIN order first (unmatched NULL-keyed left row id 4 between
    # ids 3 and 5), then the unmatched right rows in right-file order.
    expected_full = expected_inner[:5] + [
        [4, None, "d", None, None, None],
        expected_inner[5],
        [None, None, None, 300, None, None],
    ]
    sql = "SELECT * FROM l FULL OUTER JOIN r ON l.k = r.k"
    default_rows = joined_rows(query_files(paths, sql))
    hash_rows = joined_rows(query_files(paths, sql, "hash"))
    merge_rows = joined_rows(query_files(paths, sql, "sort_merge"))
    assert default_rows == expected_full
    assert hash_rows == expected_full
    assert merge_rows == expected_full
    assert schema_dicts(query_files(paths, sql)) == schema_dicts(
        query_files(paths, sql, "sort_merge")
    )


def test_int64_float64_mixed_keys_and_signed_zero(tmp_path):
    left = tmp_path / "li.caef"
    right = tmp_path / "rf.caef"
    write_file(
        left,
        Table(Schema([ColumnSchema("k", "int64")]), {"k": [0, 1, 2, 2, 3]}),
    )
    write_file(
        right,
        Table(
            Schema([ColumnSchema("k", "float64", nullable=True)]),
            {"k": [2.0, -0.0, 2.0, None, 1.0]},
        ),
    )
    sql = "SELECT a.k, b.k FROM a INNER JOIN b ON a.k = b.k"
    sources = {"a": left, "b": right}
    expected = [
        # +0 and -0 match; duplicate keys produce the full combination set
        # in left order / right order regardless of strategy.
        [0, -0.0],
        [1, 1.0],
        [2, 2.0],
        [2, 2.0],
        [2, 2.0],
        [2, 2.0],
    ]
    assert joined_rows(query_files(sources, sql, "hash")) == expected
    assert joined_rows(query_files(sources, sql, "sort_merge")) == expected


def test_utf8_keys_equivalence(tmp_path):
    left = tmp_path / "lu.caef"
    right = tmp_path / "ru.caef"
    write_file(
        left,
        Table(
            Schema([ColumnSchema("k", "utf8", nullable=True)]),
            {"k": ["x", "y", "y", None, "z"]},
        ),
    )
    write_file(
        right,
        Table(
            Schema([ColumnSchema("k", "utf8", nullable=True)]),
            {"k": ["y", "x", "y", "w", None]},
        ),
    )
    sql = "SELECT a.k, b.k FROM a LEFT JOIN b ON a.k = b.k"
    sources = {"a": left, "b": right}
    expected = [
        ["x", "x"],
        ["y", "y"],
        ["y", "y"],
        ["y", "y"],
        ["y", "y"],
        [None, None],
        ["z", None],
    ]
    default = joined_rows(query_files(sources, sql))
    assert joined_rows(query_files(sources, sql, "hash")) == default
    assert joined_rows(query_files(sources, sql, "sort_merge")) == expected
    assert default == expected


def test_repeated_runs_are_byte_stable(paths):
    sql = "SELECT * FROM l LEFT JOIN r ON l.k = r.k"
    first = joined_rows(query_files(paths, sql, "sort_merge"))
    for _ in range(3):
        assert joined_rows(query_files(paths, sql, "sort_merge")) == first
        assert joined_rows(query_files(paths, sql, "hash")) == first


# ---------------------------------------------------------------------------
# Statements without a JOIN: any legal strategy is accepted and inert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["hash", "sort_merge"])
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id, name FROM l WHERE id >= 2 ORDER BY id",
        "SELECT k, COUNT(*), SUM(id) FROM l GROUP BY k HAVING COUNT(*) > 0 ORDER BY k",
        "SELECT id * 2 AS dbl FROM l WHERE id < 3 ORDER BY dbl DESC LIMIT 1",
    ],
)
def test_strategy_inert_without_join(paths, strategy, sql):
    default = query_files(paths, sql)
    chosen = query_files(paths, sql, strategy)
    assert schema_dicts(chosen) == schema_dicts(default)
    assert joined_rows(chosen) == joined_rows(default)
    plan = explain_files(paths, sql, strategy)
    assert [op["operator"] for op in plan["operators"] if op["operator"] == "Join"] == []
    # The plan is unchanged from the no-strategy call.
    assert plan == explain_files(paths, sql)


# ---------------------------------------------------------------------------
# Explain plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy,label", [("hash", "HASH"), ("sort_merge", "SORT_MERGE")]
)
def test_explain_strategy_field(paths, strategy, label):
    sql = "SELECT * FROM l INNER JOIN r ON l.k = r.k"
    plan = explain_files(paths, sql, strategy)
    join_ops = [op for op in plan["operators"] if op["operator"] == "Join"]
    assert join_ops == [
        {
            "operator": "Join",
            "type": "INNER",
            "left": {"table": "l", "column": "k"},
            "right": {"table": "r", "column": "k"},
            "strategy": label,
        }
    ]
    # Rest of the plan keeps the exact default shape.
    default = explain_files(paths, sql)
    default_join = [op for op in default["operators"] if op["operator"] == "Join"][0]
    assert "strategy" not in default_join
    stripped = dict(join_ops[0])
    stripped.pop("strategy")
    assert stripped == default_join


def test_explain_strategy_matches_query_and_export(paths):
    sql = "SELECT l.id, r.rid FROM l LEFT JOIN r ON l.k = r.k"
    for strategy in ("hash", "sort_merge"):
        plan = explain_files(paths, sql, strategy)
        query_result = query_files(paths, sql, strategy)
        # Output schema in the plan matches the strategy's query result.
        assert plan["output"] == schema_dicts(query_result)


# ---------------------------------------------------------------------------
# Invalid strategy: ValueError before any file is opened
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["HASH", "merge", "nested_loop", "", 7, object()])
def test_invalid_strategy_python(bad, tmp_path):
    sources = {"l": tmp_path / "missing-l.caef", "r": tmp_path / "missing-r.caef"}
    sql_join = "SELECT * FROM l INNER JOIN r ON l.k = r.k"
    sql_plain = "SELECT id FROM l"
    with pytest.raises(ValueError):
        query_files(sources, sql_join, bad)
    with pytest.raises(ValueError):
        query_files(sources, sql_plain, bad)
    with pytest.raises(ValueError):
        explain_files(sources, sql_join, bad)
    with pytest.raises(ValueError):
        export_query_files(sources, sql_join, tmp_path / "out.csv", "csv", bad)


def test_invalid_strategy_no_file_access(tmp_path):
    # An OSError would prove a file was opened; ValueError must come first.
    sources = {"l": "/nonexistent/join-left.caef", "r": "/nonexistent/join-right.caef"}
    sql = "SELECT * FROM l INNER JOIN r ON l.k = r.k"
    with pytest.raises(ValueError):
        query_files(sources, sql, "sort-merge")
    with pytest.raises(ValueError):
        explain_files(sources, sql, 42)
    destination = tmp_path / "must-not-exist.csv"
    with pytest.raises(ValueError):
        export_query_files(sources, sql, destination, "csv", "hash_join")
    assert not destination.exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_strategy_variants_equivalent(paths, capsys):
    sources_json = json.dumps({"l": str(paths["l"]), "r": str(paths["r"])})
    sql = "SELECT l.id, r.rid FROM l LEFT JOIN r ON l.k = r.k ORDER BY l.id, r.rid"
    payloads = []
    for extra in ([], ["--join-strategy", "hash"], ["--join-strategy", "sort_merge"]):
        assert main(["query-files", sources_json, sql, *extra]) == 0
        payloads.append(capsys.readouterr().out)
    assert payloads[0] == payloads[1] == payloads[2]
    assert json.loads(payloads[1])["rows"] == json.loads(payloads[0])["rows"]


def test_cli_explain_strategy(paths, capsys):
    sources_json = json.dumps({"l": str(paths["l"]), "r": str(paths["r"])})
    sql = "SELECT * FROM l INNER JOIN r ON l.k = r.k"
    assert main(["explain-files", sources_json, sql, "--join-strategy", "sort_merge"]) == 0
    plan = json.loads(capsys.readouterr().out)
    join_op = next(op for op in plan["operators"] if op["operator"] == "Join")
    assert join_op["strategy"] == "SORT_MERGE"


@pytest.mark.parametrize("command", ["query-files", "explain-files", "export-files"])
def test_cli_bad_strategy_exit_2(paths, tmp_path, capsys, command):
    sources_json = json.dumps({"l": str(paths["l"]), "r": str(paths["r"])})
    sql = "SELECT * FROM l INNER JOIN r ON l.k = r.k"
    argv = [command, sources_json, sql]
    destination = tmp_path / "out.csv"
    if command == "export-files":
        argv.append(str(destination))
    argv += ["--join-strategy", "sort-merge"]
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip()
    assert "join_strategy" in captured.err
    assert not destination.exists()


def test_cli_export_strategies_byte_identical(paths, tmp_path, capsys):
    sources_json = json.dumps({"l": str(paths["l"]), "r": str(paths["r"])})
    sql = "SELECT l.id, r.tag FROM l LEFT JOIN r ON l.k = r.k"
    outputs = {}
    for strategy in ("hash", "sort_merge"):
        dst = tmp_path / f"{strategy}.csv"
        argv = [
            "export-files",
            sources_json,
            sql,
            str(dst),
            "--join-strategy",
            strategy,
        ]
        assert main(argv) == 0
        assert capsys.readouterr().out == ""
        outputs[strategy] = dst.read_bytes()
    assert outputs["hash"] == outputs["sort_merge"]
    # Repeated runs stay byte-stable.
    again = tmp_path / "again.csv"
    main(["export-files", sources_json, sql, str(again), "--join-strategy", "sort_merge"])
    assert again.read_bytes() == outputs["sort_merge"]


def test_cli_export_strategy_failure_leaves_no_target(paths, tmp_path, capsys):
    sources_json = json.dumps({"l": str(paths["l"]), "r": str(paths["r"])})
    dst = tmp_path / "out.jsonl"
    code = main(
        [
            "export-files",
            sources_json,
            "SELECT * FROM l INNER JOIN r ON l.k = r.k",
            str(dst),
            "--format",
            "jsonl",
            "--join-strategy",
            "merge-sort",
        ]
    )
    assert code == 2
    assert capsys.readouterr().out == ""
    assert not dst.exists()


def test_cli_strategy_inert_without_join(paths, capsys):
    sources_json = json.dumps({"l": str(paths["l"])})
    sql = "SELECT id FROM l WHERE id = 1"
    for strategy in ("hash", "sort_merge"):
        assert main(
            ["query-files", sources_json, sql, "--join-strategy", strategy]
        ) == 0
        out = capsys.readouterr().out
        assert json.loads(out)["rows"] == [[1]]
        assert main(
            ["explain-files", sources_json, sql, "--join-strategy", strategy]
        ) == 0
        plan = json.loads(capsys.readouterr().out)
        assert all(op["operator"] != "Join" for op in plan["operators"])
