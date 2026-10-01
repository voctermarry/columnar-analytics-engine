"""Tests for the plan-only explain entry points and the explain CLI commands."""

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
    query_file,
    write_file,
)
from columnar_analytics.cli import main


SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("name", "utf8", nullable=True),
        ColumnSchema("flag", "bool"),
    ]
)
DATA = {
    "id": [1, 2, 3, 4],
    "k": [10, 20, 10, None],
    "name": ["a", "b", "c", "d"],
    "flag": [True, False, True, False],
}

LEFT_SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("name", "utf8", nullable=True),
    ]
)
LEFT_DATA = {"id": [1, 2, 3, 4], "k": [10, 20, 10, None], "name": ["a", "b", "c", "d"]}

RIGHT_SCHEMA = Schema(
    [
        ColumnSchema("rid", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("tag", "utf8", nullable=True),
    ]
)
RIGHT_DATA = {"rid": [100, 200, 300], "k": [10, 10, 30], "tag": ["x", "y", "z"]}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "data.caef"
    write_file(p, Table(SCHEMA, DATA))
    return p


@pytest.fixture()
def paths(tmp_path):
    left = tmp_path / "left.caef"
    right = tmp_path / "right.caef"
    write_file(left, Table(LEFT_SCHEMA, LEFT_DATA))
    write_file(right, Table(RIGHT_SCHEMA, RIGHT_DATA))
    return {"l": left, "r": right}


def operator_names(plan):
    return [op["operator"] for op in plan["operators"]]


def operators(plan, name):
    return [op for op in plan["operators"] if op["operator"] == name]


# ---------------------------------------------------------------------------
# Top-level shape
# ---------------------------------------------------------------------------


def test_plan_top_level_keys(path):
    plan = explain_file(path, "SELECT id FROM input")
    assert list(plan) == ["sources", "operators", "output"]
    # JSON-serialisable as-is.
    json.dumps(plan, ensure_ascii=False)


def test_sources_describe_the_input(path):
    plan = explain_file(path, "SELECT id FROM input")
    assert plan["sources"] == [
        {
            "name": "input",
            "row_count": 4,
            "columns": [
                {"name": "id", "type": "int64", "nullable": False},
                {"name": "k", "type": "int64", "nullable": True},
                {"name": "name", "type": "utf8", "nullable": True},
                {"name": "flag", "type": "bool", "nullable": False},
            ],
        }
    ]


def test_minimal_plan_is_scan_and_project(path):
    plan = explain_file(path, "SELECT id FROM input")
    assert operator_names(plan) == ["Scan", "Project"]
    scan = plan["operators"][0]
    assert scan == {"operator": "Scan", "table": "input", "required_columns": ["id"]}
    project = plan["operators"][1]
    assert project["expressions"] == [
        {"expression": {"kind": "column", "name": "id"}, "name": "id"}
    ]


def test_output_matches_query_result_schema(path):
    sql = (
        "SELECT name, COUNT(*), SUM(id), AVG(id) FROM input "
        "GROUP BY name ORDER BY COUNT(*) DESC NULLS LAST LIMIT 10"
    )
    plan = explain_file(path, sql)
    result = query_file(path, sql)
    assert plan["output"] == [
        {"name": col.name, "type": col.type, "nullable": col.nullable}
        for col in result.schema.columns
    ]


def test_output_matches_plain_query_with_expression(path):
    sql = "SELECT id, id * 2 + 1 AS doubled FROM input WHERE k IS NOT NULL ORDER BY doubled DESC LIMIT 2"
    plan = explain_file(path, sql)
    result = query_file(path, sql)
    assert plan["output"] == [
        {"name": col.name, "type": col.type, "nullable": col.nullable}
        for col in result.schema.columns
    ]


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def test_full_stage_order(path):
    sql = (
        "SELECT k, COUNT(*) FROM input WHERE id > 1 GROUP BY k "
        "ORDER BY COUNT(*) DESC LIMIT 3"
    )
    plan = explain_file(path, sql)
    assert operator_names(plan) == [
        "Scan",
        "Filter",
        "Aggregate",
        "Sort",
        "Limit",
        "Project",
    ]


def test_scan_required_columns_follow_schema_order(path):
    # Projection order differs from schema order; required_columns does not.
    plan = explain_file(path, "SELECT name, id FROM input WHERE k > 5")
    scan = operators(plan, "Scan")[0]
    assert scan["required_columns"] == ["id", "k", "name"]


def test_scan_required_columns_empty_for_count_star_only(path):
    plan = explain_file(path, "SELECT COUNT(*) FROM input")
    scan = operators(plan, "Scan")[0]
    assert scan["required_columns"] == []


def test_star_scan_requires_every_column(path):
    plan = explain_file(path, "SELECT * FROM input")
    scan = operators(plan, "Scan")[0]
    assert scan["required_columns"] == ["id", "k", "name", "flag"]


def test_filter_condition_is_recursive(path):
    plan = explain_file(
        path, "SELECT id FROM input WHERE NOT (k >= 10 AND name != 'x') OR flag = TRUE"
    )
    condition = operators(plan, "Filter")[0]["condition"]
    assert condition == {
        "kind": "logical",
        "operator": "OR",
        "operands": [
            {
                "kind": "not",
                "operator": "NOT",
                "operands": [
                    {
                        "kind": "logical",
                        "operator": "AND",
                        "operands": [
                            {
                                "kind": "comparison",
                                "operator": ">=",
                                "operands": [
                                    {"kind": "column", "name": "k"},
                                    {"kind": "literal", "type": "int64", "value": 10},
                                ],
                            },
                            {
                                "kind": "comparison",
                                "operator": "!=",
                                "operands": [
                                    {"kind": "column", "name": "name"},
                                    {"kind": "literal", "type": "utf8", "value": "x"},
                                ],
                            },
                        ],
                    }
                ],
            },
            {
                "kind": "comparison",
                "operator": "=",
                "operands": [
                    {"kind": "column", "name": "flag"},
                    {"kind": "literal", "type": "bool", "value": True},
                ],
            },
        ],
    }


def test_filter_condition_with_arithmetic_and_is_null(path):
    plan = explain_file(
        path, "SELECT id FROM input WHERE k + 1 > 4 AND name IS NOT NULL"
    )
    condition = operators(plan, "Filter")[0]["condition"]
    assert condition["kind"] == "logical"
    cmp_node, isnull_node = condition["operands"]
    assert cmp_node == {
        "kind": "comparison",
        "operator": ">",
        "operands": [
            {
                "kind": "arithmetic",
                "operator": "+",
                "operands": [
                    {"kind": "column", "name": "k"},
                    {"kind": "literal", "type": "int64", "value": 1},
                ],
            },
            {"kind": "literal", "type": "int64", "value": 4},
        ],
    }
    assert isnull_node == {
        "kind": "is_null",
        "operator": "IS NOT NULL",
        "operands": [{"kind": "column", "name": "name"}],
    }


def test_aggregate_operator(path):
    plan = explain_file(
        path, "SELECT k, COUNT(*), SUM(id), AVG(id), MIN(name), MAX(name) "
        "FROM input GROUP BY k"
    )
    aggregate = operators(plan, "Aggregate")[0]
    assert aggregate["group_by"] == ["k"]
    assert aggregate["aggregates"] == [
        {"function": "COUNT", "argument": None, "output": "COUNT(*)"},
        {"function": "SUM", "argument": "id", "output": "SUM(id)"},
        {"function": "AVG", "argument": "id", "output": "AVG(id)"},
        {"function": "MIN", "argument": "name", "output": "MIN(name)"},
        {"function": "MAX", "argument": "name", "output": "MAX(name)"},
    ]


def test_sort_operator(path):
    plan = explain_file(
        path, "SELECT id FROM input ORDER BY k DESC NULLS FIRST, id"
    )
    sort = operators(plan, "Sort")[0]
    assert sort["keys"] == [
        {"key": "k", "direction": "DESC", "nulls": "FIRST"},
        {"key": "id", "direction": "ASC", "nulls": "LAST"},
    ]


def test_sort_on_aggregate_result(path):
    plan = explain_file(
        path, "SELECT k, COUNT(*) FROM input GROUP BY k ORDER BY COUNT(*) DESC"
    )
    sort = operators(plan, "Sort")[0]
    assert sort["keys"] == [
        {"key": "COUNT(*)", "direction": "DESC", "nulls": "LAST"}
    ]


def test_limit_operator(path):
    plan = explain_file(path, "SELECT id FROM input LIMIT 7")
    assert operators(plan, "Limit")[0] == {"operator": "Limit", "count": 7}


def test_project_expressions_and_names(path):
    plan = explain_file(path, "SELECT id, -(id * 2) AS neg FROM input")
    project = operators(plan, "Project")[0]
    assert project["expressions"] == [
        {"expression": {"kind": "column", "name": "id"}, "name": "id"},
        {
            "expression": {
                "kind": "unary",
                "operator": "-",
                "operands": [
                    {
                        "kind": "arithmetic",
                        "operator": "*",
                        "operands": [
                            {"kind": "column", "name": "id"},
                            {"kind": "literal", "type": "int64", "value": 2},
                        ],
                    }
                ],
            },
            "name": "neg",
        },
    ]


def test_omitted_stages_stay_absent(path):
    plan = explain_file(path, "SELECT id FROM input ORDER BY id")
    assert operator_names(plan) == ["Scan", "Sort", "Project"]


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------


def test_join_plan_shape(paths):
    plan = explain_files(
        paths, "SELECT l.id, r.tag FROM l INNER JOIN r ON l.k = r.k WHERE l.id > 1"
    )
    assert [s["name"] for s in plan["sources"]] == ["l", "r"]
    assert plan["sources"][0]["row_count"] == 4
    assert plan["sources"][1]["row_count"] == 3
    assert operator_names(plan) == ["Scan", "Scan", "Join", "Filter", "Project"]
    join = operators(plan, "Join")[0]
    assert join == {
        "operator": "Join",
        "type": "inner",
        "left_key": "l.k",
        "right_key": "r.k",
    }


def test_join_scan_required_columns_are_per_source(paths):
    plan = explain_files(
        paths, "SELECT l.id, r.tag FROM l LEFT JOIN r ON l.k = r.k"
    )
    scans = operators(plan, "Scan")
    assert scans[0] == {
        "operator": "Scan",
        "table": "l",
        "required_columns": ["id", "k"],
    }
    assert scans[1] == {
        "operator": "Scan",
        "table": "r",
        "required_columns": ["k", "tag"],
    }


def test_join_filter_uses_qualified_bound_names(paths):
    plan = explain_files(
        paths, "SELECT l.id FROM l INNER JOIN r ON l.k = r.k WHERE r.tag = 'x'"
    )
    condition = operators(plan, "Filter")[0]["condition"]
    assert condition == {
        "kind": "comparison",
        "operator": "=",
        "operands": [
            {"kind": "column", "name": "r.tag"},
            {"kind": "literal", "type": "utf8", "value": "x"},
        ],
    }


def test_join_output_matches_query_result_schema(paths):
    from columnar_analytics import query_files

    sql = "SELECT * FROM l LEFT JOIN r ON l.k = r.k"
    plan = explain_files(paths, sql)
    result = query_files(paths, sql)
    assert plan["output"] == [
        {"name": col.name, "type": col.type, "nullable": col.nullable}
        for col in result.schema.columns
    ]


def test_join_aggregate_plan(paths):
    plan = explain_files(
        paths,
        "SELECT l.k, COUNT(*), SUM(r.rid) FROM l INNER JOIN r ON l.k = r.k "
        "GROUP BY l.k ORDER BY l.k LIMIT 5",
    )
    assert operator_names(plan) == [
        "Scan",
        "Scan",
        "Join",
        "Aggregate",
        "Sort",
        "Limit",
        "Project",
    ]
    aggregate = operators(plan, "Aggregate")[0]
    assert aggregate["group_by"] == ["l.k"]
    assert aggregate["aggregates"] == [
        {"function": "COUNT", "argument": None, "output": "COUNT(*)"},
        {"function": "SUM", "argument": "r.rid", "output": "SUM(r.rid)"},
    ]


def test_single_table_explain_files(paths):
    plan = explain_files(paths, "SELECT id FROM l")
    assert [s["name"] for s in plan["sources"]] == ["l"]
    assert operator_names(plan) == ["Scan", "Project"]


def test_unreferenced_source_is_not_touched(paths, tmp_path):
    # A path that does not exist must not be accessed when unreferenced.
    sources = {"l": paths["l"], "missing": tmp_path / "nope.caef"}
    plan = explain_files(sources, "SELECT id FROM l")
    assert [s["name"] for s in plan["sources"]] == ["l"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_syntax_error_before_file_access(tmp_path):
    missing = tmp_path / "missing.caef"
    with pytest.raises(QuerySyntaxError):
        explain_file(missing, "SELECT FROM")


def test_validation_error_unknown_column(path):
    with pytest.raises(QueryValidationError):
        explain_file(path, "SELECT nope FROM input")


def test_validation_error_wrong_table(path):
    with pytest.raises(QueryValidationError):
        explain_file(path, "SELECT id FROM other")


def test_validation_error_type_mismatch(path):
    with pytest.raises(QueryValidationError):
        explain_file(path, "SELECT id FROM input WHERE name = 5")


def test_explain_files_invalid_sources():
    with pytest.raises(ValueError):
        explain_files({}, "SELECT id FROM l")
    with pytest.raises(ValueError):
        explain_files(["l"], "SELECT id FROM l")
    with pytest.raises(ValueError):
        explain_files({"l": 3}, "SELECT id FROM l")


def test_explain_files_unknown_table(paths):
    with pytest.raises(QueryValidationError):
        explain_files(paths, "SELECT id FROM nope")


def test_explain_files_unqualified_join_column(paths):
    with pytest.raises(QueryValidationError):
        explain_files(paths, "SELECT id FROM l INNER JOIN r ON l.k = r.k")


def test_explain_files_unknown_join_key_column(paths):
    with pytest.raises(QueryValidationError):
        explain_files(paths, "SELECT l.id FROM l INNER JOIN r ON l.nope = r.k")


def test_explain_files_incompatible_join_keys(paths, tmp_path):
    weird = tmp_path / "weird.caef"
    write_file(
        weird,
        Table(
            Schema([ColumnSchema("k", "utf8"), ColumnSchema("v", "int64")]),
            {"k": ["a"], "v": [1]},
        ),
    )
    with pytest.raises(QueryValidationError):
        explain_files(
            {"l": paths["l"], "w": weird},
            "SELECT l.id FROM l INNER JOIN w ON l.k = w.k",
        )


def test_missing_file_raises_oserror(tmp_path):
    with pytest.raises(OSError):
        explain_file(tmp_path / "missing.caef", "SELECT id FROM input")


def test_invalid_metadata_raises_format_error(path):
    blob = bytearray(path.read_bytes())
    # Corrupt one byte inside the JSON header (declared sizes stay intact).
    blob[12] ^= 0xFF
    path.write_bytes(bytes(blob))
    with pytest.raises(ColumnarFormatError):
        explain_file(path, "SELECT id FROM input")


def test_declared_size_mismatch_raises_format_error(path):
    blob = path.read_bytes()
    path.write_bytes(blob + b"x")
    with pytest.raises(ColumnarFormatError):
        explain_file(path, "SELECT id FROM input")


def test_data_segment_corruption_is_not_reported(path):
    blob = bytearray(path.read_bytes())
    header_length = int.from_bytes(blob[5:9], "little")
    data_start = 9 + header_length
    blob[data_start] ^= 0xFF  # breaks the data CRC, but explain never reads it
    path.write_bytes(bytes(blob))
    plan = explain_file(path, "SELECT id FROM input")
    assert plan["sources"][0]["row_count"] == 4
    # ...while a real query does report it.
    with pytest.raises(ColumnarFormatError):
        query_file(path, "SELECT id FROM input")


def test_value_level_stats_corruption_is_not_reported(path):
    # Rewrite the header with a wrong min value; explain validates the header
    # structure but never checks value-level statistics against the data.
    blob = bytearray(path.read_bytes())
    header_length = struct.unpack("<I", blob[5:9])[0]
    header = json.loads(bytes(blob[9 : 9 + header_length]).decode("utf-8"))
    header["columns"][0]["min"] = header["columns"][0]["min"] + 1
    new_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    new_blob = (
        bytes(blob[:5])
        + struct.pack("<I", len(new_header))
        + new_header
        + bytes(blob[9 + header_length :])
    )
    path.write_bytes(new_blob)
    plan = explain_file(path, "SELECT id FROM input")
    assert plan["sources"][0]["row_count"] == 4


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_keyword_case_and_whitespace_do_not_change_plan(path):
    a = explain_file(path, "SELECT id FROM input WHERE k > 1 ORDER BY id LIMIT 2")
    b = explain_file(
        path, "  select  id\nfrom   INPUT where k>1  order  by id  limit 2 "
    )
    assert a == b


def test_repeated_explain_is_equal(path):
    sql = "SELECT k, COUNT(*) FROM input GROUP BY k ORDER BY k LIMIT 2"
    assert explain_file(path, sql) == explain_file(path, sql)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_explain_outputs_compact_single_line_json(path, capsys):
    code = main(["explain", str(path), "SELECT id FROM input WHERE k > 1"])
    assert code == 0
    out = capsys.readouterr().out
    assert out.endswith("\n") and out.count("\n") == 1
    assert ", " not in out  # compact separators
    payload = json.loads(out)
    assert list(payload) == ["sources", "operators", "output"]
    assert [op["operator"] for op in payload["operators"]] == [
        "Scan",
        "Filter",
        "Project",
    ]


def test_cli_explain_is_byte_identical_across_runs(path, capsys):
    sql = "SELECT k, COUNT(*) FROM input GROUP BY k ORDER BY k LIMIT 2"
    assert main(["explain", str(path), sql]) == 0
    first = capsys.readouterr().out
    assert main(["explain", str(path), sql]) == 0
    assert capsys.readouterr().out == first


def test_cli_explain_files(paths, capsys):
    sources = json.dumps({"l": str(paths["l"]), "r": str(paths["r"])})
    code = main(
        ["explain-files", sources, "SELECT l.id FROM l INNER JOIN r ON l.k = r.k"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [s["name"] for s in payload["sources"]] == ["l", "r"]
    assert [op["operator"] for op in payload["operators"]] == [
        "Scan",
        "Scan",
        "Join",
        "Project",
    ]


def test_cli_explain_query_error_exit_code(path, capsys):
    assert main(["explain", str(path), "SELECT nope FROM input"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err


def test_cli_explain_syntax_error_exit_code(path, capsys):
    assert main(["explain", str(path), "SELECT FROM"]) == 2
    assert capsys.readouterr().out == ""


def test_cli_explain_oserror_exit_code(tmp_path, capsys):
    assert main(["explain", str(tmp_path / "missing.caef"), "SELECT id FROM input"]) == 1
    assert capsys.readouterr().out == ""


def test_cli_explain_format_error_exit_code(path, capsys):
    blob = bytearray(path.read_bytes())
    blob[12] ^= 0xFF
    path.write_bytes(bytes(blob))
    assert main(["explain", str(path), "SELECT id FROM input"]) == 2
    assert capsys.readouterr().out == ""


def test_cli_explain_files_invalid_sources_json(capsys):
    assert main(["explain-files", "not json", "SELECT 1"]) == 2
    assert capsys.readouterr().out == ""


def test_cli_explain_files_invalid_sources_value(paths, capsys):
    assert main(["explain-files", "{}", "SELECT id FROM l"]) == 2
    assert capsys.readouterr().out == ""


def test_cli_explain_files_query_error(paths, capsys):
    sources = json.dumps({"l": str(paths["l"]), "r": str(paths["r"])})
    assert main(["explain-files", sources, "SELECT id FROM l INNER JOIN r ON l.k = r.k"]) == 2
    assert capsys.readouterr().out == ""
