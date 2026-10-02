"""Tests for the metadata-only explain layer and the explain CLI commands."""

from __future__ import annotations

import json
import struct
import zlib

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
    query_files,
    read_file,
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
        ColumnSchema("名前", "utf8", nullable=True),
    ]
)

DATA = {
    "id": [1, 2, 3, 4, 5],
    "n": [10, None, 30, 40, None],
    "f": [1.5, 2.5, None, -0.5, 10.0],
    "s": ["a", "b", None, "a", "c"],
    "flag": [True, False, True, False, True],
    "名前": ["い", "ろ", None, "は", "へ"],
}

RIGHT_SCHEMA = Schema(
    [
        ColumnSchema("rid", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("tag", "utf8"),
    ]
)
RIGHT_DATA = {"rid": [100, 200], "k": [1, 2], "tag": ["x", "y"]}


@pytest.fixture()
def path(tmp_path):
    p = tmp_path / "t.caef"
    write_file(p, Table(SCHEMA, DATA), compression="zlib", dictionary_encoding=["s", "名前"])
    return p


@pytest.fixture()
def sources(tmp_path, path):
    right = tmp_path / "r.caef"
    write_file(right, Table(RIGHT_SCHEMA, RIGHT_DATA))
    return {"l": path, "r": right}


def operator_kinds(plan):
    return [op["operator"] for op in plan["operators"]]


def find_operator(plan, name):
    for op in plan["operators"]:
        if op["operator"] == name:
            return op
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Top-level shape
# ---------------------------------------------------------------------------


def test_top_level_keys_fixed_and_ordered(path):
    plan = explain_file(path, "SELECT id FROM input")
    assert list(plan.keys()) == ["sources", "operators", "output"]
    # Plain dict insertion order survives a JSON round trip.
    assert list(json.loads(json.dumps(plan, ensure_ascii=False)).keys()) == [
        "sources",
        "operators",
        "output",
    ]


def test_plan_is_json_serialisable_with_unicode(path):
    plan = explain_file(path, "SELECT 名前 FROM input WHERE 名前 = 'い'")
    text = json.dumps(plan, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    assert "い" in text


def test_sources_description_follows_schema(path):
    plan = explain_file(path, "SELECT id FROM input")
    assert plan["sources"] == [
        {
            "name": "input",
            "row_count": 5,
            "columns": [
                {"name": "id", "type": "int64", "nullable": False},
                {"name": "n", "type": "int64", "nullable": True},
                {"name": "f", "type": "float64", "nullable": True},
                {"name": "s", "type": "utf8", "nullable": True},
                {"name": "flag", "type": "bool", "nullable": False},
                {"name": "名前", "type": "utf8", "nullable": True},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Operators: scan / project
# ---------------------------------------------------------------------------


def test_star_scan_required_columns_in_schema_order(path):
    plan = explain_file(path, "SELECT * FROM input")
    assert operator_kinds(plan) == ["Scan", "Project"]
    scan = find_operator(plan, "Scan")
    assert scan["source"] == "input"
    assert scan["required_columns"] == list(SCHEMA.names)
    assert [col["name"] for col in plan["output"]] == list(SCHEMA.names)


def test_scan_required_union_of_all_clauses_schema_ordered(path):
    sql = (
        "SELECT n, s, COUNT(*) FROM input WHERE f > 0 "
        "GROUP BY n, s ORDER BY n ASC LIMIT 2"
    )
    plan = explain_file(path, sql)
    scan = find_operator(plan, "Scan")
    # Referenced: n (project/group), f (where), s (group); schema order.
    assert scan["required_columns"] == ["n", "f", "s"]


def test_count_star_scan_required_columns_empty(path):
    plan = explain_file(path, "SELECT COUNT(*) FROM input")
    assert find_operator(plan, "Scan")["required_columns"] == []


def test_project_expressions_and_output_names(path):
    plan = explain_file(path, "SELECT id, (n + f) * 2 AS x FROM input LIMIT 3")
    project = find_operator(plan, "Project")
    assert [item["output"] for item in project["expressions"]] == ["id", "x"]
    first = project["expressions"][0]["expression"]
    assert first == {"kind": "column", "name": "id"}
    second = project["expressions"][1]["expression"]
    assert second["kind"] == "arithmetic"
    assert second["operator"] == "*"
    assert plan["output"] == [
        {"name": "id", "type": "int64", "nullable": False},
        {"name": "x", "type": "float64", "nullable": True},
    ]


# ---------------------------------------------------------------------------
# Filter expression tree
# ---------------------------------------------------------------------------


def test_filter_condition_recursive_tree(path):
    plan = explain_file(
        path,
        "SELECT id FROM input WHERE NOT (n <= 10 AND s = 'a') OR flag IS NULL",
    )
    cond = find_operator(plan, "Filter")["condition"]
    assert cond["kind"] == "logic"
    assert cond["operator"] == "OR"
    left, right = cond["operands"]
    assert left["kind"] == "not"
    and_node = left["operands"][0]
    assert and_node["kind"] == "logic"
    assert and_node["operator"] == "AND"
    cmp_node = and_node["operands"][0]
    assert cmp_node["kind"] == "comparison"
    assert cmp_node["operator"] == "<="
    assert cmp_node["operands"][0] == {"kind": "column", "name": "n"}
    assert cmp_node["operands"][1] == {"kind": "literal", "type": "int64", "value": 10}
    assert right["kind"] == "is_null"
    assert right["operator"] == "IS NULL"
    assert right["operands"][0] == {"kind": "column", "name": "flag"}


def test_filter_typed_literals(path):
    cases = [
        ("flag = TRUE", "bool", True),
        ("f = -1.25", "float64", -1.25),
        ("s = ''", "utf8", ""),
        ("n = -9223372036854775808", "int64", -(2**63)),
        ("id IS NOT NULL", None, None),
    ]
    for sql_suffix, type_name, value in cases:
        plan = explain_file(path, f"SELECT id FROM input WHERE {sql_suffix}")
        cond = find_operator(plan, "Filter")["condition"]
        if sql_suffix.startswith("id IS"):
            assert cond["operator"] == "IS NOT NULL"
            continue
        leaf = cond["operands"][1]
        assert leaf == {"kind": "literal", "type": type_name, "value": value}


# ---------------------------------------------------------------------------
# Aggregate / sort / limit
# ---------------------------------------------------------------------------


def test_aggregate_plan_with_group_by(path):
    plan = explain_file(
        path,
        "SELECT s, COUNT(*), SUM(n), AVG(f), MIN(s), MAX(n) FROM input "
        "GROUP BY s ORDER BY COUNT(*) DESC NULLS FIRST LIMIT 2",
    )
    agg = find_operator(plan, "Aggregate")
    assert agg["group_keys"] == ["s"]
    assert agg["aggregates"] == [
        {"function": "COUNT", "argument": None, "output": "COUNT(*)"},
        {"function": "SUM", "argument": "n", "output": "SUM(n)"},
        {"function": "AVG", "argument": "f", "output": "AVG(f)"},
        {"function": "MIN", "argument": "s", "output": "MIN(s)"},
        {"function": "MAX", "argument": "n", "output": "MAX(n)"},
    ]
    sort = find_operator(plan, "Sort")
    assert sort["keys"] == [
        {"column": "COUNT(*)", "direction": "DESC", "nulls": "FIRST"}
    ]
    assert find_operator(plan, "Limit") == {"operator": "Limit", "count": 2}
    assert operator_kinds(plan) == [
        "Scan",
        "Aggregate",
        "Sort",
        "Limit",
        "Project",
    ]


def test_aggregate_no_group_by_has_empty_group_keys(path):
    plan = explain_file(path, "SELECT COUNT(*), MIN(f) FROM input")
    agg = find_operator(plan, "Aggregate")
    assert agg["group_keys"] == []
    assert operator_kinds(plan) == ["Scan", "Aggregate", "Project"]
    assert plan["output"] == [
        {"name": "COUNT(*)", "type": "int64", "nullable": False},
        {"name": "MIN(f)", "type": "float64", "nullable": True},
    ]


def test_sort_defaults_and_alias(path):
    plan = explain_file(
        path,
        "SELECT id, n + 1 AS z FROM input ORDER BY z DESC, id ASC NULLS LAST, f",
    )
    keys = find_operator(plan, "Sort")["keys"]
    assert keys == [
        {"column": "z", "direction": "DESC", "nulls": "LAST"},
        {"column": "id", "direction": "ASC", "nulls": "LAST"},
        {"column": "f", "direction": "ASC", "nulls": "LAST"},
    ]


def test_absent_stages_omitted(path):
    plan = explain_file(path, "SELECT id FROM input")
    assert operator_kinds(plan) == ["Scan", "Project"]


# ---------------------------------------------------------------------------
# Output schema matches the executed query result schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM input",
        "SELECT id, s FROM input WHERE n IS NOT NULL ORDER BY id DESC LIMIT 2",
        "SELECT id, n * f AS q FROM input WHERE id > 1 ORDER BY q",
        "SELECT s, COUNT(*), SUM(n), AVG(f) FROM input GROUP BY s "
        "ORDER BY SUM(n) DESC NULLS FIRST LIMIT 3",
        "SELECT COUNT(*) FROM input WHERE flag",
    ],
)
def test_output_matches_query_result_schema(path, sql):
    plan = explain_file(path, sql)
    result = query_file(path, sql)
    assert plan["output"] == [
        {"name": col.name, "type": col.type, "nullable": col.nullable}
        for col in result.schema.columns
    ]


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------


def test_join_plan_shape(sources):
    sql = (
        "SELECT l.id, r.tag, COUNT(*) FROM l INNER JOIN r ON l.id = r.k "
        "WHERE l.s IS NOT NULL GROUP BY l.id, r.tag ORDER BY l.id LIMIT 5"
    )
    plan = explain_files(sources, sql)
    assert [s["name"] for s in plan["sources"]] == ["l", "r"]
    assert plan["sources"][0]["row_count"] == 5
    assert plan["sources"][1]["row_count"] == 2
    kinds = operator_kinds(plan)
    assert kinds == [
        "Scan",
        "Scan",
        "Join",
        "Filter",
        "Aggregate",
        "Sort",
        "Limit",
        "Project",
    ]
    scans = [op for op in plan["operators"] if op["operator"] == "Scan"]
    # l needs id, s (schema order); r needs k (join), tag (group)
    assert scans[0] == {"operator": "Scan", "source": "l", "required_columns": ["id", "s"]}
    assert scans[1] == {"operator": "Scan", "source": "r", "required_columns": ["k", "tag"]}
    join = find_operator(plan, "Join")
    assert join == {
        "operator": "Join",
        "type": "INNER",
        "left": {"table": "l", "column": "id"},
        "right": {"table": "r", "column": "k"},
    }


def test_left_join_output_nullability_and_on_keys_scanned(sources):
    # Nothing projected except COUNT(*): the ON keys must still be scanned.
    plan = explain_files(
        sources, "SELECT COUNT(*) FROM l LEFT JOIN r ON l.n = r.k"
    )
    scans = {op["source"]: op["required_columns"] for op in plan["operators"] if op["operator"] == "Scan"}
    assert scans == {"l": ["n"], "r": ["k"]}
    assert find_operator(plan, "Join")["type"] == "LEFT"

    star = explain_files(sources, "SELECT * FROM l LEFT JOIN r ON l.id = r.k")
    result = query_files(sources, "SELECT * FROM l LEFT JOIN r ON l.id = r.k")
    assert star["output"] == [
        {"name": col.name, "type": col.type, "nullable": col.nullable}
        for col in result.schema.columns
    ]
    # Right-side columns become nullable under LEFT JOIN.
    right_output = [col for col in star["output"] if col["name"].startswith("r.")]
    assert all(col["nullable"] for col in right_output)


def test_right_join_plan_type_and_nullability(sources):
    sql = "SELECT * FROM l RIGHT JOIN r ON l.id = r.k"
    plan = explain_files(sources, sql)
    join = find_operator(plan, "Join")
    assert join == {
        "operator": "Join",
        "type": "RIGHT",
        "left": {"table": "l", "column": "id"},
        "right": {"table": "r", "column": "k"},
    }
    result = query_files(sources, sql)
    assert plan["output"] == [
        {"name": col.name, "type": col.type, "nullable": col.nullable}
        for col in result.schema.columns
    ]
    left_output = [col for col in plan["output"] if col["name"].startswith("l.")]
    right_output = [col for col in plan["output"] if col["name"].startswith("r.")]
    # LEFT columns become nullable under RIGHT JOIN; the right side keeps
    # its file-level nullability (rid/tag non-null, k nullable).
    assert all(col["nullable"] for col in left_output)
    assert {col["name"]: col["nullable"] for col in right_output} == {
        "r.rid": False,
        "r.k": True,
        "r.tag": False,
    }


def test_full_outer_join_plan_type_and_nullability(sources):
    for strategy in (None, "hash", "sort_merge"):
        sql = "SELECT * FROM l FULL OUTER JOIN r ON l.id = r.k"
        plan = explain_files(sources, sql, strategy)
        join = find_operator(plan, "Join")
        assert join["type"] == "FULL"
        if strategy is None:
            assert "strategy" not in join
        else:
            assert join["strategy"] == strategy.upper()
        # Every result column is nullable under FULL OUTER JOIN.
        assert all(col["nullable"] for col in plan["output"])
        result = query_files(sources, sql, strategy)
        assert plan["output"] == [
            {"name": col.name, "type": col.type, "nullable": col.nullable}
            for col in result.schema.columns
        ]


def test_explain_files_single_table(sources):
    plan = explain_files(sources, "SELECT id FROM l WHERE flag = TRUE")
    assert [s["name"] for s in plan["sources"]] == ["l"]
    assert operator_kinds(plan) == ["Scan", "Filter", "Project"]


def test_unreferenced_source_is_never_opened(path, tmp_path):
    ghost = tmp_path / "ghost.caef"
    plan = explain_files({"l": path, "g": ghost}, "SELECT id FROM l")
    assert [s["name"] for s in plan["sources"]] == ["l"]


# ---------------------------------------------------------------------------
# Metadata-only semantics
# ---------------------------------------------------------------------------


def test_explain_ignores_footer_crc_but_read_fails(path):
    raw = path.read_bytes()
    # Flip one byte of the final overall CRC checksum; size stays identical.
    path.write_bytes(raw[:-5] + bytes([raw[-5] ^ 0xFF]) + raw[-4:])
    plan = explain_file(path, "SELECT id FROM input")
    assert plan["output"][0]["name"] == "id"
    with pytest.raises(ColumnarFormatError):
        read_file(path)


def test_explain_ignores_value_level_stats_corruption(tmp_path):
    p = tmp_path / "plain.caef"
    schema = Schema([ColumnSchema("id", "int64")])
    write_file(p, Table(schema, {"id": [42]}))
    raw = p.read_bytes()
    header_len = struct.unpack("<I", raw[5:9])[0]
    header = json.loads(raw[9 : 9 + header_len].decode("utf-8"))
    data = bytearray(raw[9 + header_len : -8])
    data[0] ^= 0x01  # perturb the int64 payload -> min/max stats now wrong
    header["data_crc32"] = zlib.crc32(bytes(data)) & 0xFFFFFFFF
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    body = raw[:4] + bytes([1]) + struct.pack("<I", len(header_bytes)) + header_bytes + bytes(data)
    p.write_bytes(body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF) + b"END1")
    plan = explain_file(p, "SELECT id FROM input")
    assert plan["sources"][0]["row_count"] == 1
    with pytest.raises(ColumnarFormatError):
        read_file(p)


def test_declared_size_mismatch_raises_format_error(path):
    path.write_bytes(path.read_bytes() + b"X")
    with pytest.raises(ColumnarFormatError):
        explain_file(path, "SELECT id FROM input")


def test_bad_header_json_raises_format_error(tmp_path):
    p = tmp_path / "bad.caef"
    p.write_bytes(b"CAEF" + bytes([1]) + struct.pack("<I", 4) + b"{bad")
    with pytest.raises(ColumnarFormatError):
        explain_file(p, "SELECT id FROM input")


# ---------------------------------------------------------------------------
# Error ordering and classification
# ---------------------------------------------------------------------------


def test_syntax_error_before_file_access(tmp_path):
    missing = tmp_path / "nope.caef"
    with pytest.raises(QuerySyntaxError):
        explain_file(missing, "SELCT id FROM input")


def test_validation_errors(path):
    with pytest.raises(QueryValidationError):
        explain_file(path, "SELECT nope FROM input")
    with pytest.raises(QueryValidationError):
        explain_file(path, "SELECT id FROM input WHERE s = 1")
    with pytest.raises(QueryValidationError):
        explain_file(path, "SELECT s, n FROM input GROUP BY s")


def test_missing_file_is_os_error(tmp_path):
    with pytest.raises(OSError):
        explain_file(tmp_path / "nope.caef", "SELECT id FROM input")


def test_invalid_sources_value_error(path):
    for bad in (None, [], {}, {"": str(path)}, {"l": 3}):
        with pytest.raises(ValueError):
            explain_files(bad, "SELECT id FROM l")


def test_invalid_sources_before_file_access(tmp_path):
    with pytest.raises(ValueError):
        explain_files({}, 123)


def test_join_binding_errors(sources):
    with pytest.raises(QueryValidationError):
        # utf8 vs int64 join keys
        explain_files(sources, "SELECT l.id FROM l INNER JOIN r ON l.s = r.k")
    with pytest.raises(QueryValidationError):
        # unknown right-side key column
        explain_files(sources, "SELECT l.id FROM l INNER JOIN r ON l.id = r.missing")
    with pytest.raises(QueryValidationError):
        # unqualified reference in a join query
        explain_files(sources, "SELECT id FROM l INNER JOIN r ON l.id = r.k")
    with pytest.raises(QueryValidationError):
        # unknown left table
        explain_files(sources, "SELECT z.id FROM z INNER JOIN r ON z.id = r.k")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_repeated_explain_equal_and_keyword_case_insensitive(path):
    first = explain_file(path, "select  id , count(*) from input where n > 0 group by id")
    second = explain_file(path, "SeLeCt id, COUNT(*) FrOm input WhErE n > 0 GrOuP bY id")
    assert first == second
    third = explain_file(path, "SELECT id, COUNT(*) FROM input WHERE n > 0 GROUP BY id")
    assert first == third


def test_join_explain_deterministic(sources):
    sql = "SELECT l.id, r.tag FROM l INNER JOIN r ON l.id = r.k ORDER BY l.id"
    assert explain_files(sources, sql) == explain_files(sources, sql)


# ---------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------


def test_cli_explain_single_line_compact(path, capsys):
    code = main(["explain", str(path), "SELECT id FROM input WHERE id > 1"])
    assert code == 0
    out = capsys.readouterr().out
    assert out.endswith("\n")
    assert out.count("\n") == 1
    payload = json.loads(out)
    assert list(payload.keys()) == ["sources", "operators", "output"]
    # Compact: no spaces after JSON separators.
    assert ' "operator"' not in out
    assert ", " not in out.rstrip("\n")


def test_cli_explain_files(sources, capsys):
    sources_json = json.dumps({k: str(v) for k, v in sources.items()})
    code = main(
        [
            "explain-files",
            sources_json,
            "SELECT l.id FROM l INNER JOIN r ON l.id = r.k",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert list(payload.keys()) == ["sources", "operators", "output"]
    assert [s["name"] for s in payload["sources"]] == ["l", "r"]


def test_cli_byte_identical_on_repeat_and_whitespace(path, capsys):
    assert main(["explain", str(path), "select id from input"]) == 0
    first = capsys.readouterr().out
    assert main(["explain", str(path), "SELECT  id  FROM  input"]) == 0
    second = capsys.readouterr().out
    assert first == second


def test_cli_query_errors_exit_2_empty_stdout(path, capsys):
    code = main(["explain", str(path), "SELCT id FROM input"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err

    code = main(["explain", str(path), "SELECT nope FROM input"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""

    path.write_bytes(b"CAEF" + bytes([1]) + struct.pack("<I", 2) + b"{}")
    code = main(["explain", str(path), "SELECT id FROM input"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""


def test_cli_os_error_exit_1_empty_stdout(tmp_path, capsys):
    code = main(["explain", str(tmp_path / "nope.caef"), "SELECT id FROM input"])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err


def test_cli_explain_files_invalid_json_exit_2(capsys):
    code = main(["explain-files", "{not json", "SELECT id FROM l"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""


def test_cli_explain_files_bad_sources_exit_2(path, capsys):
    code = main(["explain-files", json.dumps({"l": 3}), "SELECT id FROM l"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
