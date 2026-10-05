"""Stage-boundary tests for the split query pipeline.

These tests exercise the internal stages -- pure SQL parsing
(:mod:`columnar_analytics.parser`), name/source resolution
(:mod:`columnar_analytics.resolve`) and schema binding
(:mod:`columnar_analytics.binder`) -- directly, without creating any
columnar data file, and check that one shared bound result drives both the
executor and the explain plan.  They also pin the module dependency
direction: no internal stage module may import a later stage.
"""

import pytest

from columnar_analytics import (
    QuerySyntaxError,
    QueryValidationError,
)
from columnar_analytics.binder import _bind_select
from columnar_analytics.errors import QuerySyntaxError as ErrorsQuerySyntaxError
from columnar_analytics.executor import _run_query
from columnar_analytics.format import ColumnSchema, Schema, Table
from columnar_analytics.parser import _Parser, _Select, _tokenize
from columnar_analytics.plan import _build_explain
from columnar_analytics.query import QuerySyntaxError as QueryQuerySyntaxError
from columnar_analytics.resolve import (
    _referenced_source_paths,
    _resolve_statement,
)


def _parse(sql, allow_join=False):
    return _Parser(_tokenize(sql), allow_join=allow_join).parse()


def _schema():
    return Schema(
        [
            ColumnSchema("id", "int64"),
            ColumnSchema("v", "float64", nullable=True),
            ColumnSchema("s", "utf8"),
        ]
    )


# ---------------------------------------------------------------------------
# Parse stage: SQL text in, _Select out; no file is ever involved.
# ---------------------------------------------------------------------------


def test_parse_produces_statement_tree_without_files():
    select = _parse("select id, v * 2 as d from input where id >= 2 limit 3")
    assert isinstance(select, _Select)
    assert select.table == "input"
    assert select.limit == 3
    assert select.where is not None
    assert [item.kind for item in select.items] == ["column", "expr"]


def test_parse_join_requires_multi_table_grammar():
    with pytest.raises(QuerySyntaxError):
        _parse("select * from a inner join b on a.id = b.id")
    select = _parse("select * from a inner join b on a.id = b.id", allow_join=True)
    assert len(select.joins) == 1
    assert select.joins[0].kind == "inner"


def test_parse_syntax_errors_need_no_file():
    for sql in ("select from input", "select id from input where", 42, "select 1..2 from input"):
        with pytest.raises(QuerySyntaxError):
            _parse(sql)


def test_stage_modules_have_no_import_cycles():
    # Statically check the stage dependency direction: every internal
    # import of a stage module must point at an earlier (or equal-depth)
    # stage, so the graph is acyclic and parsing/binding stay usable in
    # isolation.
    depth = {
        "errors": 0,
        "parser": 1,
        "expr": 2,
        "binder": 3,
        "resolve": 3,
        "pushdown": 4,
        "prepare": 5,
        "executor": 5,
        "plan": 5,
        "query": 6,
    }
    import ast
    import pathlib

    package = pathlib.Path(__file__).parent.parent / "columnar_analytics"
    for name, level in depth.items():
        tree = ast.parse((package / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                target = node.module.rsplit(".", 1)[-1]
                if target in depth:
                    assert depth[target] < level, f"{name} imports later stage {target}"


# ---------------------------------------------------------------------------
# Resolve stage: source/table names are resolved without opening any file.
# ---------------------------------------------------------------------------


def test_resolve_statement_never_opens_sources(tmp_path):
    missing = tmp_path / "does-not-exist.caef"
    paths, select, strategy, from_key, steps = _resolve_statement(
        {"a": missing, "unused": tmp_path / "also-missing.caef"},
        "select a.id from a",
        join_strategy="hash",
    )
    assert from_key == "a"
    assert steps == ()
    assert strategy == "hash"
    # Only the referenced source is reported; "unused" is never touched.
    unused = tmp_path / "unused.caef"
    assert _referenced_source_paths({"a": missing, "unused": unused}, "select a.id from a") == (
        missing,
    )


def test_resolve_validates_arguments_before_parsing():
    with pytest.raises(ValueError):
        _resolve_statement({}, "select id from a")
    with pytest.raises(ValueError):
        _resolve_statement({"a": "x.caef"}, "select id from a", join_strategy="nope")
    # A syntax error still surfaces (with valid sources) before any file I/O.
    with pytest.raises(QuerySyntaxError):
        _resolve_statement({"a": "missing.caef"}, "select from a")


# ---------------------------------------------------------------------------
# Bind stage: a parsed statement binds against a bare in-memory schema.
# ---------------------------------------------------------------------------


def test_bind_against_in_memory_schema():
    bound = _bind_select(_parse("select id, v * 2 as d from input where id >= 2"), _schema())
    assert bound["mode"] == "plain"
    assert [item.output_name for item in bound["items"]] == ["id", "d"]
    assert bound["where"] is not None


def test_bind_validation_errors_need_no_file():
    select = _parse("select nope from input")
    with pytest.raises(QueryValidationError):
        _bind_select(select, _schema())
    with pytest.raises(QueryValidationError):
        _bind_select(_parse("select id from input where s > 1"), _schema())
    with pytest.raises(QueryValidationError):
        _bind_select(_parse("select id from wrong_table"), _schema())


def test_one_bound_result_drives_execution_and_plan():
    schema = _schema()
    select = _parse("select id, v * 2 as d from input where id >= 2 order by d desc limit 1")
    bound = _bind_select(select, schema)

    table = Table(schema, {"id": [1, 2, 3], "v": [1.5, None, 3.0], "s": ["a", "b", "c"]})
    result = _run_query(table, select)
    assert result.columns == {"id": [3], "d": [6.0]}

    metadata = {
        "row_count": 3,
        "columns": [
            {"name": c.name, "type": c.type, "nullable": c.nullable}
            for c in schema.columns
        ],
    }
    plan = _build_explain((("input", metadata, schema),), schema, select, bound)
    assert [op["operator"] for op in plan["operators"]] == [
        "Scan",
        "Filter",
        "Sort",
        "Limit",
        "Project",
    ]
    # The plan's declared output matches the executed result schema.
    assert plan["output"] == [
        {"name": col.name, "type": col.type, "nullable": col.nullable}
        for col in result.schema.columns
    ]


# ---------------------------------------------------------------------------
# Public surface: the exceptions and entry points keep their historical home.
# ---------------------------------------------------------------------------


def test_public_exceptions_are_shared_across_modules():
    assert QueryQuerySyntaxError is ErrorsQuerySyntaxError
    assert QuerySyntaxError is ErrorsQuerySyntaxError
    assert issubclass(QueryValidationError, Exception)
