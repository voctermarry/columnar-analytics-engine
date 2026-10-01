"""Plan-only query explanation.

Public API:

* :func:`explain_file` -- parse, bind and plan a single-file query without
  executing it
* :func:`explain_files` -- the same for the two-file (single join) entry point

Both functions accept exactly the inputs and SQL subset of
:func:`columnar_analytics.query.query_file` /
:func:`columnar_analytics.query.query_files`.  They parse and bind the
statement and read only the file metadata (header) of the sources the
statement references -- data segments are never read, decompressed or
decoded, and no query is executed.  Consequently a corrupted data-section
checksum or value-level statistics mismatch is not reported, while invalid
metadata or a declared-size mismatch still raises
:class:`~columnar_analytics.format.ColumnarFormatError`.

The returned plan is a JSON-serialisable dict with the fixed top-level key
order ``sources``, ``operators``, ``output``:

* ``sources`` -- one entry per referenced table in FROM/JOIN order, each
  with ``name``, ``row_count`` and ``columns`` (``name``/``type``/
  ``nullable`` in schema order).
* ``operators`` -- the pipeline stages in execution order: one ``Scan``
  per source, then ``Join``, ``Filter``, ``Aggregate``, ``Sort``, ``Limit``
  and ``Project``; stages the statement does not use are omitted.
* ``output`` -- the result columns in projection order
  (``name``/``type``/``nullable``), identical to the schema a real query
  would return.

Plans are deterministic: keyword case and insignificant whitespace never
change the plan content, and repeated explanations of the same metadata and
SQL produce equal dicts.
"""

from __future__ import annotations

from typing import Any

from .format import ColumnSchema, Schema, _read_header
from .query import (
    QuerySyntaxError,
    QueryValidationError,
    _Parser,
    _bind_select,
    _join_resolver,
    _resolve_table_ref,
    _rewrite_select,
    _single_table_resolver,
    _tokenize,
    _validate_sources,
)

__all__ = [
    "explain_file",
    "explain_files",
]


# ---------------------------------------------------------------------------
# Metadata-only source loading
# ---------------------------------------------------------------------------


def _header_schema(header: dict) -> Schema:
    return Schema(
        [
            ColumnSchema(entry["name"], entry["type"], entry["nullable"])
            for entry in header["columns"]
        ]
    )


def _read_source(path: Any) -> tuple[Schema, int]:
    """Read only the metadata of one columnar file.

    Returns the schema and the declared row count; data bytes are untouched.
    """
    _, _, header, _ = _read_header(path)
    return _header_schema(header), header["row_count"]


def _join_schema(
    left_key: str,
    left_schema: Schema,
    right_key: str,
    right_schema: Schema,
    join,
) -> tuple[Schema, int, int]:
    """The combined join schema plus the join key indices, without data.

    Mirrors the schema-side checks of the execution path: unknown join key
    columns, incompatible key types and duplicate combined names raise
    :class:`QueryValidationError`.
    """
    left_col = join.left_key[2]
    right_col = join.right_key[2]
    try:
        left_idx = left_schema.index(left_col)
    except KeyError:
        raise QueryValidationError(
            f"unknown column: {f'{left_key}.{left_col}'!r}"
        ) from None
    try:
        right_idx = right_schema.index(right_col)
    except KeyError:
        raise QueryValidationError(
            f"unknown column: {f'{right_key}.{right_col}'!r}"
        ) from None
    left_type = left_schema.columns[left_idx].type
    right_type = right_schema.columns[right_idx].type
    if left_type != right_type and not {left_type, right_type} <= {"int64", "float64"}:
        raise QueryValidationError(
            f"join key types are incompatible: {left_type} and {right_type}"
        )

    combined_columns = [
        ColumnSchema(f"{left_key}.{col.name}", col.type, col.nullable)
        for col in left_schema.columns
    ] + [
        ColumnSchema(
            f"{right_key}.{col.name}",
            col.type,
            True if join.kind == "left" else col.nullable,
        )
        for col in right_schema.columns
    ]
    names = [col.name for col in combined_columns]
    if len(set(names)) != len(names):
        raise QueryValidationError("joined tables produce duplicate column names")
    return Schema(combined_columns), left_idx, right_idx


# ---------------------------------------------------------------------------
# Referenced-column collection (bound trees)
# ---------------------------------------------------------------------------


def _expr_column_indices(node: tuple, out: set) -> None:
    tag = node[0]
    if tag == "column":
        out.add(node[2])
    elif tag == "unary":
        _expr_column_indices(node[1], out)
    elif tag == "arith":
        _expr_column_indices(node[2], out)
        _expr_column_indices(node[3], out)
    elif tag == "not":
        _expr_column_indices(node[1], out)
    elif tag in ("and", "or"):
        _expr_column_indices(node[1], out)
        _expr_column_indices(node[2], out)
    elif tag == "isnull":
        _expr_column_indices(node[1], out)
    elif tag == "cmp":
        _expr_column_indices(node[2], out)
        _expr_column_indices(node[3], out)
    # "literal" carries no column reference.


def _required_column_indices(bound: dict) -> set:
    """Every source column the bound statement reads, as schema indices."""
    indices: set = set()
    for item in bound["items"]:
        if item.kind == "column":
            indices.add(item.col_index)
        elif item.kind == "expr":
            _expr_column_indices(item.expr, indices)
        elif item.kind == "agg" and item.arg_index >= 0:
            indices.add(item.arg_index)
    if bound["where"] is not None:
        _expr_column_indices(bound["where"], indices)
    if bound["mode"] == "aggregate":
        indices.update(bound["group_indices"])
    if bound["mode"] == "plain":
        for entry in bound["order_by"] or ():
            if entry[0] == "col":
                indices.add(entry[1])
            else:  # "expr" sort key (a SELECT alias)
                _expr_column_indices(entry[1], indices)
    return indices


# ---------------------------------------------------------------------------
# JSON rendering of bound expressions and plan nodes
# ---------------------------------------------------------------------------


def _expr_json(node: tuple) -> dict:
    tag = node[0]
    if tag == "literal":
        return {"kind": "literal", "type": node[2], "value": node[1]}
    if tag == "column":
        return {"kind": "column", "name": node[1]}
    if tag == "unary":
        return {
            "kind": "unary",
            "operator": "-" if node[2] else "+",
            "operands": [_expr_json(node[1])],
        }
    if tag == "arith":
        return {
            "kind": "arithmetic",
            "operator": node[1],
            "operands": [_expr_json(node[2]), _expr_json(node[3])],
        }
    if tag == "cmp":
        return {
            "kind": "comparison",
            "operator": node[1],
            "operands": [_expr_json(node[2]), _expr_json(node[3])],
        }
    if tag == "isnull":
        return {
            "kind": "is_null",
            "operator": "IS NOT NULL" if node[2] else "IS NULL",
            "operands": [_expr_json(node[1])],
        }
    if tag == "not":
        return {"kind": "not", "operator": "NOT", "operands": [_expr_json(node[1])]}
    if tag in ("and", "or"):
        return {
            "kind": "logical",
            "operator": tag.upper(),
            "operands": [_expr_json(node[1]), _expr_json(node[2])],
        }
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


def _agg_json(item, schema: Schema) -> dict:
    return {
        "function": item.func,
        "argument": (
            schema.columns[item.arg_index].name if item.arg_index >= 0 else None
        ),
        "output": item.output_name,
    }


def _sort_key_json(entry: tuple, bound: dict, schema: Schema) -> dict:
    if bound["mode"] == "aggregate":
        index, descending, nulls_first = entry
        key: Any = bound["items"][index].output_name
    elif entry[0] == "col":
        _, index, descending, nulls_first = entry
        key = schema.columns[index].name
    else:  # "expr" sort key (a SELECT alias)
        _, expr, descending, nulls_first = entry
        key = _expr_json(expr)
    return {
        "key": key,
        "direction": "DESC" if descending else "ASC",
        "nulls": "FIRST" if nulls_first else "LAST",
    }


def _project_json(item, schema: Schema) -> dict:
    if item.kind == "column":
        expression = {"kind": "column", "name": item.output_name}
    elif item.kind == "expr":
        expression = _expr_json(item.expr)
    else:  # "agg"
        expression = {
            "kind": "aggregate",
            "function": item.func,
            "argument": (
                schema.columns[item.arg_index].name if item.arg_index >= 0 else None
            ),
        }
    return {"expression": expression, "name": item.output_name}


# ---------------------------------------------------------------------------
# Plan assembly
# ---------------------------------------------------------------------------


def _build_plan(
    sources: list,
    schema: Schema,
    select,
    bound: dict,
    join: dict | None = None,
) -> dict:
    """Assemble the ordered plan dict.

    ``sources`` is a list of ``(name, row_count, column_schemas)`` triples in
    FROM/JOIN order whose columns concatenate to ``schema``.  ``join`` carries
    ``kind``, ``left_key``/``right_key`` (qualified names) and
    ``key_indices`` (combined-schema indices of the two join keys).
    """
    required = _required_column_indices(bound)
    if join is not None:
        required.update(join["key_indices"])

    source_dicts = []
    operators = []
    offset = 0
    for name, row_count, columns in sources:
        width = len(columns)
        source_dicts.append(
            {
                "name": name,
                "row_count": row_count,
                "columns": [
                    {"name": col.name, "type": col.type, "nullable": col.nullable}
                    for col in columns
                ],
            }
        )
        operators.append(
            {
                "operator": "Scan",
                "table": name,
                "required_columns": [
                    columns[i].name for i in range(width) if offset + i in required
                ],
            }
        )
        offset += width

    if join is not None:
        operators.append(
            {
                "operator": "Join",
                "type": join["kind"],
                "left_key": join["left_key"],
                "right_key": join["right_key"],
            }
        )
    if bound["where"] is not None:
        operators.append(
            {"operator": "Filter", "condition": _expr_json(bound["where"])}
        )
    if bound["mode"] == "aggregate":
        operators.append(
            {
                "operator": "Aggregate",
                "group_by": [
                    schema.columns[i].name for i in bound["group_indices"]
                ],
                "aggregates": [
                    _agg_json(item, schema)
                    for item in bound["items"]
                    if item.kind == "agg"
                ],
            }
        )
    if bound["order_by"]:
        operators.append(
            {
                "operator": "Sort",
                "keys": [
                    _sort_key_json(entry, bound, schema)
                    for entry in bound["order_by"]
                ],
            }
        )
    if select.limit is not None:
        operators.append({"operator": "Limit", "count": select.limit})
    operators.append(
        {
            "operator": "Project",
            "expressions": [_project_json(item, schema) for item in bound["items"]],
        }
    )

    return {
        "sources": source_dicts,
        "operators": operators,
        "output": [
            {
                "name": item.output_name,
                "type": item.out_type,
                "nullable": item.nullable,
            }
            for item in bound["items"]
        ],
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def explain_file(path: Any, sql: str) -> dict:
    """Plan ``sql`` against the single columnar file ``path`` without running it.

    Accepts the same statement as :func:`~columnar_analytics.query.query_file`
    and returns the ordered, JSON-serialisable plan dict described in the
    module docstring.  The statement is parsed before the file is touched and
    only the file metadata is read; the data segment is never accessed.
    Raises :class:`QuerySyntaxError`, :class:`QueryValidationError`,
    :class:`~columnar_analytics.format.ColumnarFormatError` and
    :class:`OSError` under the same conditions as ``query_file`` (minus the
    data-level checks that require reading the data segment).
    """
    tokens = _tokenize(sql)
    select = _Parser(tokens).parse()
    schema, row_count = _read_source(path)
    bound = _bind_select(select, schema, "input")
    sources = [("input", row_count, schema.columns)]
    return _build_plan(sources, schema, select, bound)


def explain_files(sources: Any, sql: str) -> dict:
    """Plan ``sql`` against the tables named by ``sources`` without running it.

    Mirrors :func:`~columnar_analytics.query.query_files`: ``sources`` maps
    table names to columnar file paths, only the tables referenced by the
    statement are accessed, and only their metadata is read.  A non-mapping
    or empty ``sources``, non-string keys or non-path values raise
    :class:`ValueError` before anything else happens; syntax errors raise
    :class:`QuerySyntaxError` before any file is touched; unknown tables or
    columns, unqualified join references and type incompatibilities raise
    :class:`QueryValidationError`; malformed metadata raises
    :class:`~columnar_analytics.format.ColumnarFormatError`; other I/O
    failures propagate as :class:`OSError`.
    """
    paths = _validate_sources(sources)
    tokens = _tokenize(sql)
    select = _Parser(tokens, allow_join=True).parse()
    left_key = _resolve_table_ref(select.table, select.table_quoted, tuple(paths))

    if select.join is None:
        rewritten = _rewrite_select(select, _single_table_resolver(left_key))
        schema, row_count = _read_source(paths[left_key])
        bound = _bind_select(rewritten, schema, expected_table=None)
        return _build_plan(
            [(left_key, row_count, schema.columns)], schema, rewritten, bound
        )

    join = select.join
    right_key = _resolve_table_ref(join.table, join.table_quoted, tuple(paths))
    if right_key == left_key:
        raise QueryValidationError(f"duplicate table {right_key!r} in join")
    on_left = _resolve_table_ref(join.left_key[0], join.left_key[1], (left_key, right_key))
    on_right = _resolve_table_ref(join.right_key[0], join.right_key[1], (left_key, right_key))
    if on_left != left_key or on_right != right_key:
        raise QueryValidationError(
            "ON keys must reference the left and right tables respectively"
        )
    # Qualifier checks (unqualified / unknown-table column references) do
    # not need the files and run before any read.
    rewritten = _rewrite_select(select, _join_resolver(left_key, right_key))
    left_schema, left_rows = _read_source(paths[left_key])
    right_schema, right_rows = _read_source(paths[right_key])
    combined, left_idx, right_idx = _join_schema(
        left_key, left_schema, right_key, right_schema, join
    )
    bound = _bind_select(rewritten, combined, expected_table=None)
    left_width = len(left_schema.columns)
    join_info = {
        "kind": join.kind,
        "left_key": f"{left_key}.{left_schema.columns[left_idx].name}",
        "right_key": f"{right_key}.{right_schema.columns[right_idx].name}",
        "key_indices": (left_idx, left_width + right_idx),
    }
    return _build_plan(
        [
            (left_key, left_rows, left_schema.columns),
            (right_key, right_rows, right_schema.columns),
        ],
        combined,
        rewritten,
        bound,
        join=join_info,
    )
