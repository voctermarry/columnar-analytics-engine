"""Source and statement resolution between parsing and binding.

This stage owns everything that maps a parsed statement onto the actual
sources without ever reading table data: ``sources`` argument validation,
join-strategy validation (shared with the join layer), table-name
resolution, the qualified-name rewrite that canonicalises every column
reference, the joined-schema derivation shared by execution and explain,
and the metadata-to-schema helpers.  Unreferenced sources are never
opened; the only bytes read here are the 5-byte format-version prefix
peek, which treats anything unreadable as v1 so the historical read order
and error behaviour are preserved.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import QuerySyntaxError, QueryValidationError
from .format import ColumnSchema, Schema, Table
from .join import _derive_step_schema, _validate_join_strategy
from .parser import _Parser, _RefItem, _Select, _tokenize


def _peek_format_version(path: Any) -> int | None:
    """Best-effort read of a file's format version byte (no validation).

    Returns ``None`` for unreadable or unrecognisable files; callers treat
    that as "not v2" so the legacy read path reports the proper error.
    """
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(5)
    except OSError:
        return None
    if len(prefix) < 5 or prefix[:4] != b"CAEF":
        return None
    return prefix[4]
def _qualify_table(table: Table, key: str) -> Table:
    """Re-wrap ``table`` naming every column ``key.column`` (values unchanged)."""
    qualified = Schema(
        tuple(
            ColumnSchema(f"{key}.{col.name}", col.type, col.nullable)
            for col in table.schema.columns
        )
    )
    return Table._from_storage(qualified, table._columns)


def _validate_sources(sources: Any) -> dict:
    if not isinstance(sources, Mapping):
        raise ValueError("sources must be a mapping of table names to file paths")
    if not sources:
        raise ValueError("sources must not be empty")
    paths = {}
    for key, value in sources.items():
        if not isinstance(key, str) or not key:
            raise ValueError("sources keys must be non-empty strings")
        if not isinstance(value, (str, os.PathLike)):
            raise ValueError(f"sources[{key!r}] must be a file path")
        paths[key] = value
    return paths


@dataclass(frozen=True)
class _JoinStep:
    """One validated join step, normalised to intermediate-vs-new sides."""

    kind: str  # "inner" | "left" | "right" | "full"
    prior_key: str  # canonical table of the ON side already introduced
    prior_col: str
    new_key: str  # canonical table freshly introduced by this step
    new_col: str


def _resolve_statement(sources: Any, sql: str, join_strategy: Any = None) -> tuple:
    """Validate arguments and resolve the statement's table/ON references.

    No file is ever opened: only argument validation, parsing and name
    resolution run here.  Returns ``(paths, select, strategy, from_key,
    steps)`` with ``steps`` a tuple of :class:`_JoinStep` in FROM/JOIN
    order (empty for a single-table statement).
    """
    strategy = _validate_join_strategy(join_strategy)
    paths = _validate_sources(sources)
    tokens = _tokenize(sql)
    select = _Parser(tokens, allow_join=True).parse()
    all_keys = tuple(paths)
    from_key = _resolve_table_ref(select.table, select.table_quoted, all_keys)
    introduced = [from_key]
    steps: list[_JoinStep] = []
    for join in select.joins:
        new_key = _resolve_table_ref(join.table, join.table_quoted, all_keys)
        if new_key in introduced:
            raise QueryValidationError(f"duplicate table {new_key!r} in join")
        candidates = (*introduced, new_key)
        left_table = _resolve_table_ref(
            join.left_key[0], join.left_key[1], candidates
        )
        right_table = _resolve_table_ref(
            join.right_key[0], join.right_key[1], candidates
        )
        left_is_new = left_table == new_key
        right_is_new = right_table == new_key
        # Exactly one ON side introduces this step's new table; the other
        # must name a table introduced by FROM or an earlier join.
        if left_is_new == right_is_new:
            raise QueryValidationError(
                "ON keys must connect the newly joined table with an earlier table"
            )
        if left_is_new:
            prior_key, prior_col = right_table, join.right_key[2]
            new_col = join.left_key[2]
        else:
            prior_key, prior_col = left_table, join.left_key[2]
            new_col = join.right_key[2]
        steps.append(
            _JoinStep(join.kind, prior_key, prior_col, new_key, new_col)
        )
        introduced.append(new_key)
    return paths, select, strategy, from_key, tuple(steps)


def _referenced_source_paths(sources: Any, sql: str, join_strategy: Any = None) -> tuple:
    """Resolve the file paths of the tables the statement actually reads.

    Parsing-only counterpart of :func:`query_files`: no file is ever opened.
    Returns the paths in ``FROM`` / ``JOIN`` order (one entry for a
    single-table statement, one per introduced table otherwise).  Source and
    join-strategy validation raise :class:`ValueError`; lexical / grammatical
    errors raise :class:`QuerySyntaxError`; unknown or ambiguous table names
    and other join-shape problems raise :class:`QueryValidationError`.
    """
    paths, _select, _strategy, from_key, steps = _resolve_statement(
        sources, sql, join_strategy
    )
    return (paths[from_key], *(paths[step.new_key] for step in steps))


def _resolve_table_ref(name: str, quoted: bool, candidates: tuple) -> str:
    """Resolve a table reference against the available canonical names.

    Quoted references match exactly; bare references match exactly first,
    then case-insensitively when that is unambiguous.
    """
    if name in candidates:
        return name
    if not quoted:
        lowered = name.lower()
        matches = [c for c in candidates if c.lower() == lowered]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise QueryValidationError(f"ambiguous table name {name!r}")
    raise QueryValidationError(f"unknown table {name!r}")


def _single_table_resolver(table_key: str):
    def resolve(table, table_quoted, column):
        if table is not None:
            _resolve_table_ref(table, table_quoted, (table_key,))
        return column

    return resolve


def _multi_table_resolver(table_keys: tuple):
    def resolve(table, table_quoted, column):
        if table is None:
            raise QueryValidationError(
                f"column reference {column!r} must be qualified with a table name"
            )
        key = _resolve_table_ref(table, table_quoted, table_keys)
        return f"{key}.{column}"

    return resolve


def _rewrite_select(select: _Select, resolve) -> _Select:
    """Resolve every column reference to its canonical output name."""
    items = tuple(_rewrite_ref_item(item, resolve) for item in select.items)
    # ORDER BY names that match a SELECT alias stay unresolved here; the
    # binder resolves them against the aliased expressions first.
    aliases = {item.name for item in items if item.kind == "expr"}
    group_by = None
    if select.group_by is not None:
        group_by = tuple(
            (None, False, resolve(table, quoted, name))
            for table, quoted, name in select.group_by
        )
    order_by = None
    if select.order_by is not None:
        order_by = tuple(
            item
            if item.kind == "column" and item.table is None and item.name in aliases
            else _rewrite_ref_item(item, resolve)
            for item in select.order_by
        )
    where = _rewrite_expr(select.where, resolve) if select.where is not None else None
    having = (
        _rewrite_having_expr(select.having, resolve)
        if select.having is not None
        else None
    )
    return _Select(
        items=items,
        table=select.table,
        table_quoted=select.table_quoted,
        where=where,
        group_by=group_by,
        having=having,
        order_by=order_by,
        limit=select.limit,
        joins=select.joins,
        distinct=select.distinct,
    )


def _rewrite_ref_item(item: _RefItem, resolve) -> _RefItem:
    if item.kind == "star":
        return item
    if item.kind == "expr":
        return _RefItem(
            "expr",
            name=item.name,
            expr=_rewrite_expr(item.expr, resolve),
            descending=item.descending,
            nulls_first=item.nulls_first,
        )
    if item.kind == "column":
        return _RefItem(
            "column",
            name=resolve(item.table, item.table_quoted, item.name),
            descending=item.descending,
            nulls_first=item.nulls_first,
        )
    arg = item.arg
    if arg:
        arg = resolve(item.arg_table, item.arg_table_quoted, arg)
    return _RefItem(
        "agg",
        func=item.func,
        arg=arg,
        distinct=item.distinct,
        descending=item.descending,
        nulls_first=item.nulls_first,
    )


def _rewrite_expr(node: tuple, resolve) -> tuple:
    tag = node[0]
    if tag == "literal":
        return node
    if tag == "column":
        return ("column", resolve(node[2], node[3], node[1]), None, False)
    if tag == "unary":
        return ("unary", _rewrite_expr(node[1], resolve), node[2])
    if tag == "arith":
        return ("arith", node[1], _rewrite_expr(node[2], resolve), _rewrite_expr(node[3], resolve))
    if tag == "case":
        return (
            "case",
            tuple(
                (_rewrite_expr(cond, resolve), _rewrite_expr(result, resolve))
                for cond, result in node[1]
            ),
            _rewrite_expr(node[2], resolve) if node[2] is not None else None,
        )
    if tag == "not":
        return ("not", _rewrite_expr(node[1], resolve))
    if tag in ("and", "or"):
        return (tag, _rewrite_expr(node[1], resolve), _rewrite_expr(node[2], resolve))
    if tag == "isnull":
        return ("isnull", _rewrite_expr(node[1], resolve), node[2])
    if tag == "cmp":
        return ("cmp", node[1], _rewrite_expr(node[2], resolve), _rewrite_expr(node[3], resolve))
    if tag == "in":
        # Options are parser-restricted to literal constants, so the
        # operand is the only subtree carrying column references.
        return ("in", _rewrite_expr(node[1], resolve), node[2], node[3])
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


def _rewrite_having_expr(node: tuple, resolve) -> tuple:
    """Rewrite column and aggregate references inside a parsed HAVING tree."""
    tag = node[0]
    if tag == "literal":
        return node
    if tag == "column":
        return ("column", resolve(node[2], node[3], node[1]), None, False)
    if tag == "hagg":
        arg = node[2]
        if arg:
            arg = resolve(node[3], node[4], arg)
        return ("hagg", node[1], arg, None, False, node[5])
    if tag == "not":
        return ("not", _rewrite_having_expr(node[1], resolve))
    if tag in ("and", "or"):
        return (
            tag,
            _rewrite_having_expr(node[1], resolve),
            _rewrite_having_expr(node[2], resolve),
        )
    if tag == "isnull":
        return ("isnull", _rewrite_having_expr(node[1], resolve), node[2])
    if tag == "cmp":
        return (
            "cmp",
            node[1],
            _rewrite_having_expr(node[2], resolve),
            _rewrite_having_expr(node[3], resolve),
        )
    if tag == "in":
        # Options are constant literals; only the operand may carry a
        # group column or an aggregate reference.
        return ("in", _rewrite_having_expr(node[1], resolve), node[2], node[3])
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


def _build_joined_schema(
    prior_schema: Schema,
    new_schema: Schema,
    step: _JoinStep,
) -> Schema:
    """Validate one step's ON keys and derive the combined post-step schema.

    Name resolution, key-type compatibility and duplicate result names are
    query-layer concerns and stay here (keeping their
    :class:`QueryValidationError` classification); the resulting column
    layout and outer-join nullability are derived by the shared join layer
    so execution and the explain plan can never disagree.
    """
    prior_name = f"{step.prior_key}.{step.prior_col}"
    try:
        prior_idx = prior_schema.index(prior_name)
    except KeyError:
        raise QueryValidationError(f"unknown column: {prior_name!r}") from None
    try:
        new_idx = new_schema.index(step.new_col)
    except KeyError:
        raise QueryValidationError(
            f"unknown column: {f'{step.new_key}.{step.new_col}'!r}"
        ) from None
    prior_type = prior_schema.columns[prior_idx].type
    new_type = new_schema.columns[new_idx].type
    if prior_type != new_type and not {prior_type, new_type} <= {"int64", "float64"}:
        raise QueryValidationError(
            f"join key types are incompatible: {prior_type} and {new_type}"
        )

    schema = _derive_step_schema(
        prior_schema,
        new_schema,
        new_table=step.new_key,
        kind=step.kind,
    )
    names = [col.name for col in schema.columns]
    if len(set(names)) != len(names):
        raise QueryValidationError("joined tables produce duplicate column names")
    return schema
def _schema_from_metadata(metadata: Mapping) -> Schema:
    return Schema(
        tuple(
            ColumnSchema(entry["name"], entry["type"], entry["nullable"])
            for entry in metadata["columns"]
        )
    )
