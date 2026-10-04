"""Binding: resolve a parsed statement against one schema.

The binder consumes the parsed :class:`~columnar_analytics.parser._Select`
tree and a :class:`~columnar_analytics.format.Schema` (single-file,
combined join or metadata-derived -- all the same kind) and produces the
bound plan dict: validated projection items, the bound WHERE / HAVING
expression IR, ORDER BY keys and the aggregate registry.  It performs no
file access of its own, so a statement can be bound against metadata alone
(the explain path) or against a fully read table (the execution path) with
identical results and identical
:class:`~columnar_analytics.errors.QueryValidationError` behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import QuerySyntaxError, QueryValidationError
from .expr import (
    _Aggregate,
    _Case,
    _Column,
    _Cmp,
    _Expr,
    _In,
    _IsNull,
    _Literal,
    _Logic,
    _Not,
    _bind_expr,
    _bind_in_options,
    _check_comparison_types,
    _require_boolean,
)
from .format import ColumnSchema, Schema
from .parser import _RefItem, _Select


# ---------------------------------------------------------------------------
# Binding: schema resolution, projection checks and type compatibility
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BoundItem:
    """A validated projection element."""

    kind: str  # "column" | "agg" | "expr"
    output_name: str
    # kind == "column":
    col_index: int = -1
    # kind == "agg":
    func: str = ""
    arg_index: int = -1  # -1 means COUNT(*)
    arg_type: str = ""
    distinct: bool = False  # argument values are deduplicated per group
    # kind == "expr":
    expr: tuple | None = None  # bound scalar expression
    out_type: str = ""
    nullable: bool = True


def _bind_select(select: _Select, schema: Schema, expected_table="input") -> dict:
    if expected_table is not None:
        table_matches = (
            select.table == expected_table
            if select.table_quoted
            else select.table.lower() == expected_table.lower()
        )
        if not table_matches:
            raise QueryValidationError(
                f"unknown table {select.table!r}; only {expected_table!r} is supported"
            )

    where = _bind_expr(select.where, schema) if select.where is not None else None
    if where is not None and where.type != "bool":
        raise QueryValidationError(
            f"WHERE clause must be boolean, got {where.type}"
        )

    if select.distinct and select.is_aggregate_query:
        raise QueryValidationError(
            "DISTINCT is not allowed with GROUP BY, HAVING or aggregate projections"
        )

    if not select.is_aggregate_query:
        return _bind_plain(select, schema, where)
    return _bind_aggregate(select, schema, where)


def _bind_plain(select: _Select, schema: Schema, where) -> dict:
    # Non-aggregate path: '*' or a comma-separated list of plain columns and
    # computed scalar expressions (each named with AS).
    bound_items: list[_BoundItem] = []
    if select.star:
        for i, col in enumerate(schema.columns):
            bound_items.append(
                _BoundItem(
                    "column",
                    output_name=col.name,
                    col_index=i,
                    out_type=col.type,
                    nullable=col.nullable,
                )
            )
    else:
        output_seen: set[str] = set()
        for item in select.items:
            if item.kind == "star":
                # A star mixed into a plain projection remains grammatical.
                raise QuerySyntaxError("'*' cannot be mixed with other projection items")
            if item.kind == "column":
                name = item.name
                if name in output_seen:
                    raise QueryValidationError(
                        f"duplicate column in projection: {name!r}"
                    )
                try:
                    col_index = schema.index(name)
                except KeyError:
                    raise QueryValidationError(f"unknown column: {name!r}") from None
                col = schema.columns[col_index]
                output_seen.add(col.name)
                bound_items.append(
                    _BoundItem(
                        "column",
                        output_name=col.name,
                        col_index=col_index,
                        out_type=col.type,
                        nullable=col.nullable,
                    )
                )
            else:  # "expr"
                expr = _bind_expr(item.expr, schema)
                out_type = expr.type
                # Arithmetic scalar expressions stay numeric-only; a CASE
                # expression may additionally yield bool or utf8 results
                # (its own WHEN/ELSE type consistency is checked at binding).
                if isinstance(expr, _Case):
                    allowed_types = ("int64", "float64", "bool", "utf8")
                else:
                    allowed_types = ("int64", "float64")
                if out_type not in allowed_types:
                    raise QueryValidationError(
                        f"SELECT expression must be numeric, got {out_type}"
                    )
                alias = item.name
                if alias in output_seen:
                    raise QueryValidationError(f"duplicate result column: {alias!r}")
                output_seen.add(alias)
                bound_items.append(
                    _BoundItem(
                        "expr",
                        output_name=alias,
                        expr=expr,
                        out_type=out_type,
                        nullable=expr.nullable,
                    )
                )
    aliases = {
        item.output_name: item.expr for item in bound_items if item.kind == "expr"
    }
    projected = {item.output_name: i for i, item in enumerate(bound_items)}
    order_by = _bind_plain_order_by(
        select.order_by, schema, aliases, select.distinct, projected
    )
    return {
        "mode": "plain",
        "items": tuple(bound_items),
        "aggregates": (),
        "where": where,
        "having": None,
        "order_by": order_by,
        "distinct": select.distinct,
    }


def _bind_plain_order_by(
    select_order_by, schema: Schema, aliases: dict, distinct: bool, projected: dict
):
    if select_order_by is None:
        return None
    bound: list[tuple] = []
    order_seen: set[str] = set()
    for item in select_order_by:
        if item.kind == "agg":
            raise QueryValidationError(
                f"aggregate {item.func}({_arg_label(item)}) may appear in ORDER BY "
                "only as part of an aggregate query"
            )
        name = item.name
        if name in order_seen:
            raise QueryValidationError(
                f"duplicate column in ORDER BY: {name!r}"
            )
        order_seen.add(name)
        nulls_first = item.nulls_first if item.nulls_first is not None else False
        if distinct:
            # A DISTINCT query sorts its deduplicated result rows, so every
            # ORDER BY name must be one of the projected outputs (a bare
            # projected column or an explicit AS alias).
            if name not in projected:
                try:
                    schema.index(name)
                except KeyError:
                    raise QueryValidationError(
                        f"unknown ORDER BY column: {name!r}"
                    ) from None
                raise QueryValidationError(
                    f"ORDER BY column {name!r} is not part of the SELECT DISTINCT results"
                )
            bound.append(("out", projected[name], item.descending, nulls_first))
            continue
        # A SELECT alias shadows an input column of the same name.
        if name in aliases:
            bound.append(("expr", aliases[name], item.descending, nulls_first))
            continue
        try:
            col_index = schema.index(name)
        except KeyError:
            raise QueryValidationError(
                f"unknown ORDER BY column: {name!r}"
            ) from None
        bound.append(("col", col_index, item.descending, nulls_first))
    return tuple(bound)


def _bind_aggregate(select: _Select, schema: Schema, where) -> dict:
    if select.star:
        raise QueryValidationError("'*' cannot be combined with aggregates or GROUP BY")

    if (
        select.having is not None
        and select.group_by is None
        and not select.has_aggregate
        and not _having_tree_has_aggregate(select.having)
    ):
        # HAVING filters formed groups; without aggregation or GROUP BY the
        # statement stays a non-aggregate query and HAVING has no meaning.
        raise QueryValidationError(
            "HAVING is only valid with GROUP BY or an aggregate"
        )

    # Resolve GROUP BY columns first: order matters for the grouping key, and
    # duplicates / unknown columns are rejected here.  Entries are
    # (table, table_quoted, name) triples; names reaching the binder are
    # already canonical (single-file queries never carry a qualifier).
    group_indices: list[int] = []
    group_seen: set[str] = set()
    for _table, _table_quoted, name in select.group_by or ():
        if name in group_seen:
            raise QueryValidationError(f"duplicate column in GROUP BY: {name!r}")
        group_seen.add(name)
        try:
            group_indices.append(schema.index(name))
        except KeyError:
            raise QueryValidationError(f"unknown GROUP BY column: {name!r}") from None
    group_index_set = set(group_indices)

    # One registry of distinct aggregates shared by SELECT, HAVING and ORDER
    # BY, keyed by (function, argument index, DISTINCT flag) and kept in
    # first-reference order.  HAVING-only aggregates are computed but never
    # projected.
    agg_registry: dict[tuple, int] = {}
    agg_order: list[_BoundItem] = []

    def require_aggregate(func: str, arg_name: str, distinct: bool = False) -> int:
        arg_index, arg_col, label, out_type, nullable = _resolve_agg_call(
            func, arg_name, schema, distinct
        )
        key = (func, arg_index, distinct)
        slot = agg_registry.get(key)
        if slot is None:
            slot = len(agg_order)
            agg_registry[key] = slot
            agg_order.append(
                _BoundItem(
                    "agg",
                    output_name=label,
                    func=func,
                    arg_index=arg_index,
                    arg_type=arg_col.type if arg_col is not None else "",
                    distinct=distinct,
                    out_type=out_type,
                    nullable=nullable,
                )
            )
        return slot

    # Resolve the projection.  Plain columns must be grouped; output names
    # (group column names and canonical "FUNC(arg)" labels) must be unique.
    bound_items: list[_BoundItem] = []
    output_seen: set[str] = set()
    has_plain = False
    for item in select.items:
        if item.kind == "star":
            raise QueryValidationError(
                "'*' cannot be combined with aggregates or GROUP BY"
            )
        if item.kind == "expr":
            raise QueryValidationError(
                "scalar expressions are not supported in aggregate queries"
            )
        if item.kind == "column":
            has_plain = True
            name = item.name
            try:
                col_index = schema.index(name)
            except KeyError:
                raise QueryValidationError(f"unknown column: {name!r}") from None
            if col_index not in group_index_set:
                raise QueryValidationError(
                    f"column {name!r} must appear in GROUP BY or be wrapped in an aggregate"
                )
            label = name
            if label in output_seen:
                raise QueryValidationError(f"duplicate result column: {name!r}")
            output_seen.add(label)
            col = schema.columns[col_index]
            bound_items.append(
                _BoundItem(
                    "column",
                    output_name=label,
                    col_index=col_index,
                    out_type=col.type,
                    nullable=col.nullable,
                )
            )
        else:
            slot = require_aggregate(item.func, item.arg, item.distinct)
            agg_item = agg_order[slot]
            if agg_item.output_name in output_seen:
                raise QueryValidationError(
                    f"duplicate result column: {agg_item.output_name!r}"
                )
            output_seen.add(agg_item.output_name)
            bound_items.append(agg_item)

    if select.group_by is None and has_plain:
        # No grouping: every projected column must be an aggregate.
        raise QueryValidationError(
            "without GROUP BY, the projection may contain aggregates only"
        )

    # HAVING filters the formed groups; it may reuse projected aggregates and
    # may introduce further aggregates that never reach the projection.
    having = None
    if select.having is not None:
        having = _bind_having(
            select.having, schema, group_index_set, agg_order, require_aggregate
        )
        _require_boolean(having, "HAVING clause")

    order_by = _bind_aggregate_order_by(
        select.order_by, bound_items, schema
    )
    return {
        "mode": "aggregate",
        "group_indices": tuple(group_indices),
        "items": tuple(bound_items),
        "aggregates": tuple(agg_order),
        "where": where,
        "having": having,
        "order_by": order_by,
        "distinct": False,
    }


def _having_tree_has_aggregate(node: tuple) -> bool:
    """Whether a parsed (unbound) HAVING tree names at least one aggregate."""
    tag = node[0]
    if tag == "hagg":
        return True
    if tag in ("literal", "column"):
        return False
    if tag in ("not", "isnull"):
        return _having_tree_has_aggregate(node[1])
    if tag == "cmp":
        return _having_tree_has_aggregate(node[2]) or _having_tree_has_aggregate(
            node[3]
        )
    if tag == "in":
        # The option list holds literals only; only the operand can be an aggregate.
        return _having_tree_has_aggregate(node[1])
    return _having_tree_has_aggregate(node[1]) or _having_tree_has_aggregate(node[2])


def _bind_having(
    node: tuple,
    schema: Schema,
    group_index_set: set[int],
    agg_order: list[_BoundItem],
    require_aggregate,
) -> _Expr:
    tag = node[0]
    if tag == "literal":
        return _Literal(node[1], node[2])
    if tag == "column":
        name = node[1]
        try:
            col_index = schema.index(name)
        except KeyError:
            raise QueryValidationError(f"unknown column: {name!r}") from None
        if col_index not in group_index_set:
            raise QueryValidationError(
                f"HAVING column {name!r} must appear in GROUP BY or be wrapped in an aggregate"
            )
        col = schema.columns[col_index]
        return _Column(name, col_index, col.type, col.nullable)
    if tag == "hagg":
        slot = require_aggregate(node[1], node[2], node[5])
        agg_item = agg_order[slot]
        arg_name = (
            None
            if agg_item.arg_index < 0
            else schema.columns[agg_item.arg_index].name
        )
        return _Aggregate(
            agg_item.func,
            slot,
            arg_name,
            agg_item.distinct,
            agg_item.out_type,
            agg_item.nullable,
        )
    if tag == "not":
        operand = _bind_having(
            node[1], schema, group_index_set, agg_order, require_aggregate
        )
        _require_boolean(operand, "NOT")
        return _Not(operand)
    if tag in ("and", "or"):
        left = _bind_having(
            node[1], schema, group_index_set, agg_order, require_aggregate
        )
        right = _bind_having(
            node[2], schema, group_index_set, agg_order, require_aggregate
        )
        _require_boolean(left, tag.upper())
        _require_boolean(right, tag.upper())
        return _Logic(tag, left, right)
    if tag == "isnull":
        operand = _bind_having(
            node[1], schema, group_index_set, agg_order, require_aggregate
        )
        if not operand.is_having_leaf:
            raise QuerySyntaxError(
                "IS NULL operand must be a group column, an aggregate or a literal"
            )
        return _IsNull(operand, node[2])
    if tag == "cmp":
        op = node[1]
        left = _bind_having(
            node[2], schema, group_index_set, agg_order, require_aggregate
        )
        right = _bind_having(
            node[3], schema, group_index_set, agg_order, require_aggregate
        )
        if not (left.is_having_leaf and right.is_having_leaf):
            raise QuerySyntaxError(
                "comparison operands must be group columns, aggregates or literals"
            )
        _check_comparison_types(op, left, right)
        return _Cmp(op, left, right)
    if tag == "in":
        operand = _bind_having(
            node[1], schema, group_index_set, agg_order, require_aggregate
        )
        if not operand.is_having_leaf:
            raise QuerySyntaxError(
                "the IN operand must be a group column, an aggregate or a literal"
            )
        options, nulls = _bind_in_options(node[2], operand)
        return _In(operand, options, node[3], nulls)
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


def _agg_result_type(func: str, arg_col: ColumnSchema | None) -> tuple[str, bool]:
    """Static (type, nullable) of an aggregate call; rejects illegal args."""
    if func == "COUNT":
        return "int64", False
    if func == "SUM":
        if arg_col.type not in ("int64", "float64"):
            raise QueryValidationError(
                f"SUM requires an int64 or float64 argument, got {arg_col.type}"
            )
        return arg_col.type, True
    if func == "AVG":
        if arg_col.type not in ("int64", "float64"):
            raise QueryValidationError(
                f"AVG requires an int64 or float64 argument, got {arg_col.type}"
            )
        return "float64", True
    # MIN / MAX accept all four existing types.
    return arg_col.type, True


def _resolve_agg_call(
    func: str, arg_name: str, schema: Schema, distinct: bool = False
) -> tuple[int, ColumnSchema | None, str, str, bool]:
    """Resolve one aggregate call to (arg_index, arg_col|None, label, type, nullable)."""
    if arg_name == "":
        if func != "COUNT":
            raise QuerySyntaxError(f"{func} does not accept '*'")
        out_type, nullable = _agg_result_type("COUNT", None)
        return -1, None, "COUNT(*)", out_type, nullable
    try:
        arg_index = schema.index(arg_name)
    except KeyError:
        raise QueryValidationError(f"unknown column: {arg_name!r}") from None
    arg_col = schema.columns[arg_index]
    out_type, nullable = _agg_result_type(func, arg_col)
    # The result label uses the column name as spelled in the schema.
    if distinct:
        return (
            arg_index,
            arg_col,
            f"{func}(DISTINCT {arg_col.name})",
            out_type,
            nullable,
        )
    return arg_index, arg_col, f"{func}({arg_col.name})", out_type, nullable


def _arg_label(item: _RefItem) -> str:
    return "*" if item.arg == "" else item.arg


def _bind_aggregate_order_by(
    select_order_by,
    bound_items: list[_BoundItem],
    schema: Schema,
):
    if select_order_by is None:
        return None

    # Map every SELECT result to its output position.  ORDER BY in an
    # aggregate query may only name those selected results.
    selected: dict[tuple, int] = {}
    for i, bound in enumerate(bound_items):
        if bound.kind == "column":
            selected[("column", schema.columns[bound.col_index].name)] = i
        else:
            arg_key = "*" if bound.arg_index == -1 else schema.columns[bound.arg_index].name
            selected[("agg", f"{bound.func}|{arg_key}|{bound.distinct}")] = i

    bound_order: list[tuple[int, bool, bool]] = []
    order_seen: set[str] = set()
    for item in select_order_by:
        if item.kind == "column":
            key = ("column", item.name)
            label = item.name
            if key not in selected:
                # Keep the same "unknown column" wording for names the schema
                # does not know at all; anything else is an unselected result.
                try:
                    schema.index(item.name)
                except KeyError:
                    raise QueryValidationError(
                        f"unknown ORDER BY column: {item.name!r}"
                    ) from None
                raise QueryValidationError(
                    f"ORDER BY column {item.name!r} is not part of the selected results"
                )
        else:
            if item.arg == "":
                arg_key = "*"
                label = "COUNT(*)"
            else:
                try:
                    real_name = schema.columns[schema.index(item.arg)].name
                except KeyError:
                    raise QueryValidationError(
                        f"unknown ORDER BY column: {item.arg!r}"
                    ) from None
                arg_key = real_name
                if item.distinct:
                    label = f"{item.func}(DISTINCT {real_name})"
                else:
                    label = f"{item.func}({real_name})"
            key = ("agg", f"{item.func}|{arg_key}|{item.distinct}")
            if key not in selected:
                raise QueryValidationError(
                    f"ORDER BY aggregate {label} is not part of the selected results"
                )
        if label in order_seen:
            raise QueryValidationError(f"duplicate column in ORDER BY: {label!r}")
        order_seen.add(label)
        nulls_first = item.nulls_first if item.nulls_first is not None else False
        bound_order.append((selected[key], item.descending, nulls_first))
    return tuple(bound_order)
def _collect_expr_columns(node: _Expr, indices: set) -> None:
    """Add every source-column index referenced by one bound expression.

    One traversal for WHERE, SELECT expressions, ORDER BY aliases and the
    HAVING tree: the node hierarchy itself owns the walk.  HAVING
    aggregate leaves reference no source column here -- their arguments
    are scanned separately through the aggregate registry.
    """
    indices.update(node.columns())


def _collect_required_indices(bound: Mapping) -> set:
    """Combined-schema column indices referenced anywhere in the query."""
    indices: set = set(bound.get("group_indices", ()))
    if bound["where"] is not None:
        _collect_expr_columns(bound["where"], indices)
    for item in bound["items"]:
        if item.kind == "column":
            indices.add(item.col_index)
        elif item.kind == "agg":
            if item.arg_index >= 0:
                indices.add(item.arg_index)
        else:  # "expr"
            _collect_expr_columns(item.expr, indices)
    # Aggregates introduced only by HAVING still have to be scanned.
    for item in bound.get("aggregates", ()):
        if item.arg_index >= 0:
            indices.add(item.arg_index)
    if bound.get("having") is not None:
        _collect_expr_columns(bound["having"], indices)
    for entry in bound["order_by"] or ():
        if entry[0] == "col":
            indices.add(entry[1])
        elif entry[0] == "expr":
            _collect_expr_columns(entry[1], indices)
        # Aggregate ORDER BY entries index a selected output column, which
        # is already covered by the projection scan above.
    return indices
