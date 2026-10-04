"""Explain-plan construction.

Renders the bound statement -- the very same bound dict the executor runs
-- as the JSON-serialisable operator tree: Scan operators per referenced
source (with the v2 pushdown fields where eligible), one Join operator per
step, then Filter / Aggregate / Having / Project / Distinct / Sort / Limit
exactly as execution orders them.  Nothing here reads files; the caller
supplies the source metadata it already inspected.
"""

from __future__ import annotations

from collections.abc import Mapping

from .binder import _collect_required_indices
from .expr import _Expr
from .format import Schema
from .join import _strategy_label
from .parser import _Select
from .resolve import _JoinStep


def _source_description(key: str, metadata: Mapping) -> dict:
    return {
        "name": key,
        "row_count": metadata["row_count"],
        "columns": [
            {
                "name": entry["name"],
                "type": entry["type"],
                "nullable": entry["nullable"],
            }
            for entry in metadata["columns"]
        ],
    }
def _expr_json(node: _Expr) -> dict:
    """Render a bound expression as its explain-plan JSON tree.

    Every bound node -- WHERE conditions, SELECT expressions and the HAVING
    tree including its aggregate leaves -- owns its rendering via
    :meth:`_Expr.to_json`, so plan JSON and the executable expression can
    never disagree.
    """
    return node.to_json()


def _aggregate_json(item, schema: Schema) -> dict:
    """Render one bound aggregate of the Aggregate operator as JSON.

    DISTINCT aggregates gain a ``distinct: true`` field; plain aggregates
    keep the historical three-field shape.
    """
    entry = {
        "function": item.func,
        "argument": (
            None if item.arg_index < 0 else schema.columns[item.arg_index].name
        ),
        "output": item.output_name,
    }
    if item.distinct:
        entry["distinct"] = True
    return entry


def _build_explain(
    sources: tuple,
    schema: Schema,
    select: _Select,
    bound: Mapping,
    steps: tuple[_JoinStep, ...] = (),
    strategy: str | None = None,
    scan_extras: Mapping | None = None,
) -> dict:
    referenced = _collect_required_indices(bound)
    for step in steps:
        # The ON keys feed the join even when neither is projected.
        referenced.add(schema.index(f"{step.prior_key}.{step.prior_col}"))
        referenced.add(schema.index(f"{step.new_key}.{step.new_col}"))

    operators: list = []
    referenced_names = {schema.columns[i].name for i in referenced}
    for key, _metadata, source_schema in sources:
        if not steps:
            required = [col.name for col in schema.columns if col.name in referenced_names]
        else:
            prefix = f"{key}."
            required = [
                col.name
                for col in source_schema.columns
                if f"{prefix}{col.name}" in referenced_names
            ]
        scan_operator = {"operator": "Scan", "source": key, "required_columns": required}
        if scan_extras is not None and key in scan_extras:
            # An eligible v2 (row-group-partitioned) scan reports the
            # statistics pushdown right after required_columns: total /
            # selected row groups and the pushed condition tree (null when
            # nothing was pushed).  v1 scans and scans of chains containing
            # an OUTER step keep their historical three-field shape.
            scan_operator.update(scan_extras[key])
        operators.append(scan_operator)

    # One Join operator per step, in FROM/JOIN order; each reports the two
    # qualified ON keys normalised to prior-table (intermediate) left and
    # freshly introduced table right.  An explicitly requested strategy is
    # reported on every Join operator; the implicit path (join_strategy
    # omitted) keeps the operator shape with no strategy field.
    for step in steps:
        join_operator = {
            "operator": "Join",
            "type": step.kind.upper(),
            "left": {"table": step.prior_key, "column": step.prior_col},
            "right": {"table": step.new_key, "column": step.new_col},
        }
        if strategy is not None:
            join_operator["strategy"] = _strategy_label(strategy)
        operators.append(join_operator)

    if bound["where"] is not None:
        operators.append({"operator": "Filter", "condition": _expr_json(bound["where"])})

    if bound["mode"] == "aggregate":
        # The Aggregate node computes every distinct aggregate the SELECT,
        # ORDER BY or HAVING needs, in first-reference order.  Only
        # DISTINCT aggregates carry a "distinct" field; plain aggregate
        # entries keep their historical shape.
        operators.append(
            {
                "operator": "Aggregate",
                "group_keys": [
                    schema.columns[index].name for index in bound["group_indices"]
                ],
                "aggregates": [
                    _aggregate_json(item, schema)
                    for item in bound["aggregates"]
                ],
            }
        )
        if bound["having"] is not None:
            operators.append(
                {
                    "operator": "Having",
                    "condition": _expr_json(bound["having"]),
                }
            )

    projections = []
    for item in bound["items"]:
        if item.kind == "column":
            expression = {"kind": "column", "name": schema.columns[item.col_index].name}
        elif item.kind == "agg":
            expression = {
                "kind": "aggregate",
                "function": item.func,
                "argument": (
                    None
                    if item.arg_index < 0
                    else schema.columns[item.arg_index].name
                ),
            }
            if item.distinct:
                expression["distinct"] = True
        else:  # "expr"
            expression = _expr_json(item.expr)
        projections.append({"expression": expression, "output": item.output_name})
    project_operator = {"operator": "Project", "expressions": projections}

    if bound.get("distinct"):
        # DISTINCT evaluates the projection right after the scan/join/filter
        # stages, then deduplicates the result rows before Sort and Limit.
        operators.append(project_operator)
        operators.append(
            {
                "operator": "Distinct",
                "keys": [item.output_name for item in bound["items"]],
            }
        )
        if bound["order_by"] is not None:
            keys = []
            for entry in bound["order_by"]:
                # DISTINCT ORDER BY entries index projected output columns.
                name = bound["items"][entry[1]].output_name
                keys.append(
                    {
                        "column": name,
                        "direction": "DESC" if entry[2] else "ASC",
                        "nulls": "FIRST" if entry[3] else "LAST",
                    }
                )
            operators.append({"operator": "Sort", "keys": keys})
        if select.limit is not None:
            operators.append({"operator": "Limit", "count": select.limit})
    else:
        if bound["order_by"] is not None:
            alias_by_expr = {
                id(item.expr): item.output_name
                for item in bound["items"]
                if item.kind == "expr"
            }
            keys = []
            for entry in bound["order_by"]:
                if entry[0] == "col":
                    name = schema.columns[entry[1]].name
                    descending, nulls_first = entry[2], entry[3]
                elif entry[0] == "expr":
                    name = alias_by_expr[id(entry[1])]
                    descending, nulls_first = entry[2], entry[3]
                else:  # aggregate-query ORDER BY indexes a selected output column
                    name = bound["items"][entry[0]].output_name
                    descending, nulls_first = entry[1], entry[2]
                keys.append(
                    {
                        "column": name,
                        "direction": "DESC" if descending else "ASC",
                        "nulls": "FIRST" if nulls_first else "LAST",
                    }
                )
            operators.append({"operator": "Sort", "keys": keys})

        if select.limit is not None:
            operators.append({"operator": "Limit", "count": select.limit})
        operators.append(project_operator)

    output = [
        {
            "name": item.output_name,
            "type": item.out_type,
            "nullable": item.nullable,
        }
        for item in bound["items"]
    ]

    return {
        "sources": [_source_description(key, metadata) for key, metadata, _ in sources],
        "operators": operators,
        "output": output,
    }
