"""Row-group statistics pushdown for v2 (partitioned) files.

Consumes the bound WHERE tree produced by the binder and the per-group
statistics of a v2 source, and decides which row groups can never make
the pushed condition TRUE.  Pure functions over bound expressions and
metadata mappings -- no file access happens here; the orchestration
layer hands in the inspected row groups and applies the selection.
"""

from __future__ import annotations

from collections.abc import Mapping

from .expr import _Cmp, _Column, _Expr, _IsNull, _Literal, _Logic

# ---------------------------------------------------------------------------
# Row-group statistics pushdown (v2 files)
# ---------------------------------------------------------------------------
#
# For row-group-partitioned (v2) files the AND-connected, type-compatible
# comparisons and IS [NOT] NULL conditions of a WHERE clause are evaluated
# against each group's per-column statistics.  A group is skipped only when
# its statistics prove the pushed condition can never be TRUE for any of its
# rows; anything else (OR, NOT, CASE, arithmetic, column-to-column or
# cross-source comparisons, or simply undecidable ranges) keeps the group,
# and the surviving rows are still filtered row by row with the full WHERE
# condition.
#
# In a multi-table statement only a chain made entirely of INNER JOINs is
# eligible: every source row that can reach the WHERE result must reach it
# through matches on both sides, so a condition referencing one source
# alone may prune that source's groups exactly as in the single-file case.
# Any OUTER step can emit padded rows, so no group pruning is applied to
# such chains.  Bound pushable leaves are attributed to a source through
# the combined schema's column layout, which is robust against table names
# sharing dotted prefixes.

_FLIP_CMP_OP = {"=": "=", "!=": "!=", "<": ">", "<=": ">=", ">": "<", ">=": "<="}


def _is_pushable_leaf(node: _Expr) -> bool:
    """Whether a bound WHERE leaf can be checked against column statistics."""
    if isinstance(node, _IsNull):
        return isinstance(node.operand, _Column)
    if isinstance(node, _Cmp):
        left, right = node.left, node.right
        return (isinstance(left, _Column) and isinstance(right, _Literal)) or (
            isinstance(left, _Literal) and isinstance(right, _Column)
        )
    return False


def _walk_top_and_leaves(node: _Expr, sink) -> None:
    """Feed the top-level AND-connected pushable leaves to ``sink``.

    Only the AND spine at the root is unfolded; leaves nested under OR,
    NOT or any other non-AND node are never visited.  Non-pushable leaves
    sitting directly on the AND spine are simply skipped, so they neither
    prune groups nor block the pushdown of their eligible siblings.
    """
    if isinstance(node, _Logic) and node.op == "and":
        _walk_top_and_leaves(node.left, sink)
        _walk_top_and_leaves(node.right, sink)
    elif _is_pushable_leaf(node):
        sink(node)


def _extract_pushable(where: _Expr | None) -> list:
    """The AND-connected pushable leaves of a bound WHERE tree, in order."""
    conditions: list = []
    if where is not None:
        _walk_top_and_leaves(where, conditions.append)
    return conditions


def _condition_possible(cond: _Expr, group: Mapping, col_index: int) -> bool:
    """Whether ``cond`` could be TRUE for some row of ``group``.

    Only a provable "cannot be TRUE" returns False; anything undecidable
    keeps the group.  ``col_index`` is the column's position in the
    source's local schema (the condition's bound column index is already
    local in the single-file case and translated by the caller for joins).
    """
    group_rows = group["row_count"]
    if isinstance(cond, _IsNull):
        stats = group["columns"][col_index]
        if cond.negated:  # IS NOT NULL
            return stats["null_count"] < group_rows
        return stats["null_count"] > 0
    # A column-vs-literal comparison (normalised to column OP literal).
    op = cond.op
    left, right = cond.left, cond.right
    if not isinstance(left, _Column):
        op = _FLIP_CMP_OP[op]
    stats = group["columns"][col_index]
    if stats["null_count"] == group_rows:
        # All values NULL: a comparison is never TRUE.
        return False
    literal = right.value if isinstance(left, _Column) else left.value
    minimum = stats["min"]
    maximum = stats["max"]
    if op == "=":
        return minimum <= literal <= maximum
    if op == "!=":
        return not (minimum == maximum == literal)
    if op == "<":
        return minimum < literal
    if op == "<=":
        return minimum <= literal
    if op == ">":
        return maximum > literal
    return maximum >= literal  # ">="


def _leaf_col_index(cond: _Expr) -> int:
    """The bound column index carried by one pushable leaf."""
    if isinstance(cond, _IsNull):
        return cond.operand.index
    column = cond.left if isinstance(cond.left, _Column) else cond.right
    return column.index


def _select_row_groups(groups: list, pushed: list, col_indices: list) -> list[int]:
    """Indices of the row groups whose statistics do not rule them out.

    ``col_indices`` parallels ``pushed``: the local column position of
    each pushed leaf in this source's schema.
    """
    if not pushed:
        return list(range(len(groups)))
    return [
        index
        for index, group in enumerate(groups)
        if all(
            _condition_possible(cond, group, col_index)
            for cond, col_index in zip(pushed, col_indices)
        )
    ]


def _pushed_col_indices(pushed: list) -> list[int]:
    """The bound column index carried by each pushable leaf."""
    return [_leaf_col_index(cond) for cond in pushed]


def _pushed_condition_json(pushed: list) -> dict | None:
    """Render the pushed-down condition set as one condition tree (or null)."""
    if not pushed:
        return None
    node: _Expr = pushed[0]
    for cond in pushed[1:]:
        node = _Logic("and", node, cond)
    return node.to_json()


def _source_column_spans(schemas: Mapping, table_keys: tuple) -> dict:
    """Each source's ``[start, end)`` column-index range in the combined schema.

    The combined schema concatenates the sources' own columns in
    FROM/JOIN order, so the ranges are contiguous and non-overlapping.
    """
    spans: dict = {}
    start = 0
    for key in table_keys:
        width = len(schemas[key].columns)
        spans[key] = (start, start + width)
        start += width
    return spans


def _pushable_leaves_by_source(
    bound_where: tuple | None, spans: Mapping, table_keys: tuple
) -> dict:
    """Top-level AND pushable leaves attributed to one source each.

    A leaf's bound column index falls into exactly one source's combined
    schema range; the leaves keep their SQL appearance order within each
    source.  Column-to-column comparisons never reach this point (they are
    not pushable), so no leaf can reference two sources.
    """
    by_source: dict = {key: [] for key in table_keys}

    def sink(node: tuple) -> None:
        index = _leaf_col_index(node)
        for key in table_keys:
            start, end = spans[key]
            if start <= index < end:
                by_source[key].append(node)
                return

    if bound_where is not None:
        _walk_top_and_leaves(bound_where, sink)
    return by_source


def _source_scan_info(groups: list, leaves: list, span: tuple) -> dict:
    """One v2 Scan's pushdown state.

    ``leaves`` are the bound leaves attributed to this source; ``span`` is
    its combined-schema range, so global bound column indices translate to
    the source's local block positions.  The returned dict carries the
    selected group indices (for the reader) plus the total count and the
    pushed leaves (for the explain plan); :func:`_scan_pushdown_fields`
    projects its JSON-visible fields.
    """
    start = span[0]
    local_indices = [_leaf_col_index(cond) - start for cond in leaves]
    selected = _select_row_groups(groups, leaves, local_indices)
    return {
        "groups_total": len(groups),
        "selected_groups": selected,
        "pushed_leaves": leaves,
    }


def _scan_pushdown_fields(info: Mapping) -> dict:
    """The three JSON fields appended to an eligible v2 Scan operator."""
    leaves = info["pushed_leaves"]
    return {
        "row_groups_total": info["groups_total"],
        "row_groups_selected": len(info["selected_groups"]),
        "pushed_condition": _pushed_condition_json(leaves),
    }


def _scan_pushdown_info(groups: list, bound: Mapping, schema: Schema) -> dict:
    """The single-source v2 Scan statistics (the span starts at zero)."""
    pushed = _extract_pushable(bound["where"])
    return _source_scan_info(groups, pushed, (0, len(schema.columns)))
