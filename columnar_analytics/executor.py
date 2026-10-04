"""Execution stage: run a bound statement over materialised tables.

Given a :class:`~columnar_analytics.format.Table` and a parsed
statement, binds it (through :mod:`columnar_analytics.binder`) and runs
the row-processing stages -- WHERE, grouping / aggregation / HAVING,
projection, DISTINCT deduplication, stable sorting and LIMIT -- plus the
join-step assembly shared with the explain path.  No file access happens
here; the orchestration layer hands in fully read tables.
"""

from __future__ import annotations

import math
from functools import cmp_to_key

from .binder import _bind_select
from .errors import QueryValidationError
from .expr import _eval_expr_selection, _filter_rows, _iter_batches, _row_values
from .format import ColumnSchema, Schema, Table
from .join import _derive_step_schema, _execute_join
from .parser import _INT64_MAX, _INT64_MIN

# ---------------------------------------------------------------------------
# Aggregate computation
# ---------------------------------------------------------------------------


def _aggregate_value(
    func: str,
    arg_index: int,
    arg_type: str,
    rows,
    source_columns,
    distinct: bool = False,
):
    if func == "COUNT":
        if arg_index == -1:
            return len(rows)
        col = source_columns[arg_index]
        if not distinct:
            return sum(1 for i in rows if col[i] is not None)
        return len(_distinct_values(col, arg_type, rows))

    col = source_columns[arg_index]
    if distinct:
        values = _distinct_values(col, arg_type, rows)
    else:
        values = [col[i] for i in rows if col[i] is not None]
    if not values:
        return None

    if func == "MIN":
        return min(values)
    if func == "MAX":
        return max(values)
    if func == "SUM":
        if arg_type == "int64":
            # Python integers are unbounded; the int64 result is rejected only
            # when the final total leaves the int64 range.
            total = sum(values)
            if not (_INT64_MIN <= total <= _INT64_MAX):
                raise QueryValidationError("SUM overflowed the int64 range")
            return total
        try:
            total = math.fsum(values)
        except OverflowError:
            raise QueryValidationError("SUM produced a non-finite float64 value") from None
        if not math.isfinite(total):
            raise QueryValidationError("SUM produced a non-finite float64 value")
        return total
    # AVG
    try:
        result = math.fsum(values) / len(values)
    except OverflowError:
        raise QueryValidationError("AVG produced a non-finite float64 value") from None
    if not math.isfinite(result):
        raise QueryValidationError("AVG produced a non-finite float64 value")
    return result


def _distinct_values(col, arg_type: str, rows) -> list:
    """The distinct non-NULL values of ``col`` over ``rows``, first-seen order.

    Equality follows the column's type (the same rule as SELECT DISTINCT):
    float64 0.0 and -0.0 are the same value.  The first occurrence of each
    distinct value is kept as its representative.
    """
    seen: set = set()
    values: list = []
    for i in rows:
        value = col[i]
        if value is None:
            continue
        key = _distinct_key_part(arg_type, value)
        if key not in seen:
            seen.add(key)
            values.append(value)
    return values


def _qualify_table(table: Table, key: str) -> Table:
    """Re-wrap ``table`` naming every column ``key.column`` (values unchanged)."""
    qualified = Schema(
        tuple(
            ColumnSchema(f"{key}.{col.name}", col.type, col.nullable)
            for col in table.schema.columns
        )
    )
    return Table._from_storage(qualified, table._columns)


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


def _execute_join_step(
    prior: Table,
    new: Table,
    step: _JoinStep,
    strategy: str | None = None,
) -> Table:
    """Execute one validated join step through the shared join layer.

    The combined schema (key existence, key-type compatibility and
    nullability) is derived exactly as the explain plan derives it; the
    selected algorithm only produces the match relation, while expansion
    order, outer NULL padding and result assembly are shared.
    """
    schema = _build_joined_schema(prior.schema, new.schema, step)
    prior_idx = prior.schema.index(f"{step.prior_key}.{step.prior_col}")
    new_idx = new.schema.index(step.new_col)
    return _execute_join(
        prior,
        new,
        prior_idx,
        new_idx,
        schema,
        kind=step.kind,
        strategy=strategy,
    )


def _run_query(table: Table, select: _Select, expected_table="input") -> Table:
    bound = _bind_select(select, table.schema, expected_table)
    source_columns = table._columns
    row_count = table.row_count
    where = bound["where"]

    # The WHERE filter runs batched over the source columns; only rows
    # whose condition is TRUE reach the later stages.
    if where is None:
        selected = list(range(row_count))
    else:
        selected = _filter_rows(where, source_columns, row_count)

    if bound["mode"] == "plain":
        return _run_plain(table, select, bound, selected)
    return _run_aggregate(table, select, bound, selected)


def _run_plain(table: Table, select: _Select, bound, selected: list[int]) -> Table:
    items = bound["items"]
    order_by = bound["order_by"]
    source_columns = table._columns

    out_schema_columns = [
        table.schema.columns[item.col_index]
        if item.kind == "column"
        else ColumnSchema(item.output_name, item.out_type, nullable=item.nullable)
        for item in items
    ]
    out_schema = Schema(out_schema_columns)

    if not bound.get("distinct"):
        # Execution order: WHERE (done by the caller) -> sort keys -> stable
        # sort -> LIMIT -> result expressions, so rows filtered out or cut by
        # LIMIT never evaluate the SELECT expressions.
        if order_by is not None:
            comparator = _make_row_comparator(source_columns, order_by, selected)
            selected = sorted(selected, key=cmp_to_key(comparator))

        if select.limit is not None:
            selected = selected[: select.limit]

        out_columns = _project_items(items, source_columns, selected)
        return Table._from_storage(out_schema, out_columns)

    # DISTINCT: WHERE -> project every surviving input row -> deduplicate the
    # full result rows -> sort -> LIMIT.  Projection happens before the sort
    # so ORDER BY compares deduplicated output values, and before LIMIT so a
    # row LIMIT would cut still surfaces a real division-by-zero/overflow.
    rows = _project_item_rows(items, source_columns, selected)
    rows = _deduplicate_rows(rows, out_schema)

    if order_by is not None:
        comparator = _make_projected_comparator(order_by)
        rows = sorted(rows, key=cmp_to_key(comparator))

    if select.limit is not None:
        rows = rows[: select.limit]

    width = len(items)
    out_columns = [tuple(row[c] for row in rows) for c in range(width)]
    return Table._from_storage(out_schema, out_columns)


def _project_items(items, source_columns, selected) -> list:
    out_columns = []
    for item in items:
        if item.kind == "column":
            out_columns.append(
                tuple(source_columns[item.col_index][i] for i in selected)
            )
        else:  # "expr"
            # Item-major like the historical path: the whole expression
            # column is evaluated (batch by batch) before the next item.
            out_columns.append(
                tuple(_eval_expr_selection(item.expr, source_columns, selected))
            )
    return out_columns


def _project_item_rows(items, source_columns, selected) -> list[tuple]:
    # DISTINCT projection: the historical evaluation order is row-major
    # (every item of a row before the next row), so each batch evaluates
    # every item over its rows and a failure re-runs the batch row by row
    # to surface the exact historical error.
    rows: list[tuple] = []
    for batch in _iter_batches(selected):
        rows.extend(_project_item_batch(items, source_columns, batch))
    return rows


def _project_item_batch(items, source_columns, batch) -> list[tuple]:
    try:
        columns = [
            [source_columns[item.col_index][i] for i in batch]
            if item.kind == "column"
            else item.expr.eval_batch(source_columns, batch, None)
            for item in items
        ]
    except QueryValidationError:
        # Reproduce the historical row-major error for this batch.
        return [
            tuple(
                source_columns[item.col_index][i]
                if item.kind == "column"
                else item.expr.eval(_row_values(source_columns, i), None)
                for item in items
            )
            for i in batch
        ]
    return [tuple(column[k] for column in columns) for k in range(len(batch))]


def _deduplicate_rows(rows: list[tuple], schema: Schema) -> list[tuple]:
    """Keep the first occurrence of each distinct result row.

    Equality follows the result schema column by column: NULLs compare equal
    to each other but never to a non-NULL value, bool/utf8/numeric keep
    their existing semantics, and float64 0.0 and -0.0 compare equal.
    """
    seen: set[tuple] = set()
    unique: list[tuple] = []
    for row in rows:
        key = tuple(
            _distinct_key_part(schema.columns[c].type, value)
            for c, value in enumerate(row)
        )
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _distinct_key_part(type_name: str, value):
    if value is None:
        return (False,)
    if type_name == "float64":
        # 0.0 and -0.0 are the same distinct value; the wrapper also keeps a
        # float from ever colliding with an int64 part of the same number.
        return (True, 0.0 if value == 0 else value)
    return (True, value)


def _make_projected_comparator(order_by: tuple[tuple, ...]):
    def compare(a: tuple, b: tuple) -> int:
        for entry in order_by:
            col_index = entry[1]
            c = _compare_scalar(a[col_index], b[col_index], entry[2], entry[3])
            if c:
                return c
        return 0

    return compare


def _run_aggregate(table: Table, select: _Select, bound, selected: list[int]) -> Table:
    group_indices = bound["group_indices"]
    bound_items = bound["items"]
    all_aggs = bound["aggregates"]
    having = bound["having"]
    order_by = bound["order_by"]
    source_columns = table._columns

    out_schema = Schema(
        [
            ColumnSchema(item.output_name, item.out_type, nullable=item.nullable)
            for item in bound_items
        ]
    )

    if group_indices:
        groups = _build_groups(selected, source_columns, group_indices)
    else:
        # Without GROUP BY the whole filtered stream is the single group; an
        # empty stream still produces the one all-NULL/COUNT-0 row.
        groups = [tuple(selected)]

    # Projected aggregates are the same _BoundItem instances as the registry
    # entries (frozen, appended by reference), so identity gives their slots.
    agg_slot_by_id = {id(agg): i for i, agg in enumerate(all_aggs)}

    # One materialised SELECT tuple per group that survives the post-grouping
    # HAVING filter; the original first-row group order is preserved.
    materialised: list[tuple] = []
    for rows in groups:
        agg_values = tuple(
            _aggregate_value(
                agg.func, agg.arg_index, agg.arg_type, rows, source_columns,
                agg.distinct,
            )
            for agg in all_aggs
        )
        if having is not None and _eval_having(having, source_columns, rows, agg_values) is not True:
            continue
        values = []
        for item in bound_items:
            if item.kind == "column":
                values.append(source_columns[item.col_index][rows[0]])
            else:
                values.append(agg_values[agg_slot_by_id[id(item)]])
        materialised.append(tuple(values))

    if order_by is not None:
        comparator = _make_tuple_comparator(materialised, order_by)
        order = sorted(range(len(materialised)), key=cmp_to_key(comparator))
    else:
        order = list(range(len(materialised)))

    if select.limit is not None:
        order = order[: select.limit]

    width_out = len(bound_items)
    out_columns = [tuple(materialised[r][c] for r in order) for c in range(width_out)]
    return Table._from_storage(out_schema, out_columns)


def _eval_having(
    node: _Expr,
    source_columns: tuple[tuple, ...],
    rows: tuple[int, ...],
    agg_values: tuple,
) -> bool | None:
    """Three-valued evaluation of a bound HAVING condition for one group.

    The bound tree is the same expression IR as WHERE; grouping columns
    are read from the group's first selected row, which is always present
    for a group node (the global aggregate case carries no columns).
    """
    row = _row_values(source_columns, rows[0]) if rows else ()
    return node.eval(row, agg_values)


def _build_groups(
    selected: list[int],
    source_columns: tuple[tuple, ...],
    group_indices: tuple[int, ...],
) -> list[tuple[int, ...]]:
    groups: dict[tuple, list[int]] = {}
    order: list[tuple] = []
    for row_index in selected:
        key = tuple(source_columns[col][row_index] for col in group_indices)
        bucket = groups.get(key)
        if bucket is None:
            bucket = []
            groups[key] = bucket
            order.append(key)
        bucket.append(row_index)
    return [tuple(groups[key]) for key in order]


def _make_row_comparator(
    source_columns: tuple[tuple, ...],
    order_by: tuple[tuple, ...],
    selected: list[int],
):
    # Sort keys are computed for every row that passed WHERE, before the
    # stable sort and LIMIT; expression keys (SELECT aliases) are evaluated
    # here, batch by batch over the surviving rows, so their errors surface
    # even for rows LIMIT would cut.
    key_sources: list[tuple] = []
    for entry in order_by:
        if entry[0] == "col":
            key_sources.append((source_columns[entry[1]], entry[2], entry[3]))
        else:  # "expr"
            values = dict(
                zip(selected, _eval_expr_selection(entry[1], source_columns, selected))
            )
            key_sources.append((values, entry[2], entry[3]))

    def compare(a: int, b: int) -> int:
        for values, descending, nulls_first in key_sources:
            c = _compare_scalar(values[a], values[b], descending, nulls_first)
            if c:
                return c
        return 0

    return compare


def _make_tuple_comparator(
    rows: list[tuple],
    order_by: tuple[tuple[int, bool, bool], ...],
):
    def compare(a: int, b: int) -> int:
        for col_index, descending, nulls_first in order_by:
            va = rows[a][col_index]
            vb = rows[b][col_index]
            c = _compare_scalar(va, vb, descending, nulls_first)
            if c:
                return c
        return 0

    return compare


def _compare_scalar(va, vb, descending: bool, nulls_first: bool) -> int:
    if va is None or vb is None:
        if va is None and vb is None:
            return 0
        # NULL placement follows NULLS FIRST / NULLS LAST alone; the
        # default (and the explicit LAST spelling) keeps NULLs at the
        # end for both ASC and DESC, so DESC must not flip this part.
        none_before = -1 if nulls_first else 1
        return none_before if va is None else -none_before
    c = (va > vb) - (va < vb)
    return -c if descending else c
