"""Execution: materialise prepared scans and run the bound statement.

The shared preparation stage (:mod:`columnar_analytics.prepare`) has
already settled parsing, source selection, schema binding, required
columns and v2 row-group selection.  This module only materialises the
prepared sources -- full v1 reads, column/row-group-selective v2 reads
and the per-step join-chain assembly -- and schedules the row-processing
pipeline (WHERE filter, ORDER BY sort keys, projection, DISTINCT
deduplication, grouping, aggregation, HAVING, stable sort and LIMIT) on
top of the shared batched evaluator.  All semantics live in the bound
expression IR and the join layer; this module only drives them.
"""

from __future__ import annotations

import math
from functools import cmp_to_key

from .binder import _bind_select
from .errors import QueryValidationError
from .expr import (
    _eval_expr_selection,
    _filter_rows,
    _iter_batches,
    _row_values,
)
from .format import (
    FORMAT_VERSION_PARTITIONED,
    ColumnSchema,
    Schema,
    Table,
    _read_partitioned_table,
    read_file,
)
from .join import _execute_join
from .parser import _INT64_MAX, _INT64_MIN, _Select
from .resolve import (
    _JoinStep,
    _build_joined_schema,
    _qualify_table,
)


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


def _execute_prepared(prepared) -> Table:
    """Execute a statement prepared by :mod:`columnar_analytics.prepare`.

    The prepared result already settled parsing, source selection, binding,
    required columns and v2 row-group selection, so this only reads the data
    the plan calls for and drives the row-processing pipeline: a single
    source is read as one table, a join chain is assembled step by step with
    each step's materialised intermediate result as its left input, and
    WHERE / GROUP BY / HAVING / projection / DISTINCT / ORDER BY / LIMIT run
    only after the whole chain has been built.
    """
    if not prepared.steps:
        spec = prepared.scans[0]
        table = _read_source(spec, reduce_columns=False)
        return _run_query(
            table, prepared.select, expected_table=prepared.expected_table
        )

    combined = _qualify_table(
        _read_source(prepared.scans[0], reduce_columns=True), prepared.scans[0].key
    )
    for spec, step in zip(prepared.scans[1:], prepared.steps):
        new_table = _read_source(spec, reduce_columns=True)
        combined = _execute_join_step(
            combined, new_table, step, prepared.strategy
        )
    return _run_query(combined, prepared.select, expected_table=None)


def _read_source(spec, *, reduce_columns: bool) -> Table:
    """Materialise one prepared source with exactly its planned read range.

    v1 sources keep their historical full read; a v2 source decodes only the
    required columns and the selected row groups (every group for a chain
    containing an OUTER step).

    ``reduce_columns`` is False for the single-source path (unread blocks
    stay inert placeholders, so the table keeps its full schema layout and
    ``_run_query`` re-binds exactly as it always did) and True for every
    source of a join chain, where each read table is projected to its
    required columns in source order before the shared join layer qualifies
    and assembles it -- matching both the plan's per-source required_columns
    and the historical chain read path.
    """
    if spec.version != FORMAT_VERSION_PARTITIONED:
        return read_file(spec.path)
    row_groups = None if spec.selected_groups is None else list(spec.selected_groups)
    table = _read_partitioned_table(
        spec.path,
        columns=set(spec.required_columns),
        row_groups=row_groups,
    )
    if not reduce_columns:
        return table
    return table.project(list(spec.required_columns))

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
