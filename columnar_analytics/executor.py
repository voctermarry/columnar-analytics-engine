"""Execution: run a bound statement over materialised tables.

This stage schedules the row-processing pipeline -- WHERE filter, ORDER BY
sort keys, projection, DISTINCT deduplication, grouping, aggregation,
HAVING, stable sort and LIMIT -- on top of the shared batched evaluator,
and prepares the scans that feed it: column- and row-group-selective reads
of v2 sources and the per-step join-chain assembly.  All semantics live in
the bound expression IR and the join layer; this module only drives them.
"""

from __future__ import annotations

import math
from functools import cmp_to_key
from typing import Any

from .binder import _bind_select, _collect_required_indices
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
    inspect_file,
    inspect_row_groups,
    read_file,
)
from .join import _execute_join
from .parser import _INT64_MAX, _INT64_MIN, _Select
from .pushdown import (
    _extract_pushable,
    _pushed_col_indices,
    _pushable_leaves_by_source,
    _select_row_groups,
    _source_column_spans,
    _source_scan_info,
)
from .resolve import (
    _JoinStep,
    _build_joined_schema,
    _qualify_table,
    _schema_from_metadata,
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
def _query_partitioned(
    path: Any, select: _Select, schema: Schema, expected_table
) -> Table:
    """Execute a single-source statement against a v2 (partitioned) file.

    Only the columns the bound plan references are decoded, and only the
    row groups whose statistics do not exclude them.
    """
    bound = _bind_select(select, schema, expected_table=expected_table)
    required = _collect_required_indices(bound)
    groups = inspect_row_groups(path)
    pushed = _extract_pushable(bound["where"])
    selected = _select_row_groups(groups, pushed, _pushed_col_indices(pushed))
    columns = {schema.columns[i].name for i in required}
    table = _read_partitioned_table(path, columns=columns, row_groups=selected)
    return _run_query(table, select, expected_table=expected_table)
def _run_join_chain_partitioned(
    paths, rewritten: _Select, from_key: str, steps, strategy, table_keys
) -> Table:
    """Join chain with at least one v2 (partitioned) source.

    Every source is read restricted to the columns the plan actually
    references (its join keys included); v1 sources keep the historical
    full read.  When every join step is INNER, each v2 source additionally
    reads only the row groups that the top-level AND leaves referencing
    that source alone cannot rule out; a chain containing any OUTER step
    reads every group.  WHERE still runs in full over the joined rows.
    """
    schemas = {}
    versions = {}
    for key in table_keys:
        metadata = inspect_file(paths[key])
        schemas[key] = _schema_from_metadata(metadata)
        versions[key] = metadata["format_version"]
    combined_schema = Schema(
        tuple(
            ColumnSchema(f"{from_key}.{col.name}", col.type, col.nullable)
            for col in schemas[from_key].columns
        )
    )
    for step in steps:
        combined_schema = _build_joined_schema(
            combined_schema, schemas[step.new_key], step
        )
    bound = _bind_select(rewritten, combined_schema, expected_table=None)
    referenced = _collect_required_indices(bound)
    for step in steps:
        referenced.add(combined_schema.index(f"{step.prior_key}.{step.prior_col}"))
        referenced.add(combined_schema.index(f"{step.new_key}.{step.new_col}"))
    referenced_names = {combined_schema.columns[i].name for i in referenced}

    all_inner = all(step.kind == "inner" for step in steps)
    if all_inner:
        spans = _source_column_spans(schemas, table_keys)
        leaves_by_source = _pushable_leaves_by_source(
            bound["where"], spans, table_keys
        )
        group_selection: dict = {}
        for key in table_keys:
            if versions[key] == FORMAT_VERSION_PARTITIONED:
                groups = inspect_row_groups(paths[key])
                info = _source_scan_info(groups, leaves_by_source[key], spans[key])
                group_selection[key] = info["selected_groups"]
    else:
        # An OUTER step can pad a source's rows with NULLs, so no source
        # group can be proven absent from the join result.
        group_selection = {}

    def read_source(key: str) -> Table:
        if versions[key] != FORMAT_VERSION_PARTITIONED:
            return read_file(paths[key])
        required = [
            col.name
            for col in schemas[key].columns
            if f"{key}.{col.name}" in referenced_names
        ]
        selected = group_selection.get(key)
        return _read_partitioned_table(
            paths[key], columns=set(required), row_groups=selected
        ).project(required)

    combined = _qualify_table(read_source(from_key), from_key)
    for step in steps:
        combined = _execute_join_step(
            combined, read_source(step.new_key), step, strategy
        )
    return _run_query(combined, rewritten, expected_table=None)
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
