"""Shared query preparation between execution, explain and export.

One preparation pass owns every rule that precedes -- and is identical for
-- running a statement and rendering its plan:

1. argument and join-strategy validation (no file touched),
2. SQL parsing and source / ON-name resolution (no file touched),
3. metadata inspection of the referenced sources only (never their data),
4. canonical-name rewrite and schema binding against the combined schema,
5. required-column collection (the ON keys of a join chain included), and
6. v2 row-group statistics pushdown / group selection.

Both the executor and the explain plan consume the single
:class:`_PreparedStatement` result, so the columns and row groups a query
reads and the columns, counts and ``pushed_condition`` a Scan reports can
never drift apart.  This stage reads metadata only -- data sections are
neither decompressed nor decoded here, and an unreferenced source is never
opened; the executor performs all data access afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .binder import _bind_select, _collect_required_indices
from .format import (
    FORMAT_VERSION_PARTITIONED,
    ColumnSchema,
    Schema,
    inspect_file,
    inspect_row_groups,
)
from .parser import _Parser, _Select, _tokenize
from .pushdown import (
    _pushable_leaves_by_source,
    _scan_pushdown_fields,
    _source_column_spans,
    _source_scan_info,
)
from .resolve import (
    _JoinStep,
    _build_joined_schema,
    _multi_table_resolver,
    _resolve_statement,
    _rewrite_select,
    _schema_from_metadata,
    _single_table_resolver,
)

_SINGLE_TABLE_NAME = "input"


@dataclass(frozen=True)
class _PreparedStatement:
    """The one prepared result shared by execution and the explain plan.

    ``sources`` lists ``(table key, metadata dict, bare schema)`` for every
    referenced source in FROM/JOIN order; ``schema`` is the combined schema
    the statement was bound against (identical to ``sources[0]`` for a
    single-source statement); ``rewritten`` is the canonical-name parsed
    statement that both binding and execution use; ``required_columns`` maps
    each source key to the local column names actually read, in that
    source's schema order; ``group_selection`` maps an eligible v2 source to
    the row-group indices statistics did not exclude (its absence means a
    full read: a v1 source or a chain containing an OUTER step);
    ``scan_extras`` carries the explain-plan pushdown fields for the very
    same selections.
    """

    paths: dict
    rewritten: _Select
    strategy: str | None
    from_key: str
    steps: tuple[_JoinStep, ...]
    table_keys: tuple
    sources: tuple  # (key, metadata, bare schema), FROM/JOIN order
    schema: Schema  # combined schema
    bound: dict
    expected_table: str | None
    required_columns: dict  # key -> ordered list of local column names
    group_selection: dict  # key -> selected group indices (v2, eligible)
    scan_extras: dict | None  # key -> plan pushdown fields


def prepare_single(path: Any, sql: str) -> _PreparedStatement:
    """Prepare a ``FROM input`` statement against one fixed-input file."""
    # The single-file grammar rejects JOINs and qualified names; only after
    # parsing succeeds is the file's metadata touched.
    select = _Parser(_tokenize(sql), allow_join=False).parse()
    return _prepare(
        paths={_SINGLE_TABLE_NAME: path},
        select=select,
        strategy=None,
        from_key=_SINGLE_TABLE_NAME,
        steps=(),
        single_table_fixed=True,
    )


def prepare_mapped(
    sources: Any, sql: str, join_strategy: Any = None
) -> _PreparedStatement:
    """Prepare a statement against a table-name to path mapping."""
    paths, select, strategy, from_key, steps = _resolve_statement(
        sources, sql, join_strategy
    )
    return _prepare(
        paths=paths,
        select=select,
        strategy=strategy,
        from_key=from_key,
        steps=steps,
        single_table_fixed=False,
    )


def _prepare(
    *,
    paths: dict,
    select: _Select,
    strategy: str | None,
    from_key: str,
    steps: tuple[_JoinStep, ...],
    single_table_fixed: bool,
) -> _PreparedStatement:
    table_keys = (from_key, *(step.new_key for step in steps))

    # Canonical-name rewrite / qualifier validation needs no file: a fixed
    # input statement keeps its bare names, a JOIN-less mapped statement has
    # its single qualifier stripped, and a join chain rewrites every
    # reference to "table.column".
    if not steps:
        expected_table = _SINGLE_TABLE_NAME if single_table_fixed else None
        if single_table_fixed:
            rewritten = select
        else:
            rewritten = _rewrite_select(select, _single_table_resolver(from_key))
    else:
        expected_table = None
        rewritten = _rewrite_select(select, _multi_table_resolver(table_keys))

    # Metadata is inspected for referenced sources only, in FROM/JOIN order;
    # the combined schema derives each step exactly as execution does, so
    # binding and planning see one and the same layout.
    source_entries = []
    from_metadata = inspect_file(paths[from_key])
    from_schema = _schema_from_metadata(from_metadata)
    source_entries.append((from_key, from_metadata, from_schema))
    if not steps:
        combined_schema = from_schema
    else:
        combined_schema = Schema(
            tuple(
                ColumnSchema(f"{from_key}.{col.name}", col.type, col.nullable)
                for col in from_schema.columns
            )
        )
        for step in steps:
            metadata = inspect_file(paths[step.new_key])
            new_schema = _schema_from_metadata(metadata)
            source_entries.append((step.new_key, metadata, new_schema))
            combined_schema = _build_joined_schema(
                combined_schema, new_schema, step
            )

    bound = _bind_select(rewritten, combined_schema, expected_table=expected_table)

    referenced = _collect_required_indices(bound)
    for step in steps:
        # The ON keys feed every join even when neither is projected.
        referenced.add(combined_schema.index(f"{step.prior_key}.{step.prior_col}"))
        referenced.add(combined_schema.index(f"{step.new_key}.{step.new_col}"))

    required_columns = _required_columns_per_source(
        source_entries, combined_schema, referenced, steps
    )
    group_selection, scan_extras = _plan_row_groups(
        source_entries,
        paths,
        bound,
        table_keys,
        steps,
    )

    return _PreparedStatement(
        paths=paths,
        rewritten=rewritten,
        strategy=strategy,
        from_key=from_key,
        steps=steps,
        table_keys=table_keys,
        sources=tuple(source_entries),
        schema=combined_schema,
        bound=bound,
        expected_table=expected_table,
        required_columns=required_columns,
        group_selection=group_selection,
        scan_extras=scan_extras,
    )


def _required_columns_per_source(
    source_entries, combined_schema, referenced, steps
) -> dict:
    """Each source's read columns, in that source's schema order.

    A single-source statement (fixed input or JOIN-less mapping) reads bare
    names straight from the combined schema; a join chain maps combined
    ``table.column`` names back to each source's local columns.
    """
    referenced_names = {combined_schema.columns[i].name for i in referenced}
    required: dict = {}
    if not steps:
        key, _metadata, _schema = source_entries[0]
        required[key] = [
            col.name for col in combined_schema.columns if col.name in referenced_names
        ]
        return required
    for key, _metadata, source_schema in source_entries:
        prefix = f"{key}."
        required[key] = [
            col.name
            for col in source_schema.columns
            if f"{prefix}{col.name}" in referenced_names
        ]
    return required


def _plan_row_groups(source_entries, paths, bound, table_keys, steps):
    """Select v2 row groups and build the Scan pushdown fields together.

    Statistics pushdown is eligible for a single-source statement and for a
    chain made entirely of INNER joins; any OUTER step disables pruning for
    every source.  The selected group indices (consumed by the executor) and
    the plan fields (total / selected counts and pushed_condition) derive
    from one shared scan-info object, so they always agree.
    """
    eligible = not steps or all(step.kind == "inner" for step in steps)
    if not eligible:
        # An OUTER step can pad a source's rows with NULLs, so no source
        # group can be proven absent from the join result.
        return {}, None

    bare_schemas = {
        key: source_schema for key, _metadata, source_schema in source_entries
    }
    spans = _source_column_spans(bare_schemas, table_keys)
    leaves_by_source = _pushable_leaves_by_source(bound["where"], spans, table_keys)
    group_selection: dict = {}
    scan_extras: dict = {}
    for key, metadata, _source_schema in source_entries:
        if metadata["format_version"] != FORMAT_VERSION_PARTITIONED:
            continue
        groups = inspect_row_groups(paths[key])
        info = _source_scan_info(groups, leaves_by_source[key], spans[key])
        group_selection[key] = info["selected_groups"]
        scan_extras[key] = _scan_pushdown_fields(info)
    return group_selection, scan_extras
