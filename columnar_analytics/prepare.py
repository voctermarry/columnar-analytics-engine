"""Shared query preparation: one parse/resolve/bind/scan-plan pipeline.

Every public entry point -- :func:`columnar_analytics.query.query_file`,
``query_files``, ``explain_file``, ``explain_files`` and the export layer
through the two query entries -- prepares its statement here before the
prepared result either drives a data read (execution) or the explain plan.
The preparation itself never touches a data section: only source
validation, parsing, name/source resolution, referenced-source metadata,
schema binding, required-column collection and v2 row-group selection run
here, so the query, explain and export paths can never disagree.

The two flavours share one result type:

* :func:`_prepare_single` -- one file addressed directly, read as the
  fixed ``input`` table with the single-file grammar (no JOIN clauses);
* :func:`_prepare_multi` -- a table-name to path mapping, parsed with the
  join grammar, its sources resolved in FROM/JOIN order.

A JOIN-less multi statement yields the same binding, required-column
projection and row-group selection as the equivalent single-file
statement; a join chain attributes the required columns to their owning
sources, and statistics pushdown is enabled only for an all-INNER chain.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    _scan_pushdown_info,
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


@dataclass(frozen=True)
class _ScanSpec:
    """One referenced source's shared read/plan description.

    ``required_columns`` lists the source's own columns the statement
    actually reads (its ON keys included), in the source's schema order.
    A v1 source is still read in full at execution, while the explain Scan
    always reports exactly those columns.

    ``selected_groups`` is non-None only for a v2 source of an eligible
    statement (a single source or an all-INNER chain) and lists the row
    groups execution decodes; None means every group is read (a v1 file or
    a chain containing an OUTER step).  ``explain_fields`` carries the
    three pushdown fields appended to an eligible v2 Scan, or None when the
    Scan keeps its historical shape.
    """

    key: str
    path: Any
    metadata: Mapping
    schema: Schema
    version: int
    required_columns: tuple[str, ...]
    selected_groups: tuple[int, ...] | None
    explain_fields: dict | None


@dataclass(frozen=True)
class _PreparedQuery:
    """Everything the shared preparation settled before any data is read."""

    # The statement tree execution runs: the raw parse for the single-file
    # entries, the canonical-name rewrite for mapped sources.
    select: _Select
    bound: Mapping
    # The schema the bound result is relative to: a bare source schema for
    # one source, the qualified FROM/JOIN concatenation for a chain.
    schema: Schema
    scans: tuple[_ScanSpec, ...]
    steps: tuple[_JoinStep, ...]
    strategy: str | None
    # _run_query's expected-table guard: "input" for query_file, None for
    # mapped sources.
    expected_table: str | None

    @property
    def scan_extras(self) -> dict | None:
        """The plan builder's per-source pushdown fields (None disables)."""
        extras = {
            spec.key: spec.explain_fields
            for spec in self.scans
            if spec.explain_fields is not None
        }
        return extras if extras else None

    def source_tuples(self) -> tuple:
        """The (key, metadata, bare schema) triples consumed by the plan."""
        return tuple(
            (spec.key, spec.metadata, spec.schema) for spec in self.scans
        )

    def required_by_source(self) -> dict:
        return {spec.key: spec.required_columns for spec in self.scans}


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _prepare_single(path: Any, sql: str) -> _PreparedQuery:
    """Prepare a single-file statement (fixed ``input`` table, no joins).

    The statement is parsed with the single-file grammar before the file is
    touched; only afterwards is the one source's metadata read and bound.
    """
    tokens = _tokenize(sql)
    select = _Parser(tokens).parse()
    return _prepare_single_parsed(
        path, select, table_name="input", expected_table="input"
    )


def _prepare_multi(sources: Any, sql: str, join_strategy: Any = None) -> _PreparedQuery:
    """Prepare a mapped-sources statement in FROM/JOIN order.

    Argument and join-strategy validation, parsing and table/ON resolution
    finish before any file is opened; the referenced sources are then read
    in FROM/JOIN order (unreferenced sources are never opened).
    """
    paths, select, strategy, from_key, steps = _resolve_statement(
        sources, sql, join_strategy
    )
    if not steps:
        rewritten = _rewrite_select(select, _single_table_resolver(from_key))
        return _prepare_single_parsed(
            paths[from_key],
            rewritten,
            table_name=from_key,
            expected_table=None,
        )

    table_keys = (from_key, *(step.new_key for step in steps))
    # Qualifier checks (unqualified / unknown-table column references) do
    # not need the files and run before any metadata is read.
    rewritten = _rewrite_select(select, _multi_table_resolver(table_keys))

    # Metadata is read for the referenced sources only, in FROM/JOIN order;
    # each step validates its ON keys against the intermediate schema and
    # derives the combined schema exactly as execution would.
    scan_inputs: list[tuple] = []
    from_metadata = inspect_file(paths[from_key])
    from_schema = _schema_from_metadata(from_metadata)
    scan_inputs.append((from_key, paths[from_key], from_metadata, from_schema))
    combined_schema = _qualified_schema(from_schema, from_key)
    bare_schemas = {from_key: from_schema}
    for step in steps:
        metadata = inspect_file(paths[step.new_key])
        new_schema = _schema_from_metadata(metadata)
        bare_schemas[step.new_key] = new_schema
        scan_inputs.append((step.new_key, paths[step.new_key], metadata, new_schema))
        combined_schema = _build_joined_schema(combined_schema, new_schema, step)

    bound = _bind_select(rewritten, combined_schema, expected_table=None)

    # The ON keys feed every join even when neither is projected.
    referenced = _collect_required_indices(bound)
    for step in steps:
        referenced.add(combined_schema.index(f"{step.prior_key}.{step.prior_col}"))
        referenced.add(combined_schema.index(f"{step.new_key}.{step.new_col}"))

    spans = _source_column_spans(bare_schemas, table_keys)
    required_by_source = _attribute_required_columns(
        bare_schemas, spans, table_keys, referenced
    )

    # Statistics pushdown is restricted to all-INNER chains; a chain with
    # any OUTER step reads every group and keeps every Scan's historical
    # shape.  In an eligible chain each v2 Scan reports its group counts and
    # the leaves attributed to that source; v1 Scans stay unchanged.
    all_inner = all(step.kind == "inner" for step in steps)
    leaves_by_source = (
        _pushable_leaves_by_source(bound["where"], spans, table_keys)
        if all_inner
        else None
    )

    scans: list[_ScanSpec] = []
    for key, path, metadata, bare_schema in scan_inputs:
        selected_groups = None
        explain_fields = None
        if all_inner and metadata["format_version"] == FORMAT_VERSION_PARTITIONED:
            groups = inspect_row_groups(path)
            info = _source_scan_info(groups, leaves_by_source[key], spans[key])
            selected_groups = tuple(info["selected_groups"])
            explain_fields = _scan_pushdown_fields(info)
        scans.append(
            _ScanSpec(
                key=key,
                path=path,
                metadata=metadata,
                schema=bare_schema,
                version=metadata["format_version"],
                required_columns=required_by_source[key],
                selected_groups=selected_groups,
                explain_fields=explain_fields,
            )
        )

    return _PreparedQuery(
        select=rewritten,
        bound=bound,
        schema=combined_schema,
        scans=tuple(scans),
        steps=steps,
        strategy=strategy,
        expected_table=None,
    )


def _prepare_single_parsed(
    path: Any, select: _Select, *, table_name: str, expected_table: str | None
) -> _PreparedQuery:
    """Bind one already-parsed single-source statement and plan its scan.

    Shared by the fixed-``input`` file entry and a JOIN-less mapped
    statement: both read one source's metadata, bind against its bare
    schema and select the referenced columns / v2 row groups.
    """
    metadata = inspect_file(path)
    schema = _schema_from_metadata(metadata)
    bound = _bind_select(select, schema, expected_table=expected_table)

    referenced = _collect_required_indices(bound)
    required = tuple(schema.columns[i].name for i in sorted(referenced))

    selected_groups = None
    explain_fields = None
    if metadata["format_version"] == FORMAT_VERSION_PARTITIONED:
        groups = inspect_row_groups(path)
        info = _scan_pushdown_info(groups, bound, schema)
        selected_groups = tuple(info["selected_groups"])
        explain_fields = _scan_pushdown_fields(info)

    spec = _ScanSpec(
        key=table_name,
        path=path,
        metadata=metadata,
        schema=schema,
        version=metadata["format_version"],
        required_columns=required,
        selected_groups=selected_groups,
        explain_fields=explain_fields,
    )
    return _PreparedQuery(
        select=select,
        bound=bound,
        schema=schema,
        scans=(spec,),
        steps=(),
        strategy=None,
        expected_table=expected_table,
    )


# ---------------------------------------------------------------------------
# Schema / required-column helpers
# ---------------------------------------------------------------------------


def _qualified_schema(schema: Schema, key: str) -> Schema:
    """One source's bare schema renamed to ``key.column`` columns."""
    return Schema(
        tuple(
            ColumnSchema(f"{key}.{col.name}", col.type, col.nullable)
            for col in schema.columns
        )
    )


def _attribute_required_columns(
    bare_schemas: Mapping, spans: Mapping, table_keys: tuple, referenced: set
) -> dict:
    """Project required combined-schema indices to per-source local names.

    Each source keeps the columns it owns (its ON keys included) in its own
    schema order; attribution goes through the contiguous combined-schema
    spans, which is robust against table names sharing dotted prefixes.
    """
    result: dict = {}
    for key in table_keys:
        start, end = spans[key]
        local_schema = bare_schemas[key]
        result[key] = tuple(
            local_schema.columns[i - start].name
            for i in range(start, end)
            if i in referenced
        )
    return result
