"""Equi-join execution layer.

This module is the single owner of join *mechanics* for every multi-table
statement; it deliberately knows nothing about SQL parsing, qualified-name
binding, WHERE / GROUP BY / ORDER BY, explain plans or file export.  The
query layer resolves names and key columns and hands this layer plain
:class:`~columnar_analytics.format.Table` inputs; the export layer only
shares :func:`_validate_join_strategy`.

The boundary is split in two halves so that adding a join algorithm never
copies the second half of any query:

* **Algorithms only describe match relations.**  A matcher receives the two
  key columns and returns ``{left_row_index: (right_row_index, ...)}`` for
  the rows that match.  Right-row indices inside one entry follow the right
  table's original row order; nothing else (inverse bookkeeping, row
  expansion, outer padding, ordering, schema) is an algorithm concern.
  Algorithms register themselves in :data:`_JOIN_ALGORITHMS`.

* **The common pipeline assembles results.**  Given a match relation it
  derives the inverse relation, expands every matched combination, pads
  unmatched rows with NULL and wraps the combined schema, all in one place:
  key semantics (NULL never matches; int64 and float64 keys compare equal,
  including ``0.0`` and ``-0.0``), deterministic unordered row order, outer
  NULL placement and result nullability therefore cannot drift between
  algorithms.

Deterministic row order when the statement gives no explicit ORDER BY:

* INNER / LEFT / FULL follow the current intermediate (left) row order, and
  within one left row the matching new (right) rows follow new-file order;
* RIGHT follows new-file row order, with one new row's matches expanded in
  intermediate row order;
* FULL emits the LEFT output and then appends unmatched new rows in
  new-file order.

A chained statement runs one independent join per step, so each step obeys
the same rules with the previous step's materialised result as its left
input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .format import ColumnSchema, Schema, Table

__all__: list[str] = []


# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _JoinAlgorithm:
    """One registered join algorithm.

    ``label`` is the strategy name surfaced by an explain plan Join
    operator; ``find`` builds the match relation (left row -> right rows in
    right-file order) from the two key columns.
    """

    label: str
    find: Callable[[tuple, tuple], dict[int, list[int]]]


def _hash_matches(left_keys: tuple, right_keys: tuple) -> dict[int, list[int]]:
    """Right-key hash index: the implicit default algorithm as well."""
    # NULL keys never match, so they stay out of the right-side index.
    # Python equality/hashing already gives the equi-join key rules: int64
    # and float64 values compare equal (1 == 1.0), as do 0.0 and -0.0.
    index: dict[Any, list[int]] = {}
    for j, key in enumerate(right_keys):
        if key is not None:
            index.setdefault(key, []).append(j)
    matches_by_left: dict[int, list[int]] = {}
    for i, key in enumerate(left_keys):
        if key is None:
            continue
        matches = index.get(key)
        if matches:
            matches_by_left[i] = matches
    return matches_by_left


def _sort_merge_matches(left_keys: tuple, right_keys: tuple) -> dict[int, list[int]]:
    """Stably sort both sides by key (NULLs excluded) and merge equal runs."""
    left_order = sorted(
        (i for i, key in enumerate(left_keys) if key is not None),
        key=left_keys.__getitem__,
    )
    right_order = sorted(
        (j for j, key in enumerate(right_keys) if key is not None),
        key=right_keys.__getitem__,
    )

    # Index each non-NULL right key to the run of right rows carrying it;
    # runs are visited in sorted order and keep right-file row order.
    right_runs: dict[Any, list[int]] = {}
    run_order: list[Any] = []
    for j in right_order:
        key = right_keys[j]
        run = right_runs.get(key)
        if run is None:
            run = []
            right_runs[key] = run
            run_order.append(key)
        run.append(j)

    matches_by_left: dict[int, list[int]] = {}
    p = 0
    for i in left_order:
        key = left_keys[i]
        while p < len(run_order) and run_order[p] < key:
            p += 1
        if p < len(run_order) and run_order[p] == key:
            matches_by_left[i] = right_runs[run_order[p]]
    return matches_by_left


# The registry is the only place strategies are named: validation, the
# explain-plan label and execution dispatch all read it.  A new algorithm
# registers one entry here and automatically works for every join kind and
# chain without touching the query, explain or export layers.
_JOIN_ALGORITHMS: dict[str, _JoinAlgorithm] = {
    "hash": _JoinAlgorithm("HASH", _hash_matches),
    "sort_merge": _JoinAlgorithm("SORT_MERGE", _sort_merge_matches),
}

_JOIN_STRATEGIES = frozenset(_JOIN_ALGORITHMS)


def _validate_join_strategy(join_strategy: Any) -> str | None:
    """Validate the optional ``join_strategy`` argument.

    Returns the canonical lowercase strategy name, or ``None`` when the
    caller did not request one.  Any other value (including a non-string)
    raises :class:`ValueError`; callers run this before opening any file.
    """
    if join_strategy is None:
        return None
    if not isinstance(join_strategy, str) or join_strategy not in _JOIN_ALGORITHMS:
        allowed = ", ".join(sorted(_JOIN_ALGORITHMS))
        raise ValueError(
            f"invalid join_strategy: {join_strategy!r}; expected one of {allowed}"
        )
    return join_strategy


def _strategy_label(strategy: str) -> str:
    """The explain-plan label of a validated strategy (registry lookup)."""
    return _JOIN_ALGORITHMS[strategy].label


def _find_matches(
    left_keys: tuple, right_keys: tuple, strategy: str | None
) -> dict[int, list[int]]:
    """Build one step's match relation with the selected algorithm.

    ``strategy`` is ``None`` for the implicit historical path (the hash
    lookup); any validated strategy name dispatches through the registry.
    """
    if strategy is None:
        return _hash_matches(left_keys, right_keys)
    return _JOIN_ALGORITHMS[strategy].find(left_keys, right_keys)


# ---------------------------------------------------------------------------
# Result assembly (shared by every algorithm)
# ---------------------------------------------------------------------------


def _invert_matches(
    matches_by_left: dict[int, list[int]], left_row_count: int
) -> dict[int, list[int]]:
    """Inverse relation ``right row -> matching left rows``.

    Left rows are walked in their original row order, so each right row's
    list keeps left-file order regardless of how the algorithm discovered
    the matches.
    """
    matches_by_right: dict[int, list[int]] = {}
    for i in range(left_row_count):
        for j in matches_by_left.get(i, ()):
            matches_by_right.setdefault(j, []).append(i)
    return matches_by_right


def _assemble_join_columns(
    left_cols: tuple,
    right_cols: tuple,
    left_row_count: int,
    right_row_count: int,
    matches_by_left: dict[int, list[int]],
    kind: str,
) -> list:
    """Expand a match relation into joined columns for every join kind.

    INNER / LEFT / FULL are driven by left (intermediate) row order: matched
    combinations first, then unmatched left rows; FULL additionally appends
    the unmatched right rows in right (new-file) order.  RIGHT is driven by
    right-file order, with each right row's combinations expanded in left
    row order.  Padding rows carry NULLs on the unmatched side.
    """
    matches_by_right = _invert_matches(matches_by_left, left_row_count)
    left_width = len(left_cols)
    right_width = len(right_cols)
    out = [[] for _ in range(left_width + right_width)]

    def emit_match(i: int, j: int) -> None:
        for c in range(left_width):
            out[c].append(left_cols[c][i])
        for c in range(right_width):
            out[left_width + c].append(right_cols[c][j])

    def emit_unmatched_left(i: int) -> None:
        for c in range(left_width):
            out[c].append(left_cols[c][i])
        for c in range(right_width):
            out[left_width + c].append(None)

    def emit_unmatched_right(j: int) -> None:
        for c in range(left_width):
            out[c].append(None)
        for c in range(right_width):
            out[left_width + c].append(right_cols[c][j])

    if kind == "right":
        for j in range(right_row_count):
            left_matches = matches_by_right.get(j)
            if left_matches:
                for i in left_matches:
                    emit_match(i, j)
            else:
                emit_unmatched_right(j)
        return out

    for i in range(left_row_count):
        right_matches = matches_by_left.get(i)
        if right_matches:
            for j in right_matches:
                emit_match(i, j)
        elif kind in ("left", "full"):
            emit_unmatched_left(i)
    if kind == "full":
        for j in range(right_row_count):
            if j not in matches_by_right:
                emit_unmatched_right(j)
    return out


def _derive_step_schema(
    prior_schema: Schema,
    new_schema: Schema,
    *,
    new_table: str,
    kind: str,
) -> Schema:
    """Derive the combined post-step schema and its column nullability.

    The intermediate (prior) schema already names its columns
    ``table.column``; the freshly introduced table still carries its bare
    schema and its columns are prefixed with the new table name here.  An
    outer side whose unmatched rows are padded with NULL gains nullable
    result columns: the intermediate is the join's left side, the new table
    its right side; LEFT pads the right side, RIGHT pads the left side,
    FULL pads both; INNER keeps the existing nullability.

    This is pure schema derivation: the query layer owns key existence,
    key-type compatibility and duplicate-name validation (and their
    exception classification) and calls this for both execution and the
    explain plan, so nullability cannot drift between the two paths.
    """
    pad_left = kind in ("right", "full")
    pad_right = kind in ("left", "full")
    combined_columns = [
        ColumnSchema(
            col.name,
            col.type,
            True if pad_left else col.nullable,
        )
        for col in prior_schema.columns
    ] + [
        ColumnSchema(
            f"{new_table}.{col.name}",
            col.type,
            True if pad_right else col.nullable,
        )
        for col in new_schema.columns
    ]
    return Schema(combined_columns)


def _execute_join(
    prior: Table,
    new: Table,
    prior_key_index: int,
    new_key_index: int,
    schema: Schema,
    *,
    kind: str,
    strategy: str | None,
) -> Table:
    """Run one equi-join step: match relation plus the common assembly.

    ``prior`` is the materialised intermediate result (left input, columns
    already named ``table.column``); ``new`` is the freshly introduced table
    (right input, bare column names).  ``schema`` is the post-step schema
    the query layer derived and validated (key names/types and duplicate
    names are therefore already settled).  The algorithm only locates
    matches; expansion order, outer NULL padding and the result shape are
    handled here and are identical for every registered algorithm.
    """
    matches_by_left = _find_matches(
        prior._columns[prior_key_index],
        new._columns[new_key_index],
        strategy,
    )
    columns = _assemble_join_columns(
        prior._columns,
        new._columns,
        prior.row_count,
        new.row_count,
        matches_by_left,
        kind,
    )
    return Table._from_storage(schema, columns)
