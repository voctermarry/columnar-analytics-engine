"""Repeatability and cross-query isolation tests for the public query entries.

These tests pin down one contract: *the same query over the same data is
stable and reconcilable*.  Every scenario is expressed only through the
public entry points (``write_file`` / ``write_partitioned_file`` to build
fixed columnar inputs, ``query_file`` / ``query_files`` and the installed
CLI to execute), and every comparison uses only the public ``Table``
surface (``schema.columns``, ``column_names``, ``column``, ``row_count``)
plus the documented CLI JSON shape.  No parser, planner, executor or
exporter internals are touched, and no new public surface is added.

For each scenario the same data + same query is executed and reconciled
three ways:

* **same process, consecutive runs** — three back-to-back executions;
* **recreated query context** — a brand-new interpreter running the
  installed CLI (twice, byte-identical), reconciled against the
  in-process result;
* **equivalent batch layouts** — the same logical table written as v1
  and as v2 with several ``row_group_size`` values (duplicate group keys
  and join keys deliberately span row groups), plus mixed v1/v2 join
  source combinations.

Comparison rules follow the ordering promise of each query:

* queries with ``ORDER BY`` must match row-by-row and column-by-column,
  including NULL positions, value types (bool vs int64 vs float64 vs
  utf8 are tagged separately) and exact float bit patterns (``0.0`` and
  ``-0.0`` are distinct; no fuzzy tolerance anywhere);
* queries without an ordering promise are reconciled as *multisets* of
  whole row values, so a harmless reordering never fails the test but
  lost or duplicated rows always do;
* every assertion group also checks the output column names, logical
  types and nullability, so projection pruning or alias drift is caught.

Top-N scenarios always sort by keys that determine a total order (ties
in the leading key are broken by a unique trailing key), so no
undefined tie order is frozen into a contract.

Cross-query state leakage is probed by re-running a baseline query after:
fully consuming an intervening result, partially consuming a result and
abandoning it, running a legal query that returns an empty set, and
interleaving several different queries (including both join strategies).

All fixtures are fixed literals: no wall clock, no randomness, no
directory enumeration order and no locale dependence.  Temporary files
live in pytest's ``tmp_path`` and are released after the module.

Only syntax and operators documented as supported by the current public
entries are used; anything the entries do not support (e.g. simple CASE,
subqueries, compound ON) is intentionally absent rather than
test-driven into the product, and scenario names state exactly what
they cover.
"""

from __future__ import annotations

import gc
import json
import struct
import subprocess
import sys
from collections import Counter
from typing import NamedTuple

import pytest

from columnar_analytics import (
    ColumnSchema,
    Schema,
    Table,
    inspect_row_groups,
    query_file,
    query_files,
    write_file,
    write_partitioned_file,
)

# ---------------------------------------------------------------------------
# Fixed datasets (explicit literals only — fully deterministic)
# ---------------------------------------------------------------------------
#
# t: 24 rows.  ``n`` repeats values at distant row positions so duplicate
# group/join keys land in different row groups for every v2 batching; ``f``
# mixes 0.0/-0.0, duplicated values (Top-N ties) and NULLs; ``s`` mixes
# duplicates, empty strings, Unicode and NULLs.
#
# u: join partner with duplicate keys (3 and 7), NULL keys and unmatched
# keys (90, 95).  w: third table joining u.k (int64) on wk (float64),
# including a -0.0 key (matches 0.0) and an unmatched 42.5.

T_SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("f", "float64", nullable=True),
        ColumnSchema("s", "utf8", nullable=True),
        ColumnSchema("flag", "bool"),
    ]
)

T_DATA = {
    "id": list(range(24)),
    "n": [
        3, None, -2, 7, 3, 0, None, 7, -2, 3, 11, None,
        7, 0, 3, -2, None, 7, 3, 0, 11, None, -2, 7,
    ],
    "f": [
        1.5, None, -0.0, 2.75, 1.5, 0.0, None, -3.25, 1.5, 2.75, None, 0.0,
        -0.0, 4.5, 1.5, None, 2.75, -3.25, 0.0, 1.5, None, 4.5, -0.0, 2.75,
    ],
    "s": [
        "alpha", None, "beta", "", "gamma", "alpha", None, "beta",
        "名前", "alpha", "", None, "gamma", "beta", "alpha", None,
        "café", "gamma", "alpha", "", None, "beta", "名前", "alpha",
    ],
    "flag": [i % 2 == 0 for i in range(24)],
}

U_SCHEMA = Schema(
    [
        ColumnSchema("uid", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("tag", "utf8", nullable=True),
    ]
)

U_DATA = {
    "uid": list(range(100, 112)),
    "k": [3, 3, None, 7, -2, 90, 0, 7, None, 11, 3, 95],
    "tag": ["x", None, "y", "x", None, "z", "y", None, "x", "z", None, "y"],
}

W_SCHEMA = Schema(
    [
        ColumnSchema("wid", "int64"),
        ColumnSchema("wk", "float64", nullable=True),
        ColumnSchema("wtag", "utf8", nullable=True),
    ]
)

W_DATA = {
    "wid": list(range(200, 209)),
    "wk": [3.0, None, 7.0, -0.0, 42.5, -2.0, 3.0, 11.0, None],
    "wtag": ["p", "q", None, "p", None, "q", "p", None, "q"],
}

T_TABLE = Table(T_SCHEMA, T_DATA)
U_TABLE = Table(U_SCHEMA, U_DATA)
W_TABLE = Table(W_SCHEMA, W_DATA)
EMPTY_TABLE = Table(T_SCHEMA, {name: [] for name in T_SCHEMA.names})

# ---------------------------------------------------------------------------
# Equivalent storage layouts: v1 vs v2 with different batch divisions
# ---------------------------------------------------------------------------

# (label, kind, row_group_size, compression, dictionary-encoded columns)
T_VARIANTS = [
    ("v1_none_plain", "v1", None, "none", ()),
    ("v1_zlib_dict", "v1", None, "zlib", ("s",)),
    ("v2_r1_none_plain", "v2", 1, "none", ()),
    ("v2_r3_zlib_dict", "v2", 3, "zlib", ("s",)),
    ("v2_r5_none_plain", "v2", 5, "none", ()),
    ("v2_r24_zlib_plain", "v2", 24, "zlib", ()),
]

U_VARIANTS = [
    ("v1", "v1", None, "none", ()),
    ("v2_r2", "v2", 2, "zlib", ("tag",)),
    ("v2_r7", "v2", 7, "none", ()),
]

W_VARIANTS = [
    ("v1", "v1", None, "none", ()),
    ("v2_r4", "v2", 4, "zlib", ("wtag",)),
]

EMPTY_VARIANTS = [
    ("v1", "v1", None, "none", ()),
    ("v2_r3", "v2", 3, "zlib", ("s",)),
]

# Join source-layout combinations: all v1, all v2, and a v1/v2 mix.
JOIN_COMBOS = [
    ("all_v1", {"t": "v1_none_plain", "u": "v1", "w": "v1"}),
    ("all_v2", {"t": "v2_r3_zlib_dict", "u": "v2_r2", "w": "v2_r4"}),
    ("mixed_v1_v2", {"t": "v2_r1_none_plain", "u": "v1", "w": "v2_r4"}),
]
JOIN_COMBO_MAP = dict(JOIN_COMBOS)

# Layout used for the same-process / fresh-context / leakage runs.
DEFAULT_SINGLE = "v2_r3_zlib_dict"
DEFAULT_EMPTY = "v2_r3"
DEFAULT_JOIN = "all_v2"


def _write_variant(path, table, kind, group_size, compression, dict_columns):
    if kind == "v1":
        write_file(
            path, table, compression=compression, dictionary_encoding=dict_columns
        )
    else:
        write_partitioned_file(
            path,
            table,
            group_size,
            compression=compression,
            dictionary_encoding=dict_columns,
        )


@pytest.fixture(scope="module")
def layouts(tmp_path_factory):
    """Write every layout variant of every table once, via public entries."""
    directory = tmp_path_factory.mktemp("repeatability_layouts")
    paths: dict[str, dict[str, object]] = {"t": {}, "u": {}, "w": {}, "empty": {}}
    for label, kind, group_size, compression, dict_columns in T_VARIANTS:
        path = directory / f"t_{label}.caef"
        _write_variant(path, T_TABLE, kind, group_size, compression, dict_columns)
        paths["t"][label] = path
    for label, kind, group_size, compression, dict_columns in U_VARIANTS:
        path = directory / f"u_{label}.caef"
        _write_variant(path, U_TABLE, kind, group_size, compression, dict_columns)
        paths["u"][label] = path
    for label, kind, group_size, compression, dict_columns in W_VARIANTS:
        path = directory / f"w_{label}.caef"
        _write_variant(path, W_TABLE, kind, group_size, compression, dict_columns)
        paths["w"][label] = path
    for label, kind, group_size, compression, dict_columns in EMPTY_VARIANTS:
        path = directory / f"empty_{label}.caef"
        _write_variant(path, EMPTY_TABLE, kind, group_size, compression, dict_columns)
        paths["empty"][label] = path
    return paths


# ---------------------------------------------------------------------------
# Result views and reconciliation helpers (public Table / CLI JSON only)
# ---------------------------------------------------------------------------


def _value_key(declared_type: str, value):
    """A hashable, type-tagged, exact comparison key for one cell.

    bool / int64 / float64 / utf8 are distinct domains (``True``, ``1``
    and ``1.0`` never compare equal here), and float64 is keyed by its
    exact bit pattern so ``0.0`` and ``-0.0`` stay distinct.  No fuzzy
    numeric tolerance is applied anywhere.
    """
    if value is None:
        return ("null",)
    if declared_type == "bool":
        return ("bool", bool(value))
    if declared_type == "int64":
        return ("int", int(value))
    if declared_type == "float64":
        return ("float", struct.pack(">d", float(value)))
    if declared_type == "utf8":
        return ("str", str(value))
    raise AssertionError(f"unexpected logical type {declared_type!r}")


class ResultView(NamedTuple):
    schema: tuple  # ((name, logical type, nullable), ...) in output order
    keys: tuple  # type-tagged comparison keys, one tuple per row
    raw: tuple  # raw value tuples, aligned with keys (for diagnostics)


def table_view(table: Table) -> ResultView:
    schema = tuple(
        (col.name, col.type, col.nullable) for col in table.schema.columns
    )
    types = [entry[1] for entry in schema]
    columns = [table.column(name) for name in table.column_names]
    keys = []
    raw = []
    for r in range(table.row_count):
        row = tuple(column[r] for column in columns)
        raw.append(row)
        keys.append(
            tuple(_value_key(types[i], row[i]) for i in range(len(types)))
        )
    return ResultView(schema, tuple(keys), tuple(raw))


def cli_json_view(payload) -> ResultView:
    schema = tuple(
        (col["name"], col["type"], col["nullable"]) for col in payload["columns"]
    )
    types = [entry[1] for entry in schema]
    keys = []
    raw = []
    for row in payload["rows"]:
        row = tuple(row)
        raw.append(row)
        keys.append(
            tuple(_value_key(types[i], row[i]) for i in range(len(types)))
        )
    return ResultView(schema, tuple(keys), tuple(raw))


def _raw_by_key(view: ResultView) -> dict:
    exemplars = {}
    for key, row in zip(view.keys, view.raw):
        exemplars.setdefault(key, row)
    return exemplars


def _first_row_diff(expected: ResultView, actual: ResultView) -> str:
    if len(expected.keys) != len(actual.keys):
        return (
            f"row count differs: expected {len(expected.keys)}, "
            f"got {len(actual.keys)}"
        )
    for i, (exp_key, act_key) in enumerate(zip(expected.keys, actual.keys)):
        if exp_key != act_key:
            return (
                f"first differing row is #{i}:\n"
                f"expected: {expected.raw[i]}\ngot: {actual.raw[i]}"
            )
    return "rows differ"


def _multiset_diff(expected: ResultView, actual: ResultView) -> str:
    expected_counts = Counter(expected.keys)
    actual_counts = Counter(actual.keys)
    exp_raw = _raw_by_key(expected)
    act_raw = _raw_by_key(actual)

    def render(counts, raw):
        lines = []
        for key, count in list(counts.items())[:5]:
            lines.append(f"  x{count} {raw[key]}")
        return "\n".join(lines) if lines else "  (none)"

    missing = expected_counts - actual_counts
    unexpected = actual_counts - expected_counts
    return (
        f"expected {sum(expected_counts.values())} rows, "
        f"got {sum(actual_counts.values())}\n"
        f"missing rows (expected but not produced):\n{render(missing, exp_raw)}\n"
        f"unexpected rows (produced but not expected):\n{render(unexpected, act_raw)}"
    )


def assert_same_result(
    expected: ResultView, actual: ResultView, *, ordered: bool, context: str
):
    """Reconcile two executions of the same query over the same data."""
    assert actual.schema == expected.schema, (
        f"output column names/logical types differ ({context})\n"
        f"expected: {expected.schema}\ngot: {actual.schema}"
    )
    if ordered:
        assert actual.keys == expected.keys, (
            f"ordered result is not row-by-row identical ({context})\n"
            f"{_first_row_diff(expected, actual)}"
        )
    else:
        assert Counter(actual.keys) == Counter(expected.keys), (
            f"unordered result multiset differs ({context})\n"
            f"{_multiset_diff(expected, actual)}"
        )


# ---------------------------------------------------------------------------
# Scenarios: only syntax/operators documented as supported by the public
# entries.  `ordered=True` means the query carries an ORDER BY that (with
# the unique trailing keys used here) determines a total row order.
# ---------------------------------------------------------------------------


class Scenario(NamedTuple):
    name: str
    kind: str  # "single" | "join" | "empty"
    sql: str
    ordered: bool


SCENARIOS = [
    # -- full table scan --------------------------------------------------
    Scenario("scan_full_table_unordered", "single", "SELECT * FROM input", False),
    # -- predicate filter + projection ------------------------------------
    Scenario(
        "filter_projection_unordered",
        "single",
        "SELECT id, n, s FROM input WHERE n >= 0 AND s IS NOT NULL",
        False,
    ),
    Scenario(
        "filter_projection_ordered",
        "single",
        "SELECT id, n, f, s FROM input "
        "WHERE (n < 5 OR f >= 1.5) AND s IS NOT NULL ORDER BY id",
        True,
    ),
    # -- scalar expressions (arithmetic, unary, CASE) with aliases --------
    Scenario(
        "scalar_expression_case_ordered",
        "single",
        "SELECT id, n * 2 + 1 AS calc, -n AS neg, f / 2.0 AS half, "
        "CASE WHEN n IS NULL THEN 'none' WHEN n > 5 THEN 'big' ELSE 'small' "
        "END AS bucket FROM input WHERE id < 18 ORDER BY id",
        True,
    ),
    # -- multi-key sort with explicit NULL placement, total order via id --
    Scenario(
        "sort_multikey_total_order",
        "single",
        "SELECT id, n, s FROM input "
        "ORDER BY n DESC NULLS FIRST, s ASC NULLS LAST, id ASC",
        True,
    ),
    # -- Top-N: ties in f are broken by the unique id ----------------------
    Scenario(
        "topn_limit_ties_total_order",
        "single",
        "SELECT id, f FROM input WHERE f IS NOT NULL "
        "ORDER BY f DESC NULLS LAST, id ASC LIMIT 7",
        True,
    ),
    Scenario(
        "topn_limit_zero",
        "single",
        "SELECT id, n FROM input ORDER BY id LIMIT 0",
        True,
    ),
    # -- grouping / aggregation (NULL group, empty-string group, Unicode) --
    Scenario(
        "group_aggregate_unordered",
        "single",
        "SELECT s, COUNT(*), COUNT(n), SUM(n), AVG(f), MIN(n), MAX(id) "
        "FROM input GROUP BY s",
        False,
    ),
    Scenario(
        "group_having_order_limit_ordered",
        "single",
        "SELECT s, COUNT(*), SUM(n) FROM input GROUP BY s "
        "HAVING COUNT(*) >= 2 ORDER BY COUNT(*) DESC, s ASC NULLS FIRST LIMIT 4",
        True,
    ),
    Scenario(
        "global_aggregate_zero_rows",
        "single",
        "SELECT COUNT(*), COUNT(n), SUM(n), AVG(f), MIN(s), MAX(id) "
        "FROM input WHERE id > 1000",
        False,
    ),
    Scenario(
        "group_by_zero_rows",
        "single",
        "SELECT n, COUNT(*) FROM input WHERE id > 1000 GROUP BY n",
        False,
    ),
    # -- SELECT DISTINCT whole-row deduplication ---------------------------
    Scenario(
        "distinct_whole_row_unordered",
        "single",
        "SELECT DISTINCT n, flag FROM input",
        False,
    ),
    Scenario(
        "filter_empty_result_ordered",
        "single",
        "SELECT id, n FROM input WHERE id > 1000 ORDER BY id",
        True,
    ),
    # -- joins: duplicate keys, NULL keys, unmatched keys, mixed key types --
    Scenario(
        "join_inner_duplicate_keys_unordered",
        "join",
        "SELECT t.id, u.uid FROM t INNER JOIN u ON t.n = u.k",
        False,
    ),
    Scenario(
        "join_inner_filter_ordered",
        "join",
        "SELECT t.id, u.uid, t.n, u.tag FROM t INNER JOIN u ON t.n = u.k "
        "WHERE t.id < 20 ORDER BY t.id, u.uid",
        True,
    ),
    Scenario(
        "join_left_unmatched_unordered",
        "join",
        "SELECT t.id, u.uid, u.tag FROM t LEFT JOIN u ON t.n = u.k",
        False,
    ),
    Scenario(
        "join_right_ordered",
        "join",
        "SELECT t.id, u.uid FROM t RIGHT JOIN u ON t.n = u.k "
        "ORDER BY u.uid, t.id NULLS LAST",
        True,
    ),
    Scenario(
        "join_full_outer_unordered",
        "join",
        "SELECT t.id, u.uid FROM t FULL OUTER JOIN u ON t.n = u.k",
        False,
    ),
    Scenario(
        "join_chain_mixed_key_group_unordered",
        "join",
        "SELECT t.flag, COUNT(*), SUM(u.uid) FROM t "
        "INNER JOIN u ON t.n = u.k INNER JOIN w ON u.k = w.wk "
        "GROUP BY t.flag",
        False,
    ),
    Scenario(
        "join_left_group_having_distinct_agg_ordered",
        "join",
        "SELECT u.tag, COUNT(*), COUNT(DISTINCT t.id) FROM t "
        "LEFT JOIN u ON t.n = u.k GROUP BY u.tag "
        "HAVING COUNT(*) >= 2 ORDER BY u.tag NULLS FIRST",
        True,
    ),
    # -- empty input -------------------------------------------------------
    Scenario("empty_scan_unordered", "empty", "SELECT * FROM input", False),
    Scenario(
        "empty_global_aggregate",
        "empty",
        "SELECT COUNT(*), COUNT(n), SUM(n), AVG(f), MIN(s) FROM input",
        False,
    ),
    Scenario(
        "empty_group_by",
        "empty",
        "SELECT n, COUNT(*) FROM input GROUP BY n",
        False,
    ),
]

SCENARIO_IDS = [scenario.name for scenario in SCENARIOS]


def _variant_labels(kind: str) -> list[str]:
    if kind == "single":
        return [spec[0] for spec in T_VARIANTS]
    if kind == "empty":
        return [spec[0] for spec in EMPTY_VARIANTS]
    return [name for name, _combo in JOIN_COMBOS]


def _default_label(kind: str) -> str:
    if kind == "single":
        return DEFAULT_SINGLE
    if kind == "empty":
        return DEFAULT_EMPTY
    return DEFAULT_JOIN


def run_scenario(scenario: Scenario, layouts, label: str) -> Table:
    if scenario.kind == "join":
        combo = JOIN_COMBO_MAP[label]
        sources = {name: layouts[name][variant] for name, variant in combo.items()}
        return query_files(sources, scenario.sql)
    key = "empty" if scenario.kind == "empty" else "t"
    return query_file(layouts[key][label], scenario.sql)


# ---------------------------------------------------------------------------
# 1. Same data + same query, consecutive executions in one process
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_IDS)
def test_repeated_execution_same_process(layouts, scenario):
    label = _default_label(scenario.kind)
    first = table_view(run_scenario(scenario, layouts, label))
    for repeat in (2, 3):
        again = table_view(run_scenario(scenario, layouts, label))
        assert_same_result(
            first,
            again,
            ordered=scenario.ordered,
            context=f"{scenario.name} / {label} / same-process repeat {repeat}",
        )


# ---------------------------------------------------------------------------
# 2. Same data + same query in a recreated query context (fresh interpreter
#    running the installed CLI), reconciled against the in-process result
# ---------------------------------------------------------------------------


def _cli_args(scenario: Scenario, layouts, label: str) -> list[str]:
    if scenario.kind == "join":
        combo = JOIN_COMBO_MAP[label]
        sources = {name: str(layouts[name][variant]) for name, variant in combo.items()}
        return ["query-files", json.dumps(sources), scenario.sql]
    key = "empty" if scenario.kind == "empty" else "t"
    return ["query", str(layouts[key][label]), scenario.sql]


def _run_cli(args: list[str]) -> bytes:
    proc = subprocess.run(
        [sys.executable, "-m", "columnar_analytics.cli", *args],
        capture_output=True,
    )
    assert proc.returncode == 0, (
        f"CLI failed with code {proc.returncode}: "
        f"{proc.stderr.decode('utf-8', errors='replace')}"
    )
    return proc.stdout


@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_IDS)
def test_repeated_execution_fresh_context(layouts, scenario):
    label = _default_label(scenario.kind)
    args = _cli_args(scenario, layouts, label)
    # Two brand-new interpreter contexts must agree byte-for-byte...
    first_bytes = _run_cli(args)
    second_bytes = _run_cli(args)
    assert first_bytes == second_bytes, (
        f"repeated CLI executions are not byte-identical ({scenario.name})\n"
        f"SQL: {scenario.sql}"
    )
    # ...and the fresh context must reconcile with the in-process result.
    in_process = table_view(run_scenario(scenario, layouts, label))
    fresh = cli_json_view(json.loads(first_bytes))
    assert_same_result(
        in_process,
        fresh,
        ordered=scenario.ordered,
        context=f"{scenario.name} / {label} / fresh CLI context",
    )


# ---------------------------------------------------------------------------
# 3. Same data + same query under equivalent batch divisions
#    (v1 vs v2 row-group sizes; mixed join source layouts)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIO_IDS)
def test_equivalent_batch_layouts_agree(layouts, scenario):
    labels = _variant_labels(scenario.kind)
    baseline = table_view(run_scenario(scenario, layouts, labels[0]))
    for label in labels[1:]:
        view = table_view(run_scenario(scenario, layouts, label))
        assert_same_result(
            baseline,
            view,
            ordered=scenario.ordered,
            context=f"{scenario.name} / layouts {labels[0]} vs {label}",
        )


# ---------------------------------------------------------------------------
# 4. No execution state leaks into later queries
# ---------------------------------------------------------------------------
#
# Each baseline is fully consumed (materialised into a view), then a
# disturbance runs, then the baseline runs again and must reconcile.

LEAK_BASELINES = [
    Scenario(
        "baseline_single_ordered",
        "single",
        "SELECT id, n, f, s FROM input WHERE n >= -2 OR n IS NULL ORDER BY id",
        True,
    ),
    Scenario(
        "baseline_join_group_unordered",
        "join",
        "SELECT t.flag, COUNT(*), SUM(u.uid) FROM t "
        "INNER JOIN u ON t.n = u.k GROUP BY t.flag",
        False,
    ),
]

LEAK_BASELINE_IDS = [baseline.name for baseline in LEAK_BASELINES]


def _baseline_view(layouts, baseline: Scenario) -> ResultView:
    return table_view(run_scenario(baseline, layouts, _default_label(baseline.kind)))


def _assert_baseline_unchanged(layouts, baseline: Scenario, first: ResultView, disturbance: str):
    again = _baseline_view(layouts, baseline)
    assert_same_result(
        first,
        again,
        ordered=baseline.ordered,
        context=f"{baseline.name} re-run after {disturbance}",
    )


@pytest.mark.parametrize("baseline", LEAK_BASELINES, ids=LEAK_BASELINE_IDS)
def test_no_state_leak_after_full_consumption(layouts, baseline):
    # The baseline itself is consumed in full, then run once more.
    first = _baseline_view(layouts, baseline)
    _assert_baseline_unchanged(layouts, baseline, first, "full consumption")


@pytest.mark.parametrize("baseline", LEAK_BASELINES, ids=LEAK_BASELINE_IDS)
def test_no_state_leak_after_partial_consumption(layouts, baseline):
    first = _baseline_view(layouts, baseline)
    # Run other queries, consume only a fragment of each result, abandon.
    partial_single = query_file(
        layouts["t"][DEFAULT_SINGLE],
        "SELECT s, COUNT(*) FROM input GROUP BY s ORDER BY s NULLS FIRST",
    )
    _ = partial_single.column("s")[:2]
    _ = partial_single.row_count
    combo = JOIN_COMBO_MAP[DEFAULT_JOIN]
    sources = {name: layouts[name][variant] for name, variant in combo.items()}
    partial_join = query_files(
        sources, "SELECT t.id, u.uid FROM t INNER JOIN u ON t.n = u.k"
    )
    _ = partial_join.column("t.id")[:1]
    del partial_single, partial_join
    gc.collect()
    _assert_baseline_unchanged(layouts, baseline, first, "partial consumption")


@pytest.mark.parametrize("baseline", LEAK_BASELINES, ids=LEAK_BASELINE_IDS)
def test_no_state_leak_after_empty_result(layouts, baseline):
    first = _baseline_view(layouts, baseline)
    # A legal query with an empty result, fully consumed, then the baseline.
    empty = query_file(
        layouts["t"][DEFAULT_SINGLE],
        "SELECT id, n FROM input WHERE id > 100000 ORDER BY id",
    )
    empty_view = table_view(empty)
    assert empty_view.keys == ()
    assert empty_view.schema == (("id", "int64", False), ("n", "int64", True))
    _assert_baseline_unchanged(layouts, baseline, first, "empty result")


@pytest.mark.parametrize("baseline", LEAK_BASELINES, ids=LEAK_BASELINE_IDS)
def test_no_state_leak_after_interleaved_queries(layouts, baseline):
    first = _baseline_view(layouts, baseline)
    combo = JOIN_COMBO_MAP[DEFAULT_JOIN]
    sources = {name: layouts[name][variant] for name, variant in combo.items()}
    interleaved = [
        lambda: query_file(
            layouts["t"][DEFAULT_SINGLE],
            "SELECT COUNT(*), AVG(f), MIN(s) FROM input",
        ),
        lambda: query_file(
            layouts["t"][DEFAULT_SINGLE], "SELECT DISTINCT flag FROM input"
        ),
        lambda: query_file(
            layouts["t"][DEFAULT_SINGLE],
            "SELECT id, f FROM input ORDER BY id DESC LIMIT 3",
        ),
        lambda: query_files(
            sources,
            "SELECT t.id, u.uid FROM t INNER JOIN u ON t.n = u.k "
            "ORDER BY t.id, u.uid",
            "hash",
        ),
        lambda: query_files(
            sources,
            "SELECT t.id, u.uid FROM t INNER JOIN u ON t.n = u.k "
            "ORDER BY t.id, u.uid",
            "sort_merge",
        ),
        lambda: query_files(
            sources,
            "SELECT u.tag, COUNT(*) FROM t LEFT JOIN u ON t.n = u.k "
            "GROUP BY u.tag",
        ),
    ]
    for run in interleaved:
        table_view(run())  # fully consume each interleaved result
    _assert_baseline_unchanged(layouts, baseline, first, "interleaved queries")


# ---------------------------------------------------------------------------
# 5. Evidence the "equivalent batch divisions" really are different batches
# ---------------------------------------------------------------------------


def test_batch_layouts_actually_differ(layouts):
    # v1 carries no row groups; each v2 variant splits the same 24 rows
    # into the documented consecutive batches.
    assert inspect_row_groups(layouts["t"]["v1_none_plain"]) == []
    expected_counts = {
        "v2_r1_none_plain": [1] * 24,
        "v2_r3_zlib_dict": [3] * 8,
        "v2_r5_none_plain": [5, 5, 5, 5, 4],
        "v2_r24_zlib_plain": [24],
    }
    for label, counts in expected_counts.items():
        groups = inspect_row_groups(layouts["t"][label])
        assert [group["row_count"] for group in groups] == counts
    # Duplicate group/join keys genuinely span batches (e.g. n = 3 sits in
    # four different size-5 groups), so cross-batch reconciliation is real.
    positions = [i for i, value in enumerate(T_DATA["n"]) if value == 3]
    assert len({i // 5 for i in positions}) > 1
    assert len({i // 3 for i in positions}) > 1
