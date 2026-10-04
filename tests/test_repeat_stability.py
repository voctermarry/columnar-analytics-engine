"""Repeat-execution stability and reconciliation tests (结果对账).

This module pins one contract: *the same query over the same data always
produces the same, fully reconcilable result*, no matter how often and how it
is re-run.  It does not add any public interface and does not change existing
semantics; it only exercises the public entry points
(``write_file`` / ``write_partitioned_file`` / ``read_file`` / ``query_file`` /
``query_files`` and the ``query`` CLI command).

Coverage (only paths the current public entries can express):

* full-table scan, predicate filter + projection, scalar expressions
  (arithmetic + searched ``CASE``), grouped aggregation (+ ``HAVING``),
  multi-key ``ORDER BY``, Top-N (``ORDER BY`` + ``LIMIT``), ``SELECT
  DISTINCT``, global aggregates, and equi-joins (INNER/LEFT/FULL OUTER,
  including an empty right side) via ``query_files``.
* Syntax/operators the public entries do not support (subqueries, simple
  CASE, non-equi joins, ...) are intentionally *absent* from these scenarios
  rather than emulated here; the scenario and test names state exactly what
  is covered.

Comparison rules:

* Queries with ``ORDER BY`` must match row-by-row and column-by-column,
  including NULL positions, value types (``bool`` vs ``int64``), float zero
  signs, string content and duplicate-row order.  Top-N scenarios always
  carry a unique tiebreaker key so the total order is defined; no undefined
  tie order is pinned as a product contract.
* Queries without an ordering promise are compared across storage layouts as
  a *multiset* of complete row values (reordering is not a failure, lost or
  duplicated rows are).  Within one layout the engine promises
  byte-identical repeats, so same-layout repeats are compared exactly.
* Every assertion group also checks the output column names and logical
  types, so projection-pruning or alias drift is caught.
* Float results use exact value equality (down to the zero sign); no fuzzy
  tolerance is introduced.

Stability axes per scenario:

1. same query, same process, consecutive runs;
2. a freshly created query context (a brand-new CLI interpreter process);
3. equivalent batch partitions of the same data (v1 file vs v2 row-group
   sizes 1 / 3 / 64) and, for joins, the ``hash`` / ``sort_merge`` /
   default strategies.

State-leakage checks: fully consuming other queries, consuming only part of
a result and then abandoning it (in-process and via a CLI subprocess whose
output is cut off mid-stream), running after a legal empty-result query, and
interleaving different (including failing) queries must all leave the
benchmark query result identical to its first full execution.

All fixtures are fixed literals: no clock, no randomness, no directory
enumeration order, no locale dependence.  Temporary files live in pytest's
``tmp_path`` and are released after the module finishes.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

import columnar_analytics
from columnar_analytics import (
    ColumnSchema,
    QuerySyntaxError,
    QueryValidationError,
    Schema,
    Table,
    query_file,
    query_files,
    read_file,
    write_file,
    write_partitioned_file,
)

# ---------------------------------------------------------------------------
# Fixed data fixtures (literal values only — reproducible by inspection)
# ---------------------------------------------------------------------------

MAIN_SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("grp", "utf8", nullable=True),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("f", "float64", nullable=True),
        ColumnSchema("s", "utf8", nullable=True),
        ColumnSchema("flag", "bool"),
    ]
)

# 24 rows.  ``grp`` keys repeat and straddle every row-group boundary for
# row_group_size 1 and 3 (cross-batch same keys); ``n``/``f`` carry duplicate
# values (ties) and NULLs; ``f`` carries both zero signs; ``s`` mixes
# duplicates, the empty string, NULLs and non-ASCII text.
MAIN_DATA = {
    "id": list(range(1, 25)),
    "grp": [
        "alpha", "beta", None, "gamma", "alpha", "beta", "delta", None,
        "gamma", "alpha", "beta", None, "alpha", "gamma", "beta", None,
        "alpha", "beta", "gamma", None, "alpha", "beta", "gamma", "alpha",
    ],
    "n": [
        3, None, 0, 5, -2, 3, 1, None,
        4, 0, -2, 3, 5, 1, None, 2,
        3, 0, 4, -2, 1, 5, None, 3,
    ],
    "f": [
        1.5, -0.0, None, 2.5, 1.5, 0.0, -2.25, None,
        3.5, 1.5, -0.0, 2.5, None, 1.5, 0.0, -2.25,
        3.5, None, 1.5, 2.5, -0.0, 0.0, 1.5, None,
    ],
    "s": [
        "a", "b", None, "a", "", "名前", "a", None,
        "b", "a", "", "b", None, "a", "名前", "b",
        "a", None, "", "a", "b", "名前", None, "a",
    ],
    "flag": [True, False] * 12,
}

L_SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("n", "int64", nullable=True),
    ]
)
L_DATA = {
    "id": list(range(1, 11)),
    # duplicate keys (1 x2, 2 x3), NULL keys, keys with no right match (4, 5)
    "k": [1, 2, 2, None, 3, 1, None, 4, 2, 5],
    "n": [10, 20, None, 40, 50, 60, None, 80, 90, 100],
}

R_SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("tag", "utf8", nullable=True),
    ]
)
R_DATA = {
    "id": list(range(1, 9)),
    # duplicate keys (2 x3), NULL keys, a key with no left match (6)
    "k": [2, 2, 1, None, 3, 2, None, 6],
    "tag": ["x", "y", "z", None, "w", "v", None, "u"],
}

MAIN_SCHEMA_DESC = [(c.name, c.type, c.nullable) for c in MAIN_SCHEMA.columns]

# ---------------------------------------------------------------------------
# Scenarios (names state exactly what is covered)
# ---------------------------------------------------------------------------

BENCHMARK_SQL = (
    "SELECT id, n + 7 AS n_plus, f * 2.0 AS f_double, "
    "CASE WHEN n IS NULL THEN 'none' WHEN n >= 3 THEN 'high' ELSE 'low' END "
    "AS band FROM input ORDER BY id"
)

SINGLE_SCENARIOS = [
    {
        "name": "full_scan_star",
        "sql": "SELECT * FROM input",
        "ordered": False,
        "schema": MAIN_SCHEMA_DESC,
    },
    {
        "name": "filter_and_projection",
        "sql": "SELECT id, s, flag FROM input WHERE n >= 1 AND s IS NOT NULL",
        "ordered": False,
        "schema": [("id", "int64", False), ("s", "utf8", True), ("flag", "bool", False)],
    },
    {
        "name": "scalar_expressions_and_case_ordered",
        "sql": BENCHMARK_SQL,
        "ordered": True,
        "schema": [
            ("id", "int64", False),
            ("n_plus", "int64", True),
            ("f_double", "float64", True),
            ("band", "utf8", False),
        ],
    },
    {
        "name": "where_scalar_expression_filter_ordered",
        "sql": "SELECT id, n FROM input WHERE n * 2 + 1 > 5 ORDER BY id DESC",
        "ordered": True,
        "schema": [("id", "int64", False), ("n", "int64", True)],
    },
    {
        "name": "group_aggregation_unordered",
        "sql": (
            "SELECT grp, COUNT(*), COUNT(n), SUM(n), AVG(f), MIN(s), MAX(flag) "
            "FROM input GROUP BY grp"
        ),
        "ordered": False,
        "schema": [
            ("grp", "utf8", True),
            ("COUNT(*)", "int64", False),
            ("COUNT(n)", "int64", False),
            ("SUM(n)", "int64", True),
            ("AVG(f)", "float64", True),
            ("MIN(s)", "utf8", True),
            ("MAX(flag)", "bool", True),
        ],
    },
    {
        "name": "group_aggregation_ordered",
        "sql": (
            "SELECT grp, COUNT(*), SUM(n) FROM input GROUP BY grp "
            "ORDER BY grp ASC NULLS FIRST"
        ),
        "ordered": True,
        "schema": [("grp", "utf8", True), ("COUNT(*)", "int64", False), ("SUM(n)", "int64", True)],
        "rows": [
            (None, 5, 3),
            ("alpha", 7, 13),
            ("beta", 6, 6),
            ("delta", 1, 1),
            ("gamma", 5, 14),
        ],
    },
    {
        "name": "group_aggregation_having_ordered",
        "sql": (
            "SELECT grp, COUNT(*), SUM(n) FROM input GROUP BY grp "
            "HAVING COUNT(*) >= 2 AND SUM(n) IS NOT NULL "
            "ORDER BY grp ASC NULLS LAST"
        ),
        "ordered": True,
        "schema": [("grp", "utf8", True), ("COUNT(*)", "int64", False), ("SUM(n)", "int64", True)],
        "rows": [("alpha", 7, 13), ("beta", 6, 6), ("gamma", 5, 14), (None, 5, 3)],
    },
    {
        "name": "global_aggregates_single_row",
        "sql": "SELECT COUNT(*), COUNT(n), SUM(n), AVG(n), MIN(f), MAX(f) FROM input",
        "ordered": False,
        "schema": [
            ("COUNT(*)", "int64", False),
            ("COUNT(n)", "int64", False),
            ("SUM(n)", "int64", True),
            ("AVG(n)", "float64", True),
            ("MIN(f)", "float64", True),
            ("MAX(f)", "float64", True),
        ],
        "rows": [(24, 20, 37, 37 / 20, -2.25, 3.5)],
    },
    {
        "name": "sort_multi_key_with_nulls_and_ties_ordered",
        "sql": (
            "SELECT id, f, s FROM input "
            "ORDER BY f ASC NULLS FIRST, s DESC NULLS LAST, id ASC"
        ),
        "ordered": True,
        "schema": [("id", "int64", False), ("f", "float64", True), ("s", "utf8", True)],
    },
    {
        "name": "top_n_limit_with_unique_tiebreaker_ordered",
        "sql": (
            "SELECT id, n FROM input WHERE n IS NOT NULL "
            "ORDER BY n DESC NULLS LAST, id ASC LIMIT 5"
        ),
        "ordered": True,
        "schema": [("id", "int64", False), ("n", "int64", True)],
        "rows": [(4, 5), (13, 5), (22, 5), (9, 4), (19, 4)],
    },
    {
        "name": "select_distinct_unordered",
        "sql": "SELECT DISTINCT s FROM input",
        "ordered": False,
        "schema": [("s", "utf8", True)],
        "rows_multiset": [("a",), ("b",), (None,), ("",), ("名前",)],
    },
    {
        "name": "select_distinct_ordered",
        "sql": (
            "SELECT DISTINCT grp, flag FROM input "
            "ORDER BY grp ASC NULLS LAST, flag ASC"
        ),
        "ordered": True,
        "schema": [("grp", "utf8", True), ("flag", "bool", False)],
        "rows": [
            ("alpha", False), ("alpha", True),
            ("beta", False), ("beta", True),
            ("delta", True),
            ("gamma", False), ("gamma", True),
            (None, False), (None, True),
        ],
    },
    {
        "name": "empty_result_filter_ordered",
        "sql": "SELECT id, n FROM input WHERE id > 1000000 ORDER BY id",
        "ordered": True,
        "schema": [("id", "int64", False), ("n", "int64", True)],
        "rows": [],
    },
]

EMPTY_SCENARIOS = [
    {
        "name": "empty_input_full_scan",
        "sql": "SELECT * FROM input",
        "ordered": False,
        "schema": MAIN_SCHEMA_DESC,
        "rows": [],
    },
    {
        "name": "empty_input_global_aggregates_single_row",
        "sql": "SELECT COUNT(*), COUNT(n), SUM(n), AVG(f), MIN(s), MAX(s) FROM input",
        "ordered": False,
        "schema": [
            ("COUNT(*)", "int64", False),
            ("COUNT(n)", "int64", False),
            ("SUM(n)", "int64", True),
            ("AVG(f)", "float64", True),
            ("MIN(s)", "utf8", True),
            ("MAX(s)", "utf8", True),
        ],
        "rows": [(0, 0, None, None, None, None)],
    },
    {
        "name": "empty_input_group_aggregation_zero_rows",
        "sql": "SELECT grp, COUNT(*) FROM input GROUP BY grp",
        "ordered": False,
        "schema": [("grp", "utf8", True), ("COUNT(*)", "int64", False)],
        "rows": [],
    },
    {
        "name": "empty_input_top_n_zero_rows",
        "sql": "SELECT id, n FROM input ORDER BY id DESC LIMIT 3",
        "ordered": True,
        "schema": [("id", "int64", False), ("n", "int64", True)],
        "rows": [],
    },
]

JOIN_SCENARIOS = [
    {
        "name": "inner_join_duplicate_keys_unordered",
        "sql": "SELECT l.id, l.k, r.id, r.tag FROM l INNER JOIN r ON l.k = r.k",
        "ordered": False,
        "right": "r",
        "schema": [
            ("l.id", "int64", False),
            ("l.k", "int64", True),
            ("r.id", "int64", False),
            ("r.tag", "utf8", True),
        ],
    },
    {
        "name": "inner_join_duplicate_keys_ordered",
        "sql": (
            "SELECT l.id, l.k, r.id, r.tag FROM l INNER JOIN r ON l.k = r.k "
            "ORDER BY l.id, r.id"
        ),
        "ordered": True,
        "right": "r",
        "schema": [
            ("l.id", "int64", False),
            ("l.k", "int64", True),
            ("r.id", "int64", False),
            ("r.tag", "utf8", True),
        ],
    },
    {
        "name": "left_join_unmatched_and_null_keys_ordered",
        "sql": (
            "SELECT l.id, r.id, r.tag FROM l LEFT JOIN r ON l.k = r.k "
            "ORDER BY l.id, r.id"
        ),
        "ordered": True,
        "right": "r",
        "schema": [
            ("l.id", "int64", False),
            ("r.id", "int64", True),
            ("r.tag", "utf8", True),
        ],
    },
    {
        "name": "full_outer_join_both_sides_unmatched_ordered",
        "sql": (
            "SELECT l.id, r.id FROM l FULL OUTER JOIN r ON l.k = r.k "
            "ORDER BY l.id ASC NULLS LAST, r.id ASC NULLS LAST"
        ),
        "ordered": True,
        "right": "r",
        "schema": [("l.id", "int64", True), ("r.id", "int64", True)],
    },
    {
        "name": "join_then_group_aggregation_ordered",
        "sql": (
            "SELECT r.tag, COUNT(*), SUM(l.n) FROM l INNER JOIN r ON l.k = r.k "
            "GROUP BY r.tag ORDER BY r.tag ASC NULLS FIRST"
        ),
        "ordered": True,
        "right": "r",
        "schema": [
            ("r.tag", "utf8", True),
            ("COUNT(*)", "int64", False),
            ("SUM(l.n)", "int64", True),
        ],
    },
    {
        "name": "inner_join_empty_right_zero_rows",
        "sql": "SELECT l.id, r.tag FROM l INNER JOIN r ON l.k = r.k ORDER BY l.id",
        "ordered": True,
        "right": "r_empty",
        "schema": [("l.id", "int64", False), ("r.tag", "utf8", True)],
        "rows": [],
    },
    {
        "name": "left_join_empty_right_all_nulls_ordered",
        "sql": "SELECT l.id, r.tag FROM l LEFT JOIN r ON l.k = r.k ORDER BY l.id",
        "ordered": True,
        "right": "r_empty",
        "schema": [("l.id", "int64", False), ("r.tag", "utf8", True)],
        "rows": [(i, None) for i in range(1, 11)],
    },
]

SINGLE_LAYOUTS = ["v1", "v2_g1", "v2_g3", "v2_g64"]
CLI_LAYOUTS = ["v1", "v2_g3"]
JOIN_LAYOUTS = ["v1_v1", "v2g2_v2g3", "v2g7_v2g1"]
JOIN_STRATEGIES = [None, "hash", "sort_merge"]

# ---------------------------------------------------------------------------
# Files written once per module through the public write entry points
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def main_paths(tmp_path_factory):
    directory = tmp_path_factory.mktemp("repeat_main")
    table = Table(MAIN_SCHEMA, MAIN_DATA)
    paths = {}
    path = directory / "main_v1.caef"
    write_file(path, table)
    paths["v1"] = path
    path = directory / "main_v2_g1.caef"
    write_partitioned_file(path, table, 1)
    paths["v2_g1"] = path
    path = directory / "main_v2_g3.caef"
    write_partitioned_file(
        path, table, 3, compression="zlib", dictionary_encoding=["grp", "s"]
    )
    paths["v2_g3"] = path
    path = directory / "main_v2_g64.caef"
    write_partitioned_file(path, table, 64)
    paths["v2_g64"] = path
    return paths


@pytest.fixture(scope="module")
def empty_paths(tmp_path_factory):
    directory = tmp_path_factory.mktemp("repeat_empty")
    table = Table(MAIN_SCHEMA, {name: [] for name in MAIN_SCHEMA.names})
    paths = {}
    path = directory / "empty_v1.caef"
    write_file(path, table)
    paths["v1"] = path
    path = directory / "empty_v2_g4.caef"
    write_partitioned_file(path, table, 4)
    paths["v2_g4"] = path
    return paths


@pytest.fixture(scope="module")
def join_paths(tmp_path_factory):
    directory = tmp_path_factory.mktemp("repeat_join")
    left = Table(L_SCHEMA, L_DATA)
    right = Table(R_SCHEMA, R_DATA)
    right_empty = Table(R_SCHEMA, {name: [] for name in R_SCHEMA.names})
    layouts = {}
    specs = {
        "v1_v1": (None, None),
        "v2g2_v2g3": (2, 3),
        "v2g7_v2g1": (7, 1),
    }
    for label, (l_groups, r_groups) in specs.items():
        l_path = directory / f"l_{label}.caef"
        r_path = directory / f"r_{label}.caef"
        r_empty_path = directory / f"r_empty_{label}.caef"
        if l_groups is None:
            write_file(l_path, left)
        else:
            write_partitioned_file(l_path, left, l_groups)
        if r_groups is None:
            write_file(r_path, right)
            write_file(r_empty_path, right_empty)
        else:
            write_partitioned_file(r_path, right, r_groups)
            write_partitioned_file(r_empty_path, right_empty, r_groups)
        layouts[label] = {"l": l_path, "r": r_path, "r_empty": r_empty_path}
    return layouts


# ---------------------------------------------------------------------------
# Comparison helpers (public Table surface only)
# ---------------------------------------------------------------------------


def table_view(table: Table):
    """(schema descriptor, rows) using only the public Table surface."""
    schema = [(col.name, col.type, col.nullable) for col in table.schema.columns]
    columns = [table.column(name) for name in table.column_names]
    rows = [tuple(row) for row in zip(*columns)]
    return schema, rows


def _value_key(value):
    """Type- and bit-exact comparison key.

    Distinguishes ``bool`` from ``int64`` (``True == 1`` in Python) and
    ``0.0`` from ``-0.0`` via the IEEE-754 bit pattern, so a value-type or
    zero-sign drift cannot hide behind ``==``.
    """
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", struct.pack("<d", value))
    return ("str", value)


def _norm_rows(rows):
    return [tuple(_value_key(value) for value in row) for row in rows]


def _first_row_diff(expected_rows, got_rows):
    if len(expected_rows) != len(got_rows):
        return f"row count differs: expected {len(expected_rows)}, got {len(got_rows)}"
    for i, (expected, got) in enumerate(zip(expected_rows, got_rows)):
        if _value_key_tuple(expected) != _value_key_tuple(got):
            return f"first differing row #{i}: expected {expected!r}, got {got!r}"
    return "rows differ only in value type or float zero sign"


def _value_key_tuple(row):
    return tuple(_value_key(value) for value in row)


def assert_same_view(expected, got, *, what):
    """Exact reconciliation: schema, row order, values, types, zero signs."""
    exp_schema, exp_rows = expected
    got_schema, got_rows = got
    assert got_schema == exp_schema, (
        f"output column name/type drift ({what}):\n"
        f"expected: {exp_schema}\ngot: {got_schema}"
    )
    assert _norm_rows(got_rows) == _norm_rows(exp_rows), (
        f"ordered row drift ({what}):\n{_first_row_diff(exp_rows, got_rows)}"
    )


def assert_same_multiset(expected, got, *, what):
    """Unordered reconciliation: identical schema, rows as a multiset."""
    exp_schema, exp_rows = expected
    got_schema, got_rows = got
    assert got_schema == exp_schema, (
        f"output column name/type drift ({what}):\n"
        f"expected: {exp_schema}\ngot: {got_schema}"
    )
    expected_counter = Counter(_norm_rows(exp_rows))
    got_counter = Counter(_norm_rows(got_rows))
    missing = expected_counter - got_counter
    extra = got_counter - expected_counter
    assert not missing and not extra, (
        f"row multiset drift ({what}):\n"
        f"missing: {sorted(map(repr, missing.elements()))}\n"
        f"unexpected: {sorted(map(repr, extra.elements()))}"
    )


def assert_scenario_pins(view, scenario, *, what):
    """Pin output column names/logical types and any literal row anchors."""
    got_schema, got_rows = view
    assert got_schema == scenario["schema"], (
        f"output column name/type mismatch ({what}):\n"
        f"expected: {scenario['schema']}\ngot: {got_schema}"
    )
    if "rows" in scenario:
        assert _norm_rows(got_rows) == _norm_rows(scenario["rows"]), (
            f"anchored rows mismatch ({what}):\n"
            f"{_first_row_diff(scenario['rows'], got_rows)}"
        )
    if "rows_multiset" in scenario:
        assert Counter(_norm_rows(got_rows)) == Counter(
            _norm_rows(scenario["rows_multiset"])
        ), f"anchored row multiset mismatch ({what})"


def reconcile_views(baseline, other, *, ordered, what):
    if ordered:
        assert_same_view(baseline, other, what=what)
    else:
        assert_same_multiset(baseline, other, what=what)


# ---------------------------------------------------------------------------
# Fresh-process query context through the public CLI
# ---------------------------------------------------------------------------

_PACKAGE_ROOT = str(Path(columnar_analytics.__file__).resolve().parent.parent)


def _cli_env():
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = _PACKAGE_ROOT + (os.pathsep + existing if existing else "")
    return env


def _cli_query_view(path, sql):
    proc = subprocess.run(
        [sys.executable, "-m", "columnar_analytics.cli", "query", str(path), sql],
        capture_output=True,
        env=_cli_env(),
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"CLI query failed (rc={proc.returncode}): "
        f"{proc.stderr.decode('utf-8', 'replace')}"
    )
    payload = json.loads(proc.stdout.decode("utf-8"))
    schema = [(c["name"], c["type"], c["nullable"]) for c in payload["columns"]]
    rows = [tuple(row) for row in payload["rows"]]
    return schema, rows


# ---------------------------------------------------------------------------
# 1. Same query, same process, consecutive runs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SINGLE_SCENARIOS, ids=[s["name"] for s in SINGLE_SCENARIOS])
@pytest.mark.parametrize("layout", SINGLE_LAYOUTS)
def test_same_query_repeated_in_same_process_is_identical(main_paths, scenario, layout):
    path = main_paths[layout]
    first = table_view(query_file(path, scenario["sql"]))
    assert_scenario_pins(first, scenario, what=f"{scenario['name']} / {layout}")
    for repeat in (2, 3):
        again = table_view(query_file(path, scenario["sql"]))
        assert_same_view(first, again, what=f"{scenario['name']} / {layout} / repeat {repeat}")


@pytest.mark.parametrize("scenario", JOIN_SCENARIOS, ids=[s["name"] for s in JOIN_SCENARIOS])
@pytest.mark.parametrize("layout", JOIN_LAYOUTS)
@pytest.mark.parametrize("strategy", JOIN_STRATEGIES)
def test_same_join_repeated_in_same_process_is_identical(
    join_paths, scenario, layout, strategy
):
    files = join_paths[layout]
    sources = {"l": files["l"], "r": files[scenario["right"]]}
    first = table_view(query_files(sources, scenario["sql"], strategy))
    assert_scenario_pins(
        first, scenario, what=f"{scenario['name']} / {layout} / {strategy}"
    )
    again = table_view(query_files(sources, scenario["sql"], strategy))
    assert_same_view(first, again, what=f"{scenario['name']} / {layout} / {strategy} repeat")


# ---------------------------------------------------------------------------
# 2. Equivalent batch partitions (v1 vs v2 row-group sizes) reconcile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SINGLE_SCENARIOS, ids=[s["name"] for s in SINGLE_SCENARIOS])
def test_same_query_reconciles_across_row_group_partitions(main_paths, scenario):
    baseline = table_view(query_file(main_paths[SINGLE_LAYOUTS[0]], scenario["sql"]))
    for layout in SINGLE_LAYOUTS[1:]:
        view = table_view(query_file(main_paths[layout], scenario["sql"]))
        reconcile_views(
            baseline,
            view,
            ordered=scenario["ordered"],
            what=f"{scenario['name']} / {SINGLE_LAYOUTS[0]} vs {layout}",
        )


@pytest.mark.parametrize("scenario", JOIN_SCENARIOS, ids=[s["name"] for s in JOIN_SCENARIOS])
def test_same_join_reconciles_across_partitions_and_strategies(join_paths, scenario):
    baseline = None
    baseline_where = None
    for layout in JOIN_LAYOUTS:
        files = join_paths[layout]
        sources = {"l": files["l"], "r": files[scenario["right"]]}
        for strategy in JOIN_STRATEGIES:
            view = table_view(query_files(sources, scenario["sql"], strategy))
            where = f"{layout} / strategy={strategy}"
            if baseline is None:
                baseline, baseline_where = view, where
            else:
                reconcile_views(
                    baseline,
                    view,
                    ordered=scenario["ordered"],
                    what=f"{scenario['name']} / {baseline_where} vs {where}",
                )


# ---------------------------------------------------------------------------
# 3. Freshly created query context (new CLI interpreter process)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SINGLE_SCENARIOS, ids=[s["name"] for s in SINGLE_SCENARIOS])
@pytest.mark.parametrize("layout", CLI_LAYOUTS)
def test_same_query_in_fresh_process_matches_in_process(main_paths, scenario, layout):
    in_process = table_view(query_file(main_paths[layout], scenario["sql"]))
    fresh = _cli_query_view(main_paths[layout], scenario["sql"])
    # Same layout + same SQL: the public contract is byte-identical output,
    # so the fresh context must match exactly, ordered or not.
    assert_same_view(in_process, fresh, what=f"{scenario['name']} / {layout} / fresh process")


# ---------------------------------------------------------------------------
# 4. Empty input files: repeats and partitions reconcile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", EMPTY_SCENARIOS, ids=[s["name"] for s in EMPTY_SCENARIOS])
def test_empty_input_repeats_and_partitions_reconcile(empty_paths, scenario):
    baseline = None
    for layout, path in empty_paths.items():
        for repeat in (1, 2):
            view = table_view(query_file(path, scenario["sql"]))
            assert_scenario_pins(view, scenario, what=f"{scenario['name']} / {layout}")
            if baseline is None:
                baseline = view
            else:
                assert_same_view(
                    baseline, view, what=f"{scenario['name']} / {layout} / repeat {repeat}"
                )


# ---------------------------------------------------------------------------
# 5. No execution state leaks into later queries
# ---------------------------------------------------------------------------


def test_no_state_leak_after_fully_consuming_other_queries(main_paths, join_paths):
    path = main_paths["v2_g3"]
    baseline = table_view(query_file(path, BENCHMARK_SQL))
    for scenario in SINGLE_SCENARIOS:
        consumed = query_file(path, scenario["sql"])
        # fully consume every column of every result
        for name in consumed.column_names:
            consumed.column(name)
    files = join_paths["v2g2_v2g3"]
    query_files(
        {"l": files["l"], "r": files["r"]},
        "SELECT l.id, r.tag FROM l INNER JOIN r ON l.k = r.k ORDER BY l.id, r.id",
    )
    again = table_view(query_file(path, BENCHMARK_SQL))
    assert_same_view(baseline, again, what="benchmark after fully consuming other queries")


def test_no_state_leak_after_partial_consumption_and_abandoned_result(main_paths):
    path = main_paths["v2_g3"]
    baseline = table_view(query_file(path, BENCHMARK_SQL))
    # Consume only a prefix of one column, mutate the consumer-side copies,
    # then abandon the result without touching the rest.
    partial = query_file(path, "SELECT * FROM input")
    ids = partial.column("id")
    ids[:3]
    ids.append(999)
    ids[0] = -1
    columns = partial.columns
    columns["grp"].append("zzz")
    columns["n"][0] = 12345
    del partial, ids, columns
    # A partially projected read of the same file is also abandoned midway.
    read_file(path, columns=["n", "id"])
    again = table_view(query_file(path, BENCHMARK_SQL))
    assert_same_view(baseline, again, what="benchmark after partial consumption")


def test_no_state_leak_after_legal_empty_result(main_paths):
    path = main_paths["v2_g3"]
    baseline = table_view(query_file(path, BENCHMARK_SQL))
    empty = query_file(path, "SELECT id, n FROM input WHERE id > 1000000 ORDER BY id")
    assert empty.row_count == 0
    assert empty.column_names == ("id", "n")
    again = table_view(query_file(path, BENCHMARK_SQL))
    assert_same_view(baseline, again, what="benchmark after an empty-result query")


def test_no_state_leak_when_interleaving_queries_and_errors(main_paths, join_paths):
    path = main_paths["v2_g3"]
    files = join_paths["v2g2_v2g3"]
    baseline = table_view(query_file(path, BENCHMARK_SQL))
    for round_no in (1, 2):
        query_file(path, "SELECT grp, COUNT(*) FROM input GROUP BY grp")
        query_file(path, "SELECT DISTINCT s FROM input")
        query_files(
            {"l": files["l"], "r": files["r"]},
            "SELECT l.id, r.tag FROM l LEFT JOIN r ON l.k = r.k ORDER BY l.id, r.id",
            "sort_merge",
        )
        with pytest.raises(QueryValidationError):
            query_file(path, "SELECT nope FROM input")
        with pytest.raises(QuerySyntaxError):
            query_file(path, "SELECT FROM input")
        view = table_view(query_file(path, BENCHMARK_SQL))
        assert_same_view(baseline, view, what=f"benchmark interleave round {round_no}")


def test_cli_partial_output_then_abort_leaves_file_reusable(main_paths):
    path = str(main_paths["v1"])
    cmd = [sys.executable, "-m", "columnar_analytics.cli", "query", path, "SELECT * FROM input"]
    full = subprocess.run(cmd, capture_output=True, env=_cli_env(), timeout=120)
    assert full.returncode == 0

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=_cli_env(),
    )
    try:
        proc.stdout.read(64)  # consume only a prefix, then abandon the stream
        proc.stdout.close()
    finally:
        try:
            proc.terminate()
        except OSError:
            pass
        proc.wait(timeout=60)

    again = subprocess.run(cmd, capture_output=True, env=_cli_env(), timeout=120)
    assert again.returncode == 0
    assert again.stdout == full.stdout


# ---------------------------------------------------------------------------
# 6. Queries never modify the data files they read
# ---------------------------------------------------------------------------


def test_queries_do_not_modify_data_files(main_paths, empty_paths, join_paths):
    all_paths = list(main_paths.values()) + list(empty_paths.values())
    for files in join_paths.values():
        all_paths.extend(files.values())
    before = {path: path.read_bytes() for path in all_paths}

    for layout, path in main_paths.items():
        for scenario in SINGLE_SCENARIOS:
            query_file(path, scenario["sql"])
    for path in empty_paths.values():
        for scenario in EMPTY_SCENARIOS:
            query_file(path, scenario["sql"])
    for layout in JOIN_LAYOUTS:
        files = join_paths[layout]
        for scenario in JOIN_SCENARIOS:
            sources = {"l": files["l"], "r": files[scenario["right"]]}
            for strategy in JOIN_STRATEGIES:
                query_files(sources, scenario["sql"], strategy)

    for path, blob in before.items():
        assert path.read_bytes() == blob, f"query execution modified {path.name}"


# ---------------------------------------------------------------------------
# 7. The comparison helpers themselves must be able to fail
# ---------------------------------------------------------------------------


def test_comparison_helpers_detect_order_type_and_duplicate_drift():
    schema = [("a", "int64", False)]
    base = (schema, [(1,), (2,), (2,)])

    # Reordered rows fail the ordered comparison ...
    with pytest.raises(AssertionError):
        assert_same_view(base, (schema, [(2,), (1,), (2,)]), what="reordered")
    # ... but pass the multiset comparison.
    assert_same_multiset(base, (schema, [(2,), (2,), (1,)]), what="reordered-ok")

    # A swallowed duplicate fails the multiset comparison.
    with pytest.raises(AssertionError):
        assert_same_multiset(base, (schema, [(1,), (2,)]), what="dropped-duplicate")

    # bool vs int64 value drift is detected even though True == 1.
    bool_view = ([("a", "bool", False)], [(True,)])
    with pytest.raises(AssertionError):
        assert_same_view(bool_view, ([("a", "bool", False)], [(1,)]), what="bool-vs-int")

    # Float zero-sign drift is detected even though 0.0 == -0.0.
    float_view = ([("a", "float64", False)], [(0.0,)])
    with pytest.raises(AssertionError):
        assert_same_view(
            float_view, ([("a", "float64", False)], [(-0.0,)]), what="zero-sign"
        )

    # Column name / logical type drift is detected before any row comparison.
    with pytest.raises(AssertionError):
        assert_same_view(base, ([("a", "int64", True)], [(1,), (2,), (2,)]), what="nullable")
    with pytest.raises(AssertionError):
        assert_same_view(base, ([("b", "int64", False)], [(1,), (2,), (2,)]), what="renamed")
