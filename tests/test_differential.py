"""Deterministic differential tests across storage layouts and execution paths.

The same logical datasets (generated once from fixed seeds, never produced by
the engine under test) are written through the public write entry points as:

* v1 files and v2 (row-group-partitioned) files with several legal
  ``row_group_size`` values,
* each combined with ``none`` / ``zlib`` compression and with / without
  utf8 dictionary encoding.

Every legal query scenario is run through the public ``query_file`` /
``query_files`` entries against every layout variant.  The engine results are
never compared against one of the engine's own paths: an independent, direct
row-level reference implementation (a naive nested-loop join -- a third
algorithm besides the engine's hash and sort-merge paths -- plus explicit
three-valued-logic, projection, CASE, DISTINCT, grouping, aggregate and
stable-sort evaluation; see :class:`Col` and friends below) supplies the
expected schema and rows.  Each scenario runs three times per variant so the
deterministic row order with no ORDER BY is checked as well.

The data deliberately covers all four column types, nullable columns,
duplicate join keys, an all-NULL column, an empty table, Unicode strings,
float64 positive/negative zero and values close to (but not over) the int64
boundaries.  A failing case is reproducible from the fixed seeds and the
query text alone: the seeds are module constants and no randomness, clock
reading or private engine helper is used anywhere in this module.
"""

from __future__ import annotations

import csv as csv_module
import json
import math
from functools import cmp_to_key

import pytest

from columnar_analytics import (
    ColumnSchema,
    Schema,
    Table,
    export_query_file,
    export_query_files,
    explain_file,
    explain_files,
    inspect_file,
    query_file,
    query_files,
    write_file,
    write_partitioned_file,
)

# ---------------------------------------------------------------------------
# Fixed-seed data generation
# ---------------------------------------------------------------------------
#
# A small self-contained PCG-style 64-bit generator.  Only the seeds and the
# generation rules below are needed to rebuild every test value; the engine
# never participates in producing the expected results.

SEED_MAIN = 20261003
SEED_U = 424242
SEED_W = 909091

_MASK64 = (1 << 64) - 1
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

# Repeated, empty and Unicode strings (including characters outside the BMP)
# give utf8 dictionaries, min/max stats and DISTINCT real work to do.
STRINGS = ["a", "b", "", "ab", "名前", "αβγ", "café", "🍎", "Δ", "a", "b"]


class _Rng:
    def __init__(self, seed: int):
        state = seed & _MASK64
        assert state, "the generator state must be non-zero"
        self._state = state

    def u64(self) -> int:
        self._state = (
            self._state * 6364136223846793005 + 1442695040888963407
        ) & _MASK64
        x = self._state
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & _MASK64
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & _MASK64
        x ^= x >> 31
        return x

    def below(self, n: int) -> int:
        return self.u64() % n

    def pick(self, seq):
        return seq[self.below(len(seq))]

    def chance(self, numerator: int, denominator: int = 100) -> bool:
        return self.u64() % denominator < numerator


T_SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("big", "int64", nullable=True),
        ColumnSchema("f", "float64", nullable=True),
        ColumnSchema("s", "utf8", nullable=True),
        ColumnSchema("flag", "bool"),
        ColumnSchema("zflag", "bool", nullable=True),
        ColumnSchema("znull", "utf8", nullable=True),
    ]
)

U_SCHEMA = Schema(
    [
        ColumnSchema("uid", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("tag", "utf8", nullable=True),
        ColumnSchema("uf", "float64", nullable=True),
        ColumnSchema("uflag", "bool"),
    ]
)

W_SCHEMA = Schema(
    [
        ColumnSchema("wid", "int64"),
        ColumnSchema("wk", "float64", nullable=True),
        ColumnSchema("wtag", "utf8", nullable=True),
    ]
)


def build_t(seed: int = SEED_MAIN, rows: int = 64) -> dict:
    """The main single-table dataset.

    ``big`` is zero almost everywhere but carries int64 boundary-near values
    at fixed positions (and one NULL); ``f`` carries both zero signs;
    ``znull`` is NULL in every row; ``s`` mixes empty / duplicate / Unicode
    strings with NULLs.
    """
    rng = _Rng(seed)
    n_values: list = []
    f_values: list = []
    s_values: list = []
    for i in range(rows):
        n_values.append(None if rng.chance(18) else rng.below(21) - 8)
        if rng.chance(20):
            f_values.append(None)
        else:
            f_values.append(round(rng.below(2000) / 100.0 - 10.0, 2))
        s_values.append(None if rng.chance(20) else rng.pick(STRINGS))
    # Deterministic positive/negative zero coverage.
    f_values[2] = 0.0
    f_values[3] = -0.0
    f_values[10] = 1.0
    f_values[30] = -0.0
    big_values = [0 if value is None else value for value in n_values]
    big_values[0] = _INT64_MIN + 1
    big_values[7] = 2**62
    big_values[20] = None
    big_values[57] = -(2**62)
    big_values[rows - 1] = _INT64_MAX
    zflag = [
        None if rng.chance(40) else (rng.below(2) == 0) for _ in range(rows)
    ]
    return {
        "id": list(range(rows)),
        "n": n_values,
        "big": big_values,
        "f": f_values,
        "s": s_values,
        "flag": [i % 2 == 0 for i in range(rows)],
        "zflag": zflag,
        "znull": [None for _ in range(rows)],
    }


def build_u(seed: int = SEED_U, rows: int = 40) -> dict:
    """A join partner with duplicate / NULL keys and several unmatched keys."""
    rng = _Rng(seed)
    keys: list = []
    tags: list = []
    uf: list = []
    for i in range(rows):
        if rng.chance(15):
            keys.append(None)
        elif i >= rows - 4:
            keys.append(90 + (i - (rows - 4)) * 5)  # 90, 95, 100, 105: unmatched
        else:
            keys.append(rng.below(23) - 11)  # -11 .. 11, duplicated heavily
        tags.append(None if rng.chance(25) else rng.pick(STRINGS))
        if rng.chance(20):
            uf.append(None)
        else:
            uf.append(round(rng.below(1000) / 50.0 - 10.0, 2))
    uf[4] = -0.0
    uf[14] = 0.0
    return {
        "uid": [1000 + i for i in range(rows)],
        "k": keys,
        "tag": tags,
        "uf": uf,
        "uflag": [i % 3 == 0 for i in range(rows)],
    }


def build_w(seed: int = SEED_W, rows: int = 24) -> dict:
    """A third table joining ``u`` on a float64 key (mixed int64/float64)."""
    rng = _Rng(seed)
    keys: list = []
    tags: list = []
    for i in range(rows):
        if rng.chance(17):
            keys.append(None)
        elif i >= rows - 3:
            keys.append(98.0 + i)  # unmatched
        else:
            keys.append(float(rng.below(23) - 11))
        tags.append(None if rng.chance(30) else rng.pick(STRINGS))
    keys[2] = 0.0
    keys[11] = -0.0
    return {
        "wid": [500 + i for i in range(rows)],
        "wk": keys,
        "wtag": tags,
    }


# ---------------------------------------------------------------------------
# Independent row-level reference implementation
# ---------------------------------------------------------------------------
#
# Nothing below reads engine state.  Rows are plain ``dict`` objects mapping
# qualified column names to Python values; a *schema map* gives each visible
# column's (type, nullable).  Joins use a direct nested-loop expansion (a
# third algorithm besides the engine's hash and sort-merge paths), and every
# SQL semantic below is implemented explicitly rather than taken from an
# engine result.


class RefFailure(AssertionError):
    """Raised when a reference scenario itself violates the SQL rules."""


def _distinct_key(type_name: str, value):
    if value is None:
        return (False,)
    if type_name == "float64":
        # SQL DISTINCT treats +0.0 and -0.0 as the same value.
        return (True, 0.0 if value == 0 else value)
    return (True, value)


class Val:
    """A scalar expression: typed, with static nullability."""

    type: str
    nullable: bool

    def eval(self, row: dict):
        raise NotImplementedError


class Lit(Val):
    def __init__(self, value, type_name: str):
        self.value = value
        self.type = type_name
        self.nullable = False

    def eval(self, row):
        return self.value


class Col(Val):
    def __init__(self, name: str, schema_map: dict):
        self.name = name
        self.type = schema_map[name][0]
        self.nullable = schema_map[name][1]

    def eval(self, row):
        return row[self.name]


class Neg(Val):
    def __init__(self, inner: Val):
        if inner.type not in ("int64", "float64"):
            raise RefFailure("unary minus needs a numeric operand")
        self.inner = inner
        self.type = inner.type
        self.nullable = inner.nullable

    def eval(self, row):
        value = self.inner.eval(row)
        if value is None:
            return None
        if isinstance(value, int) and not isinstance(value, bool):
            result = -value
            if not (_INT64_MIN <= result <= _INT64_MAX):
                raise RefFailure("int64 unary minus overflowed")
            return result
        return -value


class Bin(Val):
    def __init__(self, op: str, left: Val, right: Val):
        if op not in ("+", "-", "*", "/"):
            raise RefFailure(f"unsupported operator {op!r}")
        if left.type not in ("int64", "float64") or right.type not in (
            "int64",
            "float64",
        ):
            raise RefFailure("arithmetic needs numeric operands")
        self.op = op
        self.left = left
        self.right = right
        self.type = (
            "float64"
            if op == "/" or "float64" in (left.type, right.type)
            else "int64"
        )
        self.nullable = left.nullable or right.nullable

    def eval(self, row):
        a = self.left.eval(row)
        b = self.right.eval(row)
        if a is None or b is None:
            return None
        if self.op == "/" or isinstance(a, float) or isinstance(b, float):
            if self.op == "/" and b == 0:
                raise RefFailure("division by zero")
            fa, fb = float(a), float(b)
            result = {
                "+": fa + fb,
                "-": fa - fb,
                "*": fa * fb,
                "/": fa / fb,
            }[self.op]
            if not math.isfinite(result):
                raise RefFailure("non-finite float64 result")
            return result
        result = {"+": a + b, "-": a - b, "*": a * b}[self.op]
        if not (_INT64_MIN <= result <= _INT64_MAX):
            raise RefFailure("int64 arithmetic overflowed")
        return result


class Case(Val):
    """Searched CASE with explicit branch routing and lazy result eval."""

    def __init__(self, branches, else_value):
        self.branches = [(pred, value) for pred, value in branches]
        self.else_value = else_value
        types_ = [value.type for _pred, value in branches]
        if else_value is not None:
            types_.append(else_value.type)
        if set(types_) <= {"int64", "float64"}:
            self.type = "float64" if "float64" in types_ else "int64"
        elif len(set(types_)) == 1:
            self.type = types_[0]
        else:
            raise RefFailure(f"inconsistent CASE result types: {types_}")
        if else_value is None:
            self.nullable = True
        else:
            self.nullable = any(
                value.nullable for _pred, value in branches
            ) or else_value.nullable

    def eval(self, row):
        chosen = None
        for pred, value in self.branches:
            if pred.eval(row) is True:
                chosen = value.eval(row)  # only the hit branch is evaluated
                break
        else:
            chosen = None if self.else_value is None else self.else_value.eval(row)
        if chosen is None:
            return None
        if self.type == "float64" and isinstance(chosen, int) and not isinstance(
            chosen, bool
        ):
            return float(chosen)
        return chosen


# Predicates: eval returns True / False / None (UNKNOWN).


class Pred:
    def eval(self, row) -> bool | None:
        raise NotImplementedError


class Cmp(Pred):
    def __init__(self, op: str, left: Val, right: Val):
        self.op = op
        self.left = left
        self.right = right

    def eval(self, row):
        a = self.left.eval(row)
        b = self.right.eval(row)
        if a is None or b is None:
            return None
        if self.op == "=":
            result = a == b
        elif self.op == "!=":
            result = a != b
        elif self.op == "<":
            result = a < b
        elif self.op == "<=":
            result = a <= b
        elif self.op == ">":
            result = a > b
        else:
            result = a >= b
        return bool(result)


class IsNull(Pred):
    def __init__(self, inner: Val, negate: bool = False):
        self.inner = inner
        self.negate = negate

    def eval(self, row):
        result = self.inner.eval(row) is None
        return (not result) if self.negate else result


class Not(Pred):
    def __init__(self, inner: Pred):
        self.inner = inner

    def eval(self, row):
        value = self.inner.eval(row)
        return None if value is None else (not value)


class And(Pred):
    def __init__(self, left: Pred, right: Pred):
        self.left = left
        self.right = right

    def eval(self, row):
        a = self.left.eval(row)
        if a is False:
            return False
        b = self.right.eval(row)
        if b is False:
            return False
        if a is None or b is None:
            return None
        return True


class Or(Pred):
    def __init__(self, left: Pred, right: Pred):
        self.left = left
        self.right = right

    def eval(self, row):
        a = self.left.eval(row)
        if a is True:
            return True
        b = self.right.eval(row)
        if b is True:
            return True
        if a is None or b is None:
            return None
        return False


# Aggregates ---------------------------------------------------------------


def _aggregate(func: str, distinct: bool, type_name: str | None, values: list):
    if func == "COUNT*":
        return len(values)
    nonnull = [value for value in values if value is not None]
    if distinct:
        seen: set = set()
        deduped = []
        for value in nonnull:
            key = _distinct_key(type_name, value)
            if key not in seen:
                seen.add(key)
                deduped.append(value)
        nonnull = deduped
    if func == "COUNT":
        return len(nonnull)
    if not nonnull:
        return None
    if func == "MIN":
        return min(nonnull)
    if func == "MAX":
        return max(nonnull)
    if func == "SUM":
        if type_name == "int64":
            total = sum(nonnull)
            if not (_INT64_MIN <= total <= _INT64_MAX):
                raise RefFailure("SUM overflowed int64")
            return total
        total = math.fsum(nonnull)
        if not math.isfinite(total):
            raise RefFailure("SUM produced a non-finite float64")
        return total
    if func == "AVG":
        result = math.fsum(nonnull) / len(nonnull)
        if not math.isfinite(result):
            raise RefFailure("AVG produced a non-finite float64")
        return result
    raise RefFailure(f"unknown aggregate {func!r}")


def agg_type_nullable(func: str, arg_type: str | None) -> tuple[str, bool]:
    if func in ("COUNT*", "COUNT"):
        return "int64", False
    if func == "SUM":
        return arg_type, True
    if func == "AVG":
        return "float64", True
    return arg_type, True  # MIN / MAX


# Row pipeline --------------------------------------------------------------


def scan(data: dict) -> list[dict]:
    names = list(data)
    return [dict(zip(names, row)) for row in zip(*(data[name] for name in names))]


def qualify(rows: list[dict], prefix: str) -> list[dict]:
    return [
        {f"{prefix}.{name}": value for name, value in row.items()} for row in rows
    ]


def schema_for(schema: Schema, prefix: str | None = None) -> dict:
    return {
        (col.name if prefix is None else f"{prefix}.{col.name}"): (
            col.type,
            col.nullable,
        )
        for col in schema.columns
    }


def filter_rows(rows: list[dict], predicate: Pred | None) -> list[dict]:
    if predicate is None:
        return list(rows)
    return [row for row in rows if predicate.eval(row) is True]


def _compare_keys(a, b, keys) -> int:
    for value_fn, descending, nulls_first in keys:
        va, vb = value_fn(a), value_fn(b)
        if va is None or vb is None:
            if va is None and vb is None:
                continue
            none_before = -1 if nulls_first else 1
            return none_before if va is None else -none_before
        c = (va > vb) - (va < vb)
        c = -c if descending else c
        if c:
            return c
    return 0


def stable_sort(rows: list[dict], keys) -> list[dict]:
    # Python's sort is stable, so equal-key rows keep their input order.
    return sorted(rows, key=cmp_to_key(lambda a, b: _compare_keys(a, b, keys)))


def expr_order(expr: Val, descending=False, nulls_first=False):
    return (lambda row: expr.eval(row), descending, nulls_first)


def name_order(name: str, descending=False, nulls_first=False):
    return (lambda row: row[name], descending, nulls_first)


def project(rows: list[dict], outputs) -> tuple[list, list[tuple]]:
    """``outputs`` is a list of (name, Val); returns (schema rows, value rows)."""
    out_schema = [(name, expr.type, expr.nullable) for name, expr in outputs]
    out_rows = [
        tuple(expr.eval(row) for _name, expr in outputs) for row in rows
    ]
    return out_schema, out_rows


def distinct_rows(rows: list[tuple], type_names: list[str]) -> list[tuple]:
    seen: set = set()
    unique: list[tuple] = []
    for row in rows:
        key = tuple(
            _distinct_key(type_names[i], row[i]) for i in range(len(row))
        )
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def distinct_pipeline(
    rows,
    outputs,
    type_names,
    order_names,
    limit,
):
    """WHERE -> project -> whole-row DISTINCT -> stable sort -> LIMIT."""
    out_schema, projected = project(rows, outputs)
    names = [name for name, _expr in outputs]
    deduped = distinct_rows(projected, type_names)
    if order_names is not None:
        wrapped = [dict(zip(names, row)) for row in deduped]
        wrapped = stable_sort(wrapped, order_names)
        deduped = [tuple(row[name] for name in names) for row in wrapped]
    if limit is not None:
        deduped = deduped[:limit]
    return out_schema, deduped


def group_rows(rows: list[dict], key_names: tuple[str, ...]):
    """Groups in first-selected-row order; NULL keys form their own group."""
    buckets: dict = {}
    order: list = []
    for row in rows:
        key = tuple(row[name] for name in key_names)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = []
            buckets[key] = bucket
            order.append(key)
        bucket.append(row)
    return [tuple(buckets[key]) for key in order]


# Joins --------------------------------------------------------------------


def join_step(
    left_rows: list[dict],
    right_rows: list[dict],
    left_schema: dict,
    right_schema: dict,
    lkey: str,
    rkey: str,
    kind: str,
):
    """Naive nested-loop equi-join with the documented output row order.

    This deliberately finds matches a third way (neither hash nor
    sort-merge).  NULL keys never match, duplicate keys expand to the full
    combination, INNER/LEFT/FULL are driven by left-file order and RIGHT by
    right-file order.  Returns (rows, schema_map) with nullability derived
    for this one step.
    """
    matches_left: dict = {}
    matches_right: dict = {}
    for i, left in enumerate(left_rows):
        for j, right in enumerate(right_rows):
            lv, rv = left[lkey], right[rkey]
            if lv is not None and rv is not None and lv == rv:
                matches_left.setdefault(i, []).append(j)
                matches_right.setdefault(j, []).append(i)

    out: list[dict] = []

    def emit_match(i, j):
        row = dict(left_rows[i])
        row.update(right_rows[j])
        out.append(row)

    def emit_unmatched_left(i):
        row = dict(left_rows[i])
        for name in right_rows[0]:
            row[name] = None
        out.append(row)

    def emit_unmatched_right(j):
        row = {name: None for name in left_rows[0]}
        row.update(right_rows[j])
        out.append(row)

    if kind == "right":
        for j in range(len(right_rows)):
            left_hits = matches_right.get(j)
            if left_hits:
                for i in left_hits:  # kept in left-file order
                    emit_match(i, j)
            else:
                emit_unmatched_right(j)
    else:
        for i in range(len(left_rows)):
            right_hits = matches_left.get(i)
            if right_hits:
                for j in right_hits:  # right-file order within one left row
                    emit_match(i, j)
            elif kind in ("left", "full"):
                emit_unmatched_left(i)
        if kind == "full":
            for j in range(len(right_rows)):
                if j not in matches_right:
                    emit_unmatched_right(j)

    pad_left = kind in ("right", "full")
    pad_right = kind in ("left", "full")
    out_schema = {
        name: (type_name, True if pad_left else nullable)
        for name, (type_name, nullable) in left_schema.items()
    }
    out_schema.update(
        {
            name: (type_name, True if pad_right else nullable)
            for name, (type_name, nullable) in right_schema.items()
        }
    )
    return out, out_schema


def apply_steps(t_rows, u_rows, w_rows, t_map, u_map, w_map, steps):
    """Run (kind, right_table) steps; each step joins on ``*.n/k`` style keys."""
    right_inputs = {
        "u": (u_rows, u_map),
        "w": (w_rows, w_map),
    }
    rows, combined = t_rows, dict(t_map)
    for kind, right_name, lkey, rkey in steps:
        right_rows, right_map = right_inputs[right_name]
        rows, combined = join_step(
            rows, right_rows, combined, right_map, lkey, rkey, kind
        )
    return rows, combined


def schema_after_steps(t_map, u_map, w_map, steps):
    """The post-chain schema map (nullability derived per step)."""
    right_maps = {"u": u_map, "w": w_map}
    derived = dict(t_map)
    for kind, right_name, _lkey, _rkey in steps:
        pad_left = kind in ("right", "full")
        pad_right = kind in ("left", "full")
        derived = {
            name: (type_name, True if pad_left else nullable)
            for name, (type_name, nullable) in derived.items()
        }
        derived.update(
            {
                name: (type_name, True if pad_right else nullable)
                for name, (type_name, nullable) in right_maps[right_name].items()
            }
        )
    return derived


# Aggregate scenario helper -------------------------------------------------


class _GAgg(Val):
    """An aggregate result inside a group-context row (HAVING / ORDER BY)."""

    def __init__(self, label: str, type_name: str, nullable: bool):
        self.label = label
        self.type = type_name
        self.nullable = nullable

    def eval(self, row):
        return row[self.label]


def run_aggregate(
    rows: list[dict],
    schema_map: dict,
    key_names: tuple[str, ...],
    agg_specs: list,
    select_names: list[str],
    having: Pred | None,
    order_keys,
    limit: int | None,
):
    """Reference WHERE -> GROUP BY/aggregates -> HAVING -> sort -> LIMIT pipeline.

    ``agg_specs`` lists every aggregate needed by SELECT / HAVING / ORDER BY
    as ``(label, func, arg_name_or_None, distinct)``; ``select_names`` picks
    key columns and aggregate labels for the output and its column order.
    """
    groups = group_rows(rows, key_names) if key_names else [tuple(rows)]
    materialised: list[dict] = []
    for members in groups:
        context: dict = {}
        for name in key_names:
            context[name] = members[0][name]
        for label, func, arg_name, distinct in agg_specs:
            if func == "COUNT*":
                values = list(members)
                arg_type = None
            else:
                arg_type = schema_map[arg_name][0]
                values = [member[arg_name] for member in members]
            context[label] = _aggregate(func, distinct, arg_type, values)
        if having is not None and having.eval(context) is not True:
            continue
        materialised.append(context)

    if order_keys is not None:
        materialised = stable_sort(materialised, order_keys)
    if limit is not None:
        materialised = materialised[:limit]

    out_schema = []
    for name in select_names:
        if name in key_names:
            type_name, nullable = schema_map[name]
            out_schema.append((name, type_name, nullable))
        else:
            _label, func, arg_name, _distinct = next(
                spec for spec in agg_specs if spec[0] == name
            )
            arg_type = None if func == "COUNT*" else schema_map[arg_name][0]
            type_name, nullable = agg_type_nullable(func, arg_type)
            out_schema.append((name, type_name, nullable))
    out_rows = [
        tuple(context[name] for name in select_names) for context in materialised
    ]
    return out_schema, out_rows


# ---------------------------------------------------------------------------
# Scenario definitions: the SQL text and its independent reference side by
# side.  Every SQL string stays inside the documented public grammar.
# ---------------------------------------------------------------------------


DATASETS = {"t": build_t(), "u": build_u(), "w": build_w()}


def build_single_scenarios(datasets) -> list:
    t_rows = scan(datasets["t"])
    t_map = schema_for(T_SCHEMA)
    scenarios = []

    def add(name, sql, expected_fn, *, ordered=True):
        scenarios.append(
            {"name": name, "sql": sql, "expected": expected_fn, "ordered": ordered}
        )

    def c(col):
        return Col(col, t_map)

    def li(value):
        return Lit(value, "int64")

    def lf(value):
        return Lit(value, "float64")

    def lb(value):
        return Lit(value, "bool")

    def ls(value):
        return Lit(value, "utf8")

    # -- plain projection / filtering -------------------------------------

    add(
        "p1_star_unordered",
        "SELECT * FROM input",
        lambda: (
            [(name, *t_map[name]) for name in T_SCHEMA.names],
            [tuple(row[name] for name in T_SCHEMA.names) for row in t_rows],
        ),
        ordered=False,
    )

    add(
        "p2_pushdown_only",
        "SELECT id, n, f, s FROM input "
        "WHERE n >= 2 AND f < 5.0 AND id <= 50 ORDER BY id",
        lambda: project(
            stable_sort(
                filter_rows(
                    t_rows,
                    And(
                        And(Cmp(">=", c("n"), li(2)), Cmp("<", c("f"), lf(5.0))),
                        Cmp("<=", c("id"), li(50)),
                    ),
                ),
                [expr_order(c("id"))],
            ),
            [("id", c("id")), ("n", c("n")), ("f", c("f")), ("s", c("s"))],
        ),
    )

    p3_case = Case(
        [
            (IsNull(c("n")), li(-1)),
            (Cmp(">", c("n"), li(5)), Bin("*", c("n"), li(2))),
        ],
        Bin("-", c("n"), li(1)),
    )
    p3_pred = And(
        And(
            Or(Cmp(">", c("n"), li(0)), Cmp("=", c("zflag"), lb(True))),
            IsNull(c("s"), negate=True),
        ),
        And(
            Cmp(">=", Case([(IsNull(c("n")), li(0))], c("n")), li(0)),
            Cmp("<", c("id"), li(60)),
        ),
    )
    add(
        "p3_mixed_pushdown_case_projection",
        "SELECT id, n, f, s, n * 2 AS dbl, "
        "CASE WHEN n IS NULL THEN -1 WHEN n > 5 THEN n * 2 ELSE n - 1 END AS c, "
        "(f + 1.5) * 2.0 AS fs FROM input "
        "WHERE (n > 0 OR zflag = TRUE) AND s IS NOT NULL "
        "AND (CASE WHEN n IS NULL THEN 0 ELSE n END >= 0) AND id < 60 "
        "ORDER BY c DESC NULLS LAST, id ASC LIMIT 25",
        lambda: project(
            stable_sort(
                filter_rows(t_rows, p3_pred),
                [
                    expr_order(p3_case, descending=True),
                    expr_order(c("id")),
                ],
            )[:25],
            [
                ("id", c("id")),
                ("n", c("n")),
                ("f", c("f")),
                ("s", c("s")),
                ("dbl", Bin("*", c("n"), li(2))),
                ("c", p3_case),
                ("fs", Bin("*", Bin("+", c("f"), lf(1.5)), lf(2.0))),
            ],
        ),
    )

    add(
        "p4_not_or_not_null_inequality",
        "SELECT id, n, s FROM input "
        "WHERE NOT (n = 0 OR n IS NULL) AND s != 'a' ORDER BY id LIMIT 30",
        lambda: project(
            stable_sort(
                filter_rows(
                    t_rows,
                    And(
                        Not(Or(Cmp("=", c("n"), li(0)), IsNull(c("n")))),
                        Cmp("!=", c("s"), ls("a")),
                    ),
                ),
                [expr_order(c("id"))],
            )[:30],
            [("id", c("id")), ("n", c("n")), ("s", c("s"))],
        ),
    )

    add(
        "p5_zero_selected_rows",
        "SELECT id, n FROM input WHERE id > 1000 ORDER BY id LIMIT 5",
        lambda: project(
            stable_sort(
                filter_rows(t_rows, Cmp(">", c("id"), li(1000))),
                [expr_order(c("id"))],
            )[:5],
            [("id", c("id")), ("n", c("n"))],
        ),
    )

    add(
        "p6_limit_zero",
        "SELECT * FROM input ORDER BY id LIMIT 0",
        lambda: (
            [(name, *t_map[name]) for name in T_SCHEMA.names],
            [],
        ),
    )

    add(
        "p7_multi_key_nulls_directions",
        "SELECT id, n, s FROM input "
        "ORDER BY n DESC NULLS FIRST, s ASC NULLS LAST, id LIMIT 40",
        lambda: project(
            stable_sort(
                t_rows,
                [
                    expr_order(c("n"), descending=True, nulls_first=True),
                    expr_order(c("s")),
                    expr_order(c("id")),
                ],
            )[:40],
            [("id", c("id")), ("n", c("n")), ("s", c("s"))],
        ),
    )

    p8_bool_case = Case(
        [
            (Cmp("=", c("flag"), lb(True)), lb(True)),
            (Cmp(">", c("n"), li(0)), lb(False)),
        ],
        c("flag"),
    )
    p8_text_case = Case(
        [
            (IsNull(c("n")), c("s")),
            (Cmp(">=", c("n"), li(0)), ls("正")),
        ],
        c("s"),
    )
    add(
        "p8_arith_unary_neg_division_cases",
        "SELECT id, -big AS neg, id / 2 AS half, "
        "CASE WHEN flag THEN TRUE WHEN n > 0 THEN FALSE ELSE flag END AS bc, "
        "CASE WHEN n IS NULL THEN s WHEN n >= 0 THEN '正' ELSE s END AS cs "
        "FROM input WHERE id < 32 ORDER BY id",
        lambda: project(
            stable_sort(
                filter_rows(t_rows, Cmp("<", c("id"), li(32))),
                [expr_order(c("id"))],
            ),
            [
                ("id", c("id")),
                ("neg", Neg(c("big"))),
                ("half", Bin("/", c("id"), li(2))),
                ("bc", p8_bool_case),
                ("cs", p8_text_case),
            ],
        ),
    )

    add(
        "p9_non_pushdown_arithmetic_predicate",
        "SELECT id, n, f FROM input "
        "WHERE f * 2.0 > 1.0 AND n IS NOT NULL AND zflag IS NOT NULL "
        "ORDER BY f DESC, id LIMIT 30",
        lambda: project(
            stable_sort(
                filter_rows(
                    t_rows,
                    And(
                        And(
                            Cmp(">", Bin("*", c("f"), lf(2.0)), lf(1.0)),
                            IsNull(c("n"), negate=True),
                        ),
                        IsNull(c("zflag"), negate=True),
                    ),
                ),
                [expr_order(c("f"), descending=True), expr_order(c("id"))],
            )[:30],
            [("id", c("id")), ("n", c("n")), ("f", c("f"))],
        ),
    )

    p10_case = Case(
        [
            (Cmp(">", c("n"), li(5)), ls("big")),
            (Cmp("<", c("n"), li(-5)), ls("small")),
        ],
        None,
    )
    add(
        "p10_case_without_else",
        "SELECT id, n, CASE WHEN n > 5 THEN 'big' WHEN n < -5 THEN 'small' END "
        "AS bucket FROM input WHERE id < 40 ORDER BY id",
        lambda: project(
            stable_sort(
                filter_rows(t_rows, Cmp("<", c("id"), li(40))),
                [expr_order(c("id"))],
            ),
            [("id", c("id")), ("n", c("n")), ("bucket", p10_case)],
        ),
    )

    p11_bool_case = Case([(IsNull(c("s")), c("flag"))], c("zflag"))
    add(
        "p11_boolean_case_in_where",
        "SELECT id, flag, zflag FROM input "
        "WHERE (CASE WHEN s IS NULL THEN flag ELSE zflag END) = TRUE "
        "ORDER BY id LIMIT 30",
        lambda: project(
            stable_sort(
                filter_rows(t_rows, Cmp("=", p11_bool_case, lb(True))),
                [expr_order(c("id"))],
            )[:30],
            [("id", c("id")), ("flag", c("flag")), ("zflag", c("zflag"))],
        ),
    )

    # -- SELECT DISTINCT ---------------------------------------------------

    add(
        "d1_distinct_first_occurrence_order",
        "SELECT DISTINCT n, flag FROM input WHERE id >= 8",
        lambda: distinct_pipeline(
            filter_rows(t_rows, Cmp(">=", c("id"), li(8))),
            [("n", c("n")), ("flag", c("flag"))],
            ["int64", "bool"],
            None,
            None,
        ),
        ordered=False,
    )

    add(
        "d2_distinct_ordered_limit",
        "SELECT DISTINCT n, flag FROM input "
        "ORDER BY n NULLS FIRST, flag DESC LIMIT 20",
        lambda: distinct_pipeline(
            t_rows,
            [("n", c("n")), ("flag", c("flag"))],
            ["int64", "bool"],
            [
                name_order("n", nulls_first=True),
                name_order("flag", descending=True),
            ],
            20,
        ),
    )

    d3_case = Case(
        [
            (IsNull(c("f")), lf(0.0)),
            (Cmp("=", c("f"), lf(0.0)), lf(0.0)),
        ],
        lf(1.0),
    )
    add(
        "d3_distinct_signed_zero_coalesces",
        "SELECT DISTINCT CASE WHEN f IS NULL THEN 0.0 WHEN f = 0.0 THEN 0.0 "
        "ELSE 1.0 END AS bucket FROM input ORDER BY bucket",
        lambda: distinct_pipeline(
            t_rows,
            [("bucket", d3_case)],
            ["float64"],
            [name_order("bucket")],
            None,
        ),
    )

    add(
        "d4_distinct_empty_result",
        "SELECT DISTINCT s FROM input WHERE id > 1000 ORDER BY s",
        lambda: distinct_pipeline(
            filter_rows(t_rows, Cmp(">", c("id"), li(1000))),
            [("s", c("s"))],
            ["utf8"],
            [name_order("s")],
            None,
        ),
    )

    # -- aggregates --------------------------------------------------------

    agg1_specs = [
        ("COUNT(*)", "COUNT*", None, False),
        ("COUNT(n)", "COUNT", "n", False),
        ("COUNT(DISTINCT n)", "COUNT", "n", True),
        ("SUM(n)", "SUM", "n", False),
        ("SUM(DISTINCT n)", "SUM", "n", True),
        ("AVG(f)", "AVG", "f", False),
        ("MIN(s)", "MIN", "s", False),
        ("MAX(n)", "MAX", "n", False),
        ("MIN(flag)", "MIN", "flag", False),
        ("MAX(DISTINCT f)", "MAX", "f", True),
    ]
    agg1_names = [spec[0] for spec in agg1_specs]

    def agg1_expected(predicate):
        return run_aggregate(
            filter_rows(t_rows, predicate),
            t_map,
            (),
            agg1_specs,
            agg1_names,
            None,
            None,
            None,
        )

    add(
        "agg1_global_aggregates",
        "SELECT COUNT(*), COUNT(n), COUNT(DISTINCT n), SUM(n), SUM(DISTINCT n), "
        "AVG(f), MIN(s), MAX(n), MIN(flag), MAX(DISTINCT f) FROM input WHERE id >= 5",
        lambda: agg1_expected(Cmp(">=", c("id"), li(5))),
        ordered=False,
    )
    add(
        "agg1b_global_aggregates_zero_rows",
        "SELECT COUNT(*), COUNT(n), COUNT(DISTINCT n), SUM(n), SUM(DISTINCT n), "
        "AVG(f), MIN(s), MAX(n), MIN(flag), MAX(DISTINCT f) FROM input WHERE id > 1000",
        lambda: agg1_expected(Cmp(">", c("id"), li(1000))),
        ordered=False,
    )

    agg2_specs = [
        ("COUNT(*)", "COUNT*", None, False),
        ("COUNT(DISTINCT n)", "COUNT", "n", True),
        ("SUM(n)", "SUM", "n", False),
        ("AVG(f)", "AVG", "f", False),
        ("MIN(n)", "MIN", "n", False),
        ("MAX(id)", "MAX", "id", False),
    ]
    agg2_names = [
        "s",
        "COUNT(*)",
        "COUNT(DISTINCT n)",
        "SUM(n)",
        "AVG(f)",
        "MIN(n)",
        "MAX(id)",
    ]
    add(
        "agg2_group_by_utf8_having_order_limit",
        "SELECT s, COUNT(*), COUNT(DISTINCT n), SUM(n), AVG(f), MIN(n), MAX(id) "
        "FROM input GROUP BY s HAVING COUNT(*) >= 2 AND AVG(f) IS NOT NULL "
        "ORDER BY COUNT(*) DESC, s ASC NULLS FIRST LIMIT 15",
        lambda: run_aggregate(
            t_rows,
            t_map,
            ("s",),
            agg2_specs,
            agg2_names,
            And(
                Cmp(
                    ">=",
                    _GAgg("COUNT(*)", "int64", False),
                    li(2),
                ),
                IsNull(_GAgg("AVG(f)", "float64", True), negate=True),
            ),
            [
                name_order("COUNT(*)", descending=True),
                name_order("s", nulls_first=True),
            ],
            15,
        ),
    )

    agg3_specs = [
        ("COUNT(*)", "COUNT*", None, False),
        ("COUNT(DISTINCT s)", "COUNT", "s", True),
        ("SUM(DISTINCT n)", "SUM", "n", True),
    ]
    add(
        "agg3_multi_group_key_first_row_order",
        "SELECT flag, n, COUNT(*), COUNT(DISTINCT s), SUM(DISTINCT n) "
        "FROM input WHERE id >= 10 GROUP BY n, flag",
        lambda: run_aggregate(
            filter_rows(t_rows, Cmp(">=", c("id"), li(10))),
            t_map,
            ("n", "flag"),
            agg3_specs,
            ["flag", "n", "COUNT(*)", "COUNT(DISTINCT s)", "SUM(DISTINCT n)"],
            None,
            None,
            None,
        ),
        ordered=False,
    )

    add(
        "agg4_group_by_nullable_bool",
        "SELECT zflag, COUNT(*), COUNT(DISTINCT id) FROM input GROUP BY zflag "
        "ORDER BY zflag NULLS FIRST, COUNT(*) DESC",
        lambda: run_aggregate(
            t_rows,
            t_map,
            ("zflag",),
            [
                ("COUNT(*)", "COUNT*", None, False),
                ("COUNT(DISTINCT id)", "COUNT", "id", True),
            ],
            ["zflag", "COUNT(*)", "COUNT(DISTINCT id)"],
            None,
            [
                name_order("zflag", nulls_first=True),
                name_order("COUNT(*)", descending=True),
            ],
            None,
        ),
    )

    add(
        "agg5_group_by_zero_rows",
        "SELECT n, COUNT(*) FROM input WHERE id > 1000 GROUP BY n ORDER BY n",
        lambda: run_aggregate(
            filter_rows(t_rows, Cmp(">", c("id"), li(1000))),
            t_map,
            ("n",),
            [("COUNT(*)", "COUNT*", None, False)],
            ["n", "COUNT(*)"],
            None,
            [name_order("n")],
            None,
        ),
    )

    global_specs = [
        ("COUNT(*)", "COUNT*", None, False),
        ("SUM(n)", "SUM", "n", False),
    ]
    add(
        "agg6_global_having_keeps_row",
        "SELECT COUNT(*), SUM(n) FROM input HAVING COUNT(*) > 10",
        lambda: run_aggregate(
            t_rows,
            t_map,
            (),
            global_specs,
            ["COUNT(*)", "SUM(n)"],
            Cmp(">", _GAgg("COUNT(*)", "int64", False), li(10)),
            None,
            None,
        ),
        ordered=False,
    )
    add(
        "agg6b_global_having_removes_row",
        "SELECT COUNT(*), SUM(n) FROM input HAVING COUNT(*) > 1000",
        lambda: run_aggregate(
            t_rows,
            t_map,
            (),
            global_specs,
            ["COUNT(*)", "SUM(n)"],
            Cmp(">", _GAgg("COUNT(*)", "int64", False), li(1000)),
            None,
            None,
        ),
        ordered=False,
    )

    add(
        "agg7_limit_zero",
        "SELECT n, COUNT(*) FROM input WHERE n IS NOT NULL GROUP BY n ORDER BY n LIMIT 0",
        lambda: run_aggregate(
            filter_rows(t_rows, IsNull(c("n"), negate=True)),
            t_map,
            ("n",),
            [("COUNT(*)", "COUNT*", None, False)],
            ["n", "COUNT(*)"],
            None,
            [name_order("n")],
            0,
        ),
    )

    add(
        "agg8_group_by_float64_signed_zero_one_group",
        "SELECT f, COUNT(*), COUNT(DISTINCT n), SUM(n) FROM input "
        "WHERE id < 50 GROUP BY f ORDER BY f NULLS FIRST LIMIT 60",
        lambda: run_aggregate(
            filter_rows(t_rows, Cmp("<", c("id"), li(50))),
            t_map,
            ("f",),
            [
                ("COUNT(*)", "COUNT*", None, False),
                ("COUNT(DISTINCT n)", "COUNT", "n", True),
                ("SUM(n)", "SUM", "n", False),
            ],
            ["f", "COUNT(*)", "COUNT(DISTINCT n)", "SUM(n)"],
            None,
            [name_order("f", nulls_first=True)],
            60,
        ),
    )

    add(
        "agg9_distinct_float_aggregates",
        "SELECT COUNT(DISTINCT f), SUM(DISTINCT f), AVG(DISTINCT f), "
        "MIN(DISTINCT f), MAX(DISTINCT f) FROM input",
        lambda: run_aggregate(
            t_rows,
            t_map,
            (),
            [
                ("COUNT(DISTINCT f)", "COUNT", "f", True),
                ("SUM(DISTINCT f)", "SUM", "f", True),
                ("AVG(DISTINCT f)", "AVG", "f", True),
                ("MIN(DISTINCT f)", "MIN", "f", True),
                ("MAX(DISTINCT f)", "MAX", "f", True),
            ],
            [
                "COUNT(DISTINCT f)",
                "SUM(DISTINCT f)",
                "AVG(DISTINCT f)",
                "MIN(DISTINCT f)",
                "MAX(DISTINCT f)",
            ],
            None,
            None,
            None,
        ),
        ordered=False,
    )

    # -- all-NULL column ---------------------------------------------------

    add(
        "e1_group_by_all_null_column",
        "SELECT znull, COUNT(*), MIN(znull), MAX(znull), COUNT(znull) "
        "FROM input GROUP BY znull",
        lambda: run_aggregate(
            t_rows,
            t_map,
            ("znull",),
            [
                ("COUNT(*)", "COUNT*", None, False),
                ("MIN(znull)", "MIN", "znull", False),
                ("MAX(znull)", "MAX", "znull", False),
                ("COUNT(znull)", "COUNT", "znull", False),
            ],
            ["znull", "COUNT(*)", "MIN(znull)", "MAX(znull)", "COUNT(znull)"],
            None,
            None,
            None,
        ),
        ordered=False,
    )
    add(
        "e2_all_null_is_null_pushdown",
        "SELECT id, n FROM input WHERE znull IS NULL AND n IS NOT NULL "
        "ORDER BY id LIMIT 5",
        lambda: project(
            stable_sort(
                filter_rows(
                    t_rows,
                    And(IsNull(c("znull")), IsNull(c("n"), negate=True)),
                ),
                [expr_order(c("id"))],
            )[:5],
            [("id", c("id")), ("n", c("n"))],
        ),
    )
    add(
        "e3_distinct_all_null_column",
        "SELECT DISTINCT znull FROM input ORDER BY znull NULLS FIRST",
        lambda: distinct_pipeline(
            t_rows,
            [("znull", c("znull"))],
            ["utf8"],
            [name_order("znull", nulls_first=True)],
            None,
        ),
    )

    return scenarios


SINGLE_SCENARIOS = build_single_scenarios(DATASETS)


def build_join_scenarios(datasets) -> list:
    t_rows = qualify(scan(datasets["t"]), "t")
    u_rows = qualify(scan(datasets["u"]), "u")
    w_rows = qualify(scan(datasets["w"]), "w")
    t_map = schema_for(T_SCHEMA, "t")
    u_map = schema_for(U_SCHEMA, "u")
    w_map = schema_for(W_SCHEMA, "w")
    scenarios = []

    def add(name, sql, expected_fn, *, ordered=True):
        scenarios.append(
            {"name": name, "sql": sql, "expected": expected_fn, "ordered": ordered}
        )

    def cc(map_, name):
        return Col(name, map_)

    tu_inner = [("inner", "u", "t.n", "u.k")]
    tu_left = [("left", "u", "t.n", "u.k")]
    tu_right = [("right", "u", "t.n", "u.k")]
    tu_full = [("full", "u", "t.n", "u.k")]
    chain_inner = tu_inner + [("inner", "w", "u.k", "w.wk")]
    chain_left_inner = tu_left + [("inner", "w", "u.k", "w.wk")]
    chain_inner_right = tu_inner + [("right", "w", "u.k", "w.wk")]

    def joined(steps):
        return apply_steps(t_rows, u_rows, w_rows, t_map, u_map, w_map, steps)

    def mapped(steps):
        return schema_after_steps(t_map, u_map, w_map, steps)

    add(
        "j1_inner_pushdown_order_limit",
        "SELECT t.id, u.uid, t.n, u.k FROM t INNER JOIN u ON t.n = u.k "
        "WHERE t.id < 55 AND u.uid > 1002 ORDER BY t.id, u.uid LIMIT 40",
        lambda: project(
            stable_sort(
                filter_rows(
                    joined(tu_inner)[0],
                    And(
                        Cmp("<", cc(mapped(tu_inner), "t.id"), Lit(55, "int64")),
                        Cmp(">", cc(mapped(tu_inner), "u.uid"), Lit(1002, "int64")),
                    ),
                ),
                [
                    expr_order(cc(mapped(tu_inner), "t.id")),
                    expr_order(cc(mapped(tu_inner), "u.uid")),
                ],
            )[:40],
            [
                ("t.id", cc(mapped(tu_inner), "t.id")),
                ("u.uid", cc(mapped(tu_inner), "u.uid")),
                ("t.n", cc(mapped(tu_inner), "t.n")),
                ("u.k", cc(mapped(tu_inner), "u.k")),
            ],
        ),
    )

    left_map = mapped(tu_left)
    add(
        "j2_left_outer_or_predicate_nulls_first",
        "SELECT t.id, u.uid, t.n, u.tag FROM t LEFT JOIN u ON t.n = u.k "
        "WHERE (u.uid IS NULL OR t.flag = TRUE) AND t.id >= 20 "
        "ORDER BY u.uid NULLS FIRST, t.id NULLS FIRST LIMIT 50",
        lambda: project(
            stable_sort(
                filter_rows(
                    joined(tu_left)[0],
                    And(
                        Or(
                            IsNull(cc(left_map, "u.uid")),
                            Cmp("=", cc(left_map, "t.flag"), Lit(True, "bool")),
                        ),
                        Cmp(">=", cc(left_map, "t.id"), Lit(20, "int64")),
                    ),
                ),
                [
                    expr_order(cc(left_map, "u.uid"), nulls_first=True),
                    expr_order(cc(left_map, "t.id"), nulls_first=True),
                ],
            )[:50],
            [
                ("t.id", cc(left_map, "t.id")),
                ("u.uid", cc(left_map, "u.uid")),
                ("t.n", cc(left_map, "t.n")),
                ("u.tag", cc(left_map, "u.tag")),
            ],
        ),
    )

    right_map = mapped(tu_right)
    add(
        "j3_right_outer_three_valued_where",
        "SELECT t.id, u.uid, u.k, t.s FROM t RIGHT JOIN u ON t.n = u.k "
        "WHERE t.id IS NULL OR u.k >= 0 ORDER BY u.uid, t.id NULLS LAST LIMIT 60",
        lambda: project(
            stable_sort(
                filter_rows(
                    joined(tu_right)[0],
                    Or(
                        IsNull(cc(right_map, "t.id")),
                        Cmp(">=", cc(right_map, "u.k"), Lit(0, "int64")),
                    ),
                ),
                [
                    expr_order(cc(right_map, "u.uid")),
                    expr_order(cc(right_map, "t.id")),
                ],
            )[:60],
            [
                ("t.id", cc(right_map, "t.id")),
                ("u.uid", cc(right_map, "u.uid")),
                ("u.k", cc(right_map, "u.k")),
                ("t.s", cc(right_map, "t.s")),
            ],
        ),
    )

    full_map = mapped(tu_full)
    add(
        "j4_full_outer_unordered_deterministic",
        "SELECT t.id, u.uid FROM t FULL OUTER JOIN u ON t.n = u.k",
        lambda: project(
            joined(tu_full)[0],
            [
                ("t.id", cc(full_map, "t.id")),
                ("u.uid", cc(full_map, "u.uid")),
            ],
        ),
        ordered=False,
    )

    chain_map = mapped(chain_inner)
    j5_specs = [
        ("COUNT(*)", "COUNT*", None, False),
        ("COUNT(DISTINCT u.uid)", "COUNT", "u.uid", True),
        ("SUM(w.wid)", "SUM", "w.wid", False),
        ("AVG(w.wk)", "AVG", "w.wk", False),
    ]
    add(
        "j5_three_table_chain_group_having_order_limit",
        "SELECT t.s, COUNT(*), COUNT(DISTINCT u.uid), SUM(w.wid), AVG(w.wk) "
        "FROM t INNER JOIN u ON t.n = u.k INNER JOIN w ON u.k = w.wk "
        "WHERE t.flag = TRUE GROUP BY t.s "
        "HAVING COUNT(*) >= 2 AND SUM(w.wid) > 0 "
        "ORDER BY COUNT(*) DESC, t.s NULLS FIRST LIMIT 20",
        lambda: run_aggregate(
            filter_rows(
                joined(chain_inner)[0],
                Cmp("=", cc(chain_map, "t.flag"), Lit(True, "bool")),
            ),
            chain_map,
            ("t.s",),
            j5_specs,
            ["t.s", "COUNT(*)", "COUNT(DISTINCT u.uid)", "SUM(w.wid)", "AVG(w.wk)"],
            And(
                Cmp(">=", _GAgg("COUNT(*)", "int64", False), Lit(2, "int64")),
                Cmp(">", _GAgg("SUM(w.wid)", "int64", True), Lit(0, "int64")),
            ),
            [
                name_order("COUNT(*)", descending=True),
                name_order("t.s", nulls_first=True),
            ],
            20,
        ),
    )

    def join_distinct(steps, predicate, output_names, type_names, order_names, limit):
        map_ = mapped(steps)
        outputs = [(name, cc(map_, name)) for name in output_names]
        return distinct_pipeline(
            filter_rows(joined(steps)[0], predicate),
            outputs,
            type_names,
            order_names,
            limit,
        )

    add(
        "j6_inner_distinct_ordered",
        "SELECT DISTINCT t.flag, u.uflag FROM t INNER JOIN u ON t.n = u.k "
        "WHERE t.id < 40 ORDER BY t.flag DESC, u.uflag LIMIT 10",
        lambda: join_distinct(
            tu_inner,
            Cmp("<", cc(mapped(tu_inner), "t.id"), Lit(40, "int64")),
            ["t.flag", "u.uflag"],
            ["bool", "bool"],
            [
                name_order("t.flag", descending=True),
                name_order("u.uflag"),
            ],
            10,
        ),
    )
    add(
        "j7_left_distinct_first_occurrence",
        "SELECT DISTINCT t.n, u.tag FROM t LEFT JOIN u ON t.n = u.k WHERE t.id < 30",
        lambda: join_distinct(
            tu_left,
            Cmp("<", cc(left_map, "t.id"), Lit(30, "int64")),
            ["t.n", "u.tag"],
            ["int64", "utf8"],
            None,
            None,
        ),
        ordered=False,
    )

    li_map = mapped(chain_left_inner)
    add(
        "j8_left_then_inner_chain_ordered",
        "SELECT t.id, u.uid, w.wid, t.n, w.wk FROM t "
        "LEFT JOIN u ON t.n = u.k INNER JOIN w ON u.k = w.wk "
        "WHERE t.id < 20 AND w.wk < 6.0 "
        "ORDER BY t.id NULLS LAST, u.uid NULLS LAST, w.wid LIMIT 30",
        lambda: project(
            stable_sort(
                filter_rows(
                    joined(chain_left_inner)[0],
                    And(
                        Cmp("<", cc(li_map, "t.id"), Lit(20, "int64")),
                        Cmp("<", cc(li_map, "w.wk"), Lit(6.0, "float64")),
                    ),
                ),
                [
                    expr_order(cc(li_map, "t.id")),
                    expr_order(cc(li_map, "u.uid")),
                    expr_order(cc(li_map, "w.wid")),
                ],
            )[:30],
            [
                ("t.id", cc(li_map, "t.id")),
                ("u.uid", cc(li_map, "u.uid")),
                ("w.wid", cc(li_map, "w.wid")),
                ("t.n", cc(li_map, "t.n")),
                ("w.wk", cc(li_map, "w.wk")),
            ],
        ),
    )

    ir_map = mapped(chain_inner_right)
    add(
        "j9_inner_then_right_chain_ordered",
        "SELECT t.id, u.uid, w.wid FROM t INNER JOIN u ON t.n = u.k "
        "RIGHT JOIN w ON u.k = w.wk "
        "ORDER BY w.wid, u.uid NULLS LAST, t.id NULLS LAST LIMIT 50",
        lambda: project(
            stable_sort(
                joined(chain_inner_right)[0],
                [
                    expr_order(cc(ir_map, "w.wid")),
                    expr_order(cc(ir_map, "u.uid")),
                    expr_order(cc(ir_map, "t.id")),
                ],
            )[:50],
            [
                ("t.id", cc(ir_map, "t.id")),
                ("u.uid", cc(ir_map, "u.uid")),
                ("w.wid", cc(ir_map, "w.wid")),
            ],
        ),
    )

    add(
        "j10_full_group_having_null_aggregate",
        "SELECT u.uid, COUNT(*), AVG(t.f) FROM t FULL OUTER JOIN u ON t.n = u.k "
        "GROUP BY u.uid HAVING AVG(t.f) IS NOT NULL AND COUNT(*) >= 1 "
        "ORDER BY u.uid LIMIT 100",
        lambda: run_aggregate(
            joined(tu_full)[0],
            full_map,
            ("u.uid",),
            [
                ("COUNT(*)", "COUNT*", None, False),
                ("AVG(t.f)", "AVG", "t.f", False),
            ],
            ["u.uid", "COUNT(*)", "AVG(t.f)"],
            And(
                IsNull(_GAgg("AVG(t.f)", "float64", True), negate=True),
                Cmp(">=", _GAgg("COUNT(*)", "int64", False), Lit(1, "int64")),
            ),
            [name_order("u.uid")],
            100,
        ),
    )

    add(
        "j11_inner_zero_selected_ordered",
        "SELECT t.id, u.uid FROM t INNER JOIN u ON t.n = u.k "
        "WHERE t.id > 999 ORDER BY t.id, u.uid",
        lambda: project(
            stable_sort(
                filter_rows(
                    joined(tu_inner)[0],
                    Cmp(">", cc(mapped(tu_inner), "t.id"), Lit(999, "int64")),
                ),
                [
                    expr_order(cc(mapped(tu_inner), "t.id")),
                    expr_order(cc(mapped(tu_inner), "u.uid")),
                ],
            ),
            [
                ("t.id", cc(mapped(tu_inner), "t.id")),
                ("u.uid", cc(mapped(tu_inner), "u.uid")),
            ],
        ),
    )

    add(
        "j12_left_limit_zero",
        "SELECT t.id, u.uid FROM t LEFT JOIN u ON t.n = u.k "
        "WHERE t.id < 5 ORDER BY t.id, u.uid LIMIT 0",
        lambda: project(
            stable_sort(
                filter_rows(
                    joined(tu_left)[0],
                    Cmp("<", cc(left_map, "t.id"), Lit(5, "int64")),
                ),
                [
                    expr_order(cc(left_map, "t.id")),
                    expr_order(cc(left_map, "u.uid")),
                ],
            )[:0],
            [
                ("t.id", cc(left_map, "t.id")),
                ("u.uid", cc(left_map, "u.uid")),
            ],
        ),
    )

    add(
        "j13_inner_scalar_projection",
        "SELECT t.id, u.uid, t.n + u.k AS s FROM t INNER JOIN u ON t.n = u.k "
        "WHERE t.id < 20 ORDER BY s, t.id, u.uid",
        lambda: project(
            stable_sort(
                filter_rows(
                    joined(tu_inner)[0],
                    Cmp("<", cc(mapped(tu_inner), "t.id"), Lit(20, "int64")),
                ),
                [
                    expr_order(
                        Bin(
                            "+",
                            cc(mapped(tu_inner), "t.n"),
                            cc(mapped(tu_inner), "u.k"),
                        )
                    ),
                    expr_order(cc(mapped(tu_inner), "t.id")),
                    expr_order(cc(mapped(tu_inner), "u.uid")),
                ],
            ),
            [
                ("t.id", cc(mapped(tu_inner), "t.id")),
                ("u.uid", cc(mapped(tu_inner), "u.uid")),
                ("s", Bin("+", cc(mapped(tu_inner), "t.n"), cc(mapped(tu_inner), "u.k"))),
            ],
        ),
    )

    star_names = [
        *(f"t.{name}" for name in T_SCHEMA.names),
        *(f"u.{name}" for name in U_SCHEMA.names),
        *(f"w.{name}" for name in W_SCHEMA.names),
    ]
    add(
        "j14_three_table_star_unordered",
        "SELECT * FROM t INNER JOIN u ON t.n = u.k INNER JOIN w ON u.k = w.wk "
        "WHERE t.id < 25",
        lambda: (
            [(name, *chain_map[name]) for name in star_names],
            [
                tuple(row[name] for name in star_names)
                for row in filter_rows(
                    joined(chain_inner)[0],
                    Cmp("<", cc(chain_map, "t.id"), Lit(25, "int64")),
                )
            ],
        ),
        ordered=False,
    )

    return scenarios


JOIN_SCENARIOS = build_join_scenarios(DATASETS)


# ---------------------------------------------------------------------------
# Layout variants written through the public entries
# ---------------------------------------------------------------------------

COMPRESSIONS = ("none", "zlib")
DICT_CHOICES = (False, True)
V2_GROUP_SIZES = (1, 3, 7, 64)


def _label(kind, group_size, compression, dictionary):
    if kind == "v1":
        return f"v1_{compression}_{'dict' if dictionary else 'plain'}"
    return f"v2r{group_size}_{compression}_{'dict' if dictionary else 'plain'}"


def _write_variant(
    path, table, kind, group_size, compression, dictionary, dict_columns
):
    encoding = dict_columns if dictionary else ()
    if kind == "v1":
        write_file(path, table, compression=compression, dictionary_encoding=encoding)
    else:
        write_partitioned_file(
            path,
            table,
            group_size,
            compression=compression,
            dictionary_encoding=encoding,
        )


T_VARIANT_LABELS = sorted(
    [_label("v1", None, c, d) for c in COMPRESSIONS for d in DICT_CHOICES]
    + [
        _label("v2", g, c, d)
        for g in V2_GROUP_SIZES
        for c in COMPRESSIONS
        for d in DICT_CHOICES
    ]
)

PARTNER_SPECS = {
    "u": [
        ("v1", None, "none", False),
        ("v1", None, "none", True),
        ("v1", None, "zlib", True),
        ("v2", 5, "none", False),
        ("v2", 3, "zlib", True),
        ("v2", 40, "none", True),
        ("v2", 40, "none", False),
    ],
    "w": [
        ("v1", None, "none", False),
        ("v1", None, "none", True),
        ("v1", None, "zlib", True),
        ("v2", 2, "none", False),
        ("v2", 7, "zlib", True),
        ("v2", 24, "none", True),
    ],
}

# Five source-layout combinations per join run: all v1, all v2, two mixed
# version/compression/encoding matrices, and v2 singleton row groups.
JOIN_COMBOS = [
    ("all_v1", {"t": "v1_none_plain", "u": "v1_none_plain", "w": "v1_zlib_dict"}),
    (
        "all_v2",
        {"t": "v2r3_zlib_dict", "u": "v2r5_none_plain", "w": "v2r2_none_plain"},
    ),
    (
        "mixed_v1_v2_a",
        {"t": "v2r7_none_dict", "u": "v2r40_none_dict", "w": "v1_none_dict"},
    ),
    (
        "mixed_v1_v2_b",
        {"t": "v1_zlib_dict", "u": "v2r3_zlib_dict", "w": "v2r7_zlib_dict"},
    ),
    (
        "v2_singleton_groups",
        {"t": "v2r1_zlib_plain", "u": "v1_zlib_dict", "w": "v2r2_none_plain"},
    ),
]

EMPTY_SPECS = [
    ("v1_none_plain", "v1", None, "none", False),
    ("v1_zlib_dict", "v1", None, "zlib", True),
    ("v2r3_none_plain", "v2", 3, "none", False),
    ("v2r3_zlib_dict", "v2", 3, "zlib", True),
]


@pytest.fixture(scope="module")
def layouts(tmp_path_factory):
    """Write every t variant plus a fixed variant set for u and w, once."""
    directory = tmp_path_factory.mktemp("differential_layouts")
    tables = {
        "t": Table(T_SCHEMA, DATASETS["t"]),
        "u": Table(U_SCHEMA, DATASETS["u"]),
        "w": Table(W_SCHEMA, DATASETS["w"]),
    }
    dict_columns = {"t": ["s", "znull"], "u": ["tag"], "w": ["wtag"]}

    t_paths = {}
    for label in T_VARIANT_LABELS:
        kind = "v1" if label.startswith("v1_") else "v2"
        if kind == "v1":
            _compression = "zlib" if "zlib" in label else "none"
            _dictionary = label.endswith("dict")
            spec = ("v1", None, _compression, _dictionary)
        else:
            group_size = int(label.split("r")[1].split("_")[0])
            _compression = "zlib" if "zlib" in label else "none"
            _dictionary = label.endswith("dict")
            spec = ("v2", group_size, _compression, _dictionary)
        path = directory / f"t_{label}.caef"
        _write_variant(
            path, tables["t"], *spec, dict_columns["t"]
        )
        t_paths[label] = path

    partner_paths = {}
    for table_name in ("u", "w"):
        partner_paths[table_name] = {}
        for kind, group_size, compression, dictionary in PARTNER_SPECS[table_name]:
            label = _label(kind, group_size, compression, dictionary)
            path = directory / f"{table_name}_{label}.caef"
            _write_variant(
                path,
                tables[table_name],
                kind,
                group_size,
                compression,
                dictionary,
                dict_columns[table_name],
            )
            partner_paths[table_name][label] = path

    return {"t": t_paths, "u": partner_paths["u"], "w": partner_paths["w"]}


@pytest.fixture(scope="module")
def empty_layouts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("differential_empty")
    table = Table(
        T_SCHEMA, {name: [] for name in T_SCHEMA.names}
    )
    paths = {}
    for label, kind, group_size, compression, dictionary in EMPTY_SPECS:
        path = directory / f"empty_{label}.caef"
        _write_variant(
            path, table, kind, group_size, compression, dictionary, ["s", "znull"]
        )
        paths[label] = path
    return paths


# ---------------------------------------------------------------------------
# Result comparison (public Table surface only)
# ---------------------------------------------------------------------------


def table_view(table: Table):
    names = table.column_names
    return (
        [(col.name, col.type, col.nullable) for col in table.schema.columns],
        [tuple(table.column(name)[i] for name in names) for i in range(table.row_count)],
    )


def _first_row_diff(expected, got):
    if len(expected) != len(got):
        return f"row count differs: expected {len(expected)}, got {len(got)}"
    for i, (expected_row, got_row) in enumerate(zip(expected, got)):
        if expected_row != got_row:
            return (
                f"first differing row is #{i}:\n"
                f"expected: {expected_row}\ngot: {got_row}"
            )
    return "rows differ"


def assert_matches_reference(table: Table, expected, *, sql: str, provenance: str):
    exp_schema, exp_rows = expected
    got_schema, got_rows = table_view(table)
    assert got_schema == exp_schema, (
        f"schema mismatch ({provenance})\nSQL: {sql}\n"
        f"expected: {exp_schema}\ngot: {got_schema}"
    )
    assert got_rows == exp_rows, (
        f"row/value/NULL-position mismatch ({provenance})\nSQL: {sql}\n"
        f"{_first_row_diff(exp_rows, got_rows)}"
    )


def assert_three_repeats_match(runner, expected, *, sql: str, provenance: str):
    """Run one scenario three times; every run matches the reference and all
    three views are identical, pinning the deterministic no-ORDER-BY order."""
    first = None
    for repeat in range(3):
        table = runner()
        view = table_view(table)
        assert_matches_reference(
            table,
            expected,
            sql=sql,
            provenance=f"{provenance} / repeat {repeat + 1}",
        )
        if first is None:
            first = view
        else:
            assert view == first, f"non-repeatable result ({provenance})\nSQL: {sql}"


# ---------------------------------------------------------------------------
# Single table: every storage layout, three repeats per scenario
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario", SINGLE_SCENARIOS, ids=[s["name"] for s in SINGLE_SCENARIOS]
)
@pytest.mark.parametrize("variant", T_VARIANT_LABELS)
def test_single_table_layout_differential(layouts, scenario, variant):
    assert_three_repeats_match(
        lambda: query_file(layouts["t"][variant], scenario["sql"]),
        scenario["expected"](),
        sql=scenario["sql"],
        provenance=(
            f"{scenario['name']} / {variant} / seeds "
            f"{SEED_MAIN},{SEED_U},{SEED_W}"
        ),
    )


# ---------------------------------------------------------------------------
# Byte-identical CSV / JSONL exports across versions / compression / encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["csv", "jsonl"])
def test_single_table_exports_byte_identical_across_layouts(
    layouts, tmp_path_factory, fmt
):
    directory = tmp_path_factory.mktemp(f"single_export_{fmt}")
    ordered = [scenario for scenario in SINGLE_SCENARIOS if scenario["ordered"]]
    assert ordered
    for scenario in ordered:
        expected_schema, expected_rows = scenario["expected"]()
        blobs = {}
        for variant in T_VARIANT_LABELS:
            destination = directory / f"{scenario['name']}_{variant}.{fmt}"
            count = export_query_file(
                layouts["t"][variant], scenario["sql"], destination, fmt
            )
            # A repeat export of the same input + SQL + format stays stable.
            again = directory / f"{scenario['name']}_{variant}_again.{fmt}"
            assert (
                export_query_file(layouts["t"][variant], scenario["sql"], again, fmt)
                == count
            )
            assert again.read_bytes() == destination.read_bytes()
            blobs[variant] = destination.read_bytes()
            assert count == len(expected_rows)
        baseline = blobs[T_VARIANT_LABELS[0]]
        for variant in T_VARIANT_LABELS[1:]:
            assert blobs[variant] == baseline, (
                f"{fmt} export differs across layouts for {scenario['name']}: "
                f"{T_VARIANT_LABELS[0]} vs {variant}\nSQL: {scenario['sql']}"
            )
        # The exported bytes also describe exactly the reference rows.
        if fmt == "jsonl":
            lines = baseline.decode("utf-8").splitlines()
            assert len(lines) == len(expected_rows)
            expected_names = [name for name, _t, _n in expected_schema]
            for line, exp_row in zip(lines, expected_rows):
                obj = json.loads(line)
                assert list(obj) == expected_names
                assert tuple(obj[name] for name in expected_names) == exp_row
        else:
            decoded = baseline.decode("utf-8")
            parsed = list(csv_module.reader(decoded.splitlines()))
            assert parsed[0] == [name for name, _t, _n in expected_schema]
            assert decoded.endswith("\n")
            assert len(parsed) - 1 == len(expected_rows)


# ---------------------------------------------------------------------------
# Empty table across v1/v2 layouts
# ---------------------------------------------------------------------------


EMPTY_QUERIES = [
    (
        "empty_star",
        "SELECT * FROM input",
        lambda: (
            [
                (name, *schema_for(T_SCHEMA)[name]) for name in T_SCHEMA.names
            ],
            [],
        ),
    ),
    (
        "empty_ordered",
        "SELECT id, n FROM input WHERE n > 0 ORDER BY id",
        lambda: project(
            [],
            [
                ("id", Col("id", schema_for(T_SCHEMA))),
                ("n", Col("n", schema_for(T_SCHEMA))),
            ],
        ),
    ),
    (
        "empty_global_aggregate",
        "SELECT COUNT(*), COUNT(n), SUM(n), AVG(f), MIN(s), MAX(znull) FROM input",
        lambda: run_aggregate(
            [],
            schema_for(T_SCHEMA),
            (),
            [
                ("COUNT(*)", "COUNT*", None, False),
                ("COUNT(n)", "COUNT", "n", False),
                ("SUM(n)", "SUM", "n", False),
                ("AVG(f)", "AVG", "f", False),
                ("MIN(s)", "MIN", "s", False),
                ("MAX(znull)", "MAX", "znull", False),
            ],
            ["COUNT(*)", "COUNT(n)", "SUM(n)", "AVG(f)", "MIN(s)", "MAX(znull)"],
            None,
            None,
            None,
        ),
    ),
    (
        "empty_group_by",
        "SELECT n, COUNT(*) FROM input GROUP BY n ORDER BY n",
        lambda: run_aggregate(
            [],
            schema_for(T_SCHEMA),
            ("n",),
            [("COUNT(*)", "COUNT*", None, False)],
            ["n", "COUNT(*)"],
            None,
            [name_order("n")],
            None,
        ),
    ),
    (
        "empty_distinct",
        "SELECT DISTINCT s, flag FROM input ORDER BY s, flag",
        lambda: ([("s", "utf8", True), ("flag", "bool", False)], []),
    ),
]


@pytest.mark.parametrize("label", [spec[0] for spec in EMPTY_SPECS])
@pytest.mark.parametrize(
    "query", EMPTY_QUERIES, ids=[q[0] for q in EMPTY_QUERIES]
)
def test_empty_table_layouts(empty_layouts, label, query):
    _name, sql, expected_fn = query
    assert_three_repeats_match(
        lambda: query_file(empty_layouts[label], sql),
        expected_fn(),
        sql=sql,
        provenance=f"empty/{_name}/{label}",
    )


@pytest.mark.parametrize("fmt", ["csv", "jsonl"])
def test_empty_table_exports_byte_identical(empty_layouts, tmp_path_factory, fmt):
    directory = tmp_path_factory.mktemp(f"empty_export_{fmt}")
    sql = EMPTY_QUERIES[1][1]  # the ordered empty query
    blobs = {}
    for label in [spec[0] for spec in EMPTY_SPECS]:
        destination = directory / f"empty_{label}.{fmt}"
        count = export_query_file(empty_layouts[label], sql, destination, fmt)
        assert count == 0
        blobs[label] = destination.read_bytes()
    values = list(blobs.values())
    assert all(blob == values[0] for blob in values[1:])
    if fmt == "csv":
        assert values[0].decode("utf-8") == "id,n\n"
    else:
        assert values[0] == b""


# ---------------------------------------------------------------------------
# Multi-table: every join kind, 3-table chains, strategies vs default path
# ---------------------------------------------------------------------------


def _sources_for(layouts, combo):
    return {
        table_name: layouts[table_name][label] for table_name, label in combo.items()
    }


@pytest.mark.parametrize(
    "scenario", JOIN_SCENARIOS, ids=[s["name"] for s in JOIN_SCENARIOS]
)
@pytest.mark.parametrize("strategy", [None, "hash", "sort_merge"])
@pytest.mark.parametrize("combo_name,combo", JOIN_COMBOS)
def test_join_strategy_layout_differential(
    layouts, scenario, strategy, combo_name, combo
):
    sources = _sources_for(layouts, combo)
    label = "default" if strategy is None else strategy
    assert_three_repeats_match(
        lambda: query_files(sources, scenario["sql"], strategy),
        scenario["expected"](),
        sql=scenario["sql"],
        provenance=(
            f"{scenario['name']} / {combo_name} / {label} / seeds "
            f"{SEED_MAIN},{SEED_U},{SEED_W}"
        ),
    )


@pytest.mark.parametrize(
    "scenario", JOIN_SCENARIOS, ids=[s["name"] for s in JOIN_SCENARIOS]
)
@pytest.mark.parametrize("combo_name,combo", JOIN_COMBOS)
def test_join_strategies_agree_with_default(layouts, scenario, combo_name, combo):
    sources = _sources_for(layouts, combo)
    default_view = table_view(query_files(sources, scenario["sql"]))
    hash_view = table_view(query_files(sources, scenario["sql"], "hash"))
    merge_view = table_view(query_files(sources, scenario["sql"], "sort_merge"))
    assert hash_view == default_view, (
        f"hash path differs from default for {scenario['name']} / {combo_name}\n"
        f"SQL: {scenario['sql']}"
    )
    assert merge_view == default_view, (
        f"sort_merge path differs from default for {scenario['name']} / {combo_name}\n"
        f"SQL: {scenario['sql']}"
    )


@pytest.mark.parametrize("fmt", ["csv", "jsonl"])
@pytest.mark.parametrize("combo_name,combo", JOIN_COMBOS)
def test_join_exports_byte_identical_across_strategies_and_layouts(
    layouts, combo_name, combo, tmp_path_factory, fmt
):
    directory = tmp_path_factory.mktemp(f"join_export_{fmt}_{combo_name}")
    sources = _sources_for(layouts, combo)
    ordered = [scenario for scenario in JOIN_SCENARIOS if scenario["ordered"]]
    assert ordered
    for scenario in ordered:
        _expected_schema, expected_rows = scenario["expected"]()
        collected = {}
        for strategy in (None, "hash", "sort_merge"):
            destination = directory / f"{scenario['name']}_{strategy}.{fmt}"
            count = export_query_files(
                sources, scenario["sql"], destination, fmt, strategy
            )
            assert count == len(expected_rows)
            collected[strategy] = destination.read_bytes()
        assert collected["hash"] == collected[None], (
            f"{fmt} export: hash differs from default for "
            f"{scenario['name']}/{combo_name}\nSQL: {scenario['sql']}"
        )
        assert collected["sort_merge"] == collected[None], (
            f"{fmt} export: sort_merge differs from default for "
            f"{scenario['name']}/{combo_name}\nSQL: {scenario['sql']}"
        )

    # The same ordered scenarios must also export identically across the
    # different source-layout combinations (default join path).
    cross_directory = tmp_path_factory.mktemp(f"join_export_cross_{fmt}")
    for scenario in ordered:
        blobs = {}
        for cname, ccombo in JOIN_COMBOS:
            sources = _sources_for(layouts, ccombo)
            destination = cross_directory / f"{scenario['name']}_{cname}.{fmt}"
            export_query_files(sources, scenario["sql"], destination, fmt)
            blobs[cname] = destination.read_bytes()
        baseline = blobs[JOIN_COMBOS[0][0]]
        for cname, _ccombo in JOIN_COMBOS[1:]:
            assert blobs[cname] == baseline, (
                f"{fmt} export differs across source layouts for "
                f"{scenario['name']}: {JOIN_COMBOS[0][0]} vs {cname}\n"
                f"SQL: {scenario['sql']}"
            )


# ---------------------------------------------------------------------------
# Evidence that the compared runs really exercise distinct layouts / paths
# ---------------------------------------------------------------------------
#
# The differential comparisons above are only meaningful if the variants
# genuinely differ on disk and in planning.  These checks use the public
# metadata / explain entries to prove it (no private helpers):
# * v1 vs v2 format versions and per-group row counts,
# * v2 statistics pruning (selected groups < total) for a pushable WHERE and
#   *no* pruning for a non-pushable arithmetic predicate,
# * the two join strategies surfacing distinct HASH / SORT_MERGE labels,
# * OUTER chains disabling pruning,
# * the JOIN-less multi-file entry honouring either strategy inertly.


def test_layouts_really_span_v1_and_v2(layouts):
    from columnar_analytics import inspect_row_groups

    assert inspect_file(layouts["t"]["v1_zlib_dict"])["format_version"] == 1
    assert inspect_file(layouts["t"]["v2r3_none_plain"])["format_version"] == 2
    n = len(DATASETS["t"]["id"])
    for group_size in V2_GROUP_SIZES:
        label = _label("v2", group_size, "none", False)
        group_rows = [
            group["row_count"] for group in inspect_row_groups(layouts["t"][label])
        ]
        expected_counts = [group_size] * (n // group_size) + (
            [n % group_size] if n % group_size else []
        )
        assert group_rows == expected_counts
    # v1 carries no row groups.
    assert inspect_row_groups(layouts["t"]["v1_none_plain"]) == []


def test_v2_statistics_pushdown_actually_prunes_and_matches(layouts):
    # n >= 2 AND f < 5.0 AND id <= 50: with row_group_size 7 the first
    # group (id 0..6) still has candidates, but a deliberately extreme
    # predicate must exclude at least one whole group while keeping others.
    selective_sql = (
        "SELECT id FROM input WHERE n >= 2 AND f < 5.0 AND id <= 50 ORDER BY id"
    )
    v2_path = layouts["t"]["v2r3_none_plain"]
    v1_path = layouts["t"]["v1_none_plain"]
    plan_v2 = explain_file(v2_path, selective_sql)
    scan_v2 = plan_v2["operators"][0]
    assert scan_v2["row_groups_total"] > 0
    # The bounds n in [-8, 12] / f span the groups; at minimum the pushed
    # condition is reported and selected <= total.
    assert scan_v2["pushed_condition"] is not None
    assert scan_v2["row_groups_selected"] < scan_v2["row_groups_total"]

    # A predicate impossible for any group excludes every group; the query
    # then sees zero rows and selected is zero (path genuinely switched).
    impossible_sql = "SELECT id FROM input WHERE id > 1000"
    impossible_plan = explain_file(v2_path, impossible_sql)
    impossible_scan = impossible_plan["operators"][0]
    assert impossible_scan["row_groups_selected"] == 0
    assert table_view(query_file(v2_path, impossible_sql)) == (
        [("id", "int64", False)],
        [],
    )

    # A purely arithmetic predicate has no pushable leaf: nothing is pruned
    # (selected == total, pushed_condition null) and the full row-level WHERE
    # still returns the same rows as v1.
    non_pushable_sql = (
        "SELECT id FROM input WHERE f * 2.0 > 1.0 ORDER BY id LIMIT 30"
    )
    np_plan = explain_file(v2_path, non_pushable_sql)
    np_scan = next(op for op in np_plan["operators"] if op["operator"] == "Scan")
    assert np_scan["pushed_condition"] is None
    assert np_scan["row_groups_selected"] == np_scan["row_groups_total"]
    assert table_view(query_file(v2_path, non_pushable_sql)) == table_view(
        query_file(v1_path, non_pushable_sql)
    )

    # An OR-nested pushable leaf is not pushed either, while its eligible
    # siblings on the top-level AND spine still are (documented behaviour).
    mixed_sql = (
        "SELECT id FROM input WHERE (f * 2.0 > 1.0 OR zflag IS NOT NULL) "
        "AND n IS NOT NULL ORDER BY id LIMIT 30"
    )
    mixed_plan = explain_file(v2_path, mixed_sql)
    mixed_scan = next(
        op for op in mixed_plan["operators"] if op["operator"] == "Scan"
    )
    assert mixed_scan["pushed_condition"] is not None
    assert table_view(query_file(v2_path, mixed_sql)) == table_view(
        query_file(v1_path, mixed_sql)
    )


@pytest.mark.parametrize("strategy,label", [("hash", "HASH"), ("sort_merge", "SORT_MERGE")])
def test_explain_surfaces_distinct_join_strategies(layouts, strategy, label):
    combo = JOIN_COMBOS[1][1]  # all v2
    sources = _sources_for(layouts, combo)
    sql = "SELECT * FROM t INNER JOIN u ON t.n = u.k"
    plan = explain_files(sources, sql, strategy)
    joins = [op for op in plan["operators"] if op["operator"] == "Join"]
    assert [join.get("strategy") for join in joins] == [label]
    default_plan = explain_files(sources, sql)
    assert all("strategy" not in join for join in default_plan["operators"] if join["operator"] == "Join")


def test_explain_outer_chain_disables_group_pruning(layouts):
    # A chain containing a LEFT step performs no row-group pruning, while an
    # all-INNER chain on the same v2 sources does.
    inner_sources = _sources_for(layouts, JOIN_COMBOS[1][1])
    left_sql = (
        "SELECT t.id, u.uid FROM t LEFT JOIN u ON t.n = u.k WHERE t.id < 55"
    )
    inner_sql = (
        "SELECT t.id, u.uid FROM t INNER JOIN u ON t.n = u.k WHERE t.id < 55"
    )
    left_plan = explain_files(inner_sources, left_sql)
    inner_plan = explain_files(inner_sources, inner_sql)
    for scan in (op for op in left_plan["operators"] if op["operator"] == "Scan"):
        assert "row_groups_total" not in scan
    inner_scans = [op for op in inner_plan["operators"] if op["operator"] == "Scan"]
    assert any("row_groups_total" in scan for scan in inner_scans)


@pytest.mark.parametrize("strategy", [None, "hash", "sort_merge"])
def test_joinless_multi_file_entry_matches_single_file(layouts, strategy):
    # The multi-file entry with a JOIN-less statement reads one source and
    # must accept either strategy inertly, on both v1 and v2 layouts.
    for variant in ("v1_zlib_dict", "v2r7_none_dict"):
        source_path = layouts["t"][variant]
        sql = (
            "SELECT id, n, f FROM input WHERE n >= 2 AND f < 5.0 "
            "ORDER BY id LIMIT 20"
        )
        single = table_view(query_file(source_path, sql))
        multi = table_view(query_files({"input": source_path}, sql, strategy))
        assert multi == single, (variant, strategy)
        # The v2 single-source plan still carries pruning metadata.
        if variant.startswith("v2"):
            plan = explain_files({"input": source_path}, sql, strategy)
            scan = next(op for op in plan["operators"] if op["operator"] == "Scan")
            assert "row_groups_total" in scan
        else:
            plan = explain_files({"input": source_path}, sql, strategy)
            scan = next(op for op in plan["operators"] if op["operator"] == "Scan")
            assert "row_groups_total" not in scan
        assert not [op for op in plan["operators"] if op["operator"] == "Join"]
