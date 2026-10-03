"""Deterministic differential tests across storage layouts and execution paths.

Every dataset is generated from a fixed seed (``SEED``) and written through
the public write entry points: as a v1 file (:func:`write_file`) and as v2
row-group-partitioned files (:func:`write_partitioned_file`) with several
legal ``row_group_size`` values, combined with ``none``/``zlib`` compression
and utf8 dictionary encoding.  A curated set of legal SQL statements (no
generated text outside the public grammar) runs through the public
``query_file`` / ``query_files`` / ``export_query_file`` /
``export_query_files`` entry points on every file variant and — for
multi-table statements — with the default, ``hash`` and ``sort_merge`` join
paths.

Expected results never come from another execution path of the engine: an
independent, row-level reference evaluator in this file implements the
documented NULL three-valued logic, join expansion, grouping, deduplication
and stable-ordering semantics directly over the generated rows.  Only public
``Table`` attributes (``schema``, ``column_names``, ``columns``,
``row_count``) are read back.  Every scenario repeats each run at least
three times.  Assertion messages always carry the seed and the exact SQL
text, so any failure reproduces directly from both.
"""

from __future__ import annotations

import json
import math
import random
import struct
from dataclasses import dataclass, replace
from functools import cmp_to_key
from types import SimpleNamespace

import pytest

from columnar_analytics import (
    ColumnSchema,
    Schema,
    Table,
    export_query_file,
    export_query_files,
    query_file,
    query_files,
    write_file,
    write_partitioned_file,
)

SEED = 20261003
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
REPEATS = 3

# ---------------------------------------------------------------------------
# Seeded data generation
# ---------------------------------------------------------------------------


def _shuffled(rng, values):
    values = list(values)
    rng.shuffle(values)
    return values


def build_datasets():
    """All datasets from one fixed seed; sizes stay small on purpose."""
    rng = random.Random(SEED)
    n = 120
    ids = rng.sample(range(1, 400), n)
    flags = _shuffled(rng, [True] * 50 + [False] * 40 + [None] * 30)
    scores = _shuffled(
        rng,
        [0.0] * 12
        + [-0.0] * 12
        + [1.5] * 20
        + [-2.25] * 15
        + [3.0] * 10
        + [0.5] * 15
        + [2.5] * 16
        + [None] * 20,
    )
    names = _shuffled(
        rng,
        ["alpha"] * 20
        + ["beta"] * 15
        + ["数据"] * 15
        + ["héllo"] * 10
        + ["🚀launch"] * 8
        + [""] * 7
        + ["O'Brien"] * 5
        + ["gamma δ"] * 10
        + [None] * 30,
    )
    bigs = _shuffled(
        rng,
        [INT64_MAX - 1] * 15
        + [INT64_MAX - 7] * 15
        + [INT64_MIN + 1] * 15
        + [INT64_MIN + 10] * 15
        + [0] * 20
        + [-1] * 20
        + [None] * 20,
    )
    main_schema = Schema(
        [
            ColumnSchema("id", "int64"),
            ColumnSchema("flag", "bool", nullable=True),
            ColumnSchema("score", "float64", nullable=True),
            ColumnSchema("name", "utf8", nullable=True),
            ColumnSchema("big", "int64", nullable=True),
            ColumnSchema("all_null", "utf8", nullable=True),
        ]
    )
    main_columns = {
        "id": ids,
        "flag": flags,
        "score": scores,
        "name": names,
        "big": bigs,
        "all_null": [None] * n,
    }
    main = Table(main_schema, main_columns)
    empty = Table(main_schema, {name: [] for name in main_schema.names})

    left = Table(
        Schema(
            [
                ColumnSchema("id", "int64"),
                ColumnSchema("k", "int64", nullable=True),
                ColumnSchema("name", "utf8", nullable=True),
            ]
        ),
        {
            "id": list(range(1, 13)),
            # Duplicate keys (10/20/30), NULL keys and unmatched keys (40/60).
            "k": _shuffled(rng, [10, 10, 10, 20, 20, 30, 30, 40, 40, 60, None, None]),
            "name": _shuffled(
                rng,
                ["ada", "bob", "数据", None, "ada", "cy", "bob", "ada", None, "数据", "di", "bob"],
            ),
        },
    )
    right = Table(
        Schema(
            [
                ColumnSchema("rid", "int64"),
                ColumnSchema("k", "int64", nullable=True),
                ColumnSchema("val", "float64", nullable=True),
                ColumnSchema("fk", "float64", nullable=True),
                ColumnSchema("tag", "utf8", nullable=True),
            ]
        ),
        {
            "rid": list(range(101, 111)),
            "k": _shuffled(rng, [10, 10, 20, 30, 30, 30, None, 50, 50, 20]),
            "val": _shuffled(rng, [0.0, -0.0, 1.5, None, 2.5, -0.0, 0.0, 1.5, None, 2.5]),
            "fk": _shuffled(rng, [0.0, -0.0, 1.5, 2.5, None, 0.0, 1.5, 2.5, None, -0.0]),
            "tag": _shuffled(rng, ["x", "y", "数据", None, "x", "y", None, "x", "数据", "z"]),
        },
    )
    third = Table(
        Schema(
            [
                ColumnSchema("tid", "int64"),
                ColumnSchema("fk", "float64", nullable=True),
                ColumnSchema("flag", "bool", nullable=True),
            ]
        ),
        {
            "tid": list(range(1001, 1009)),
            # 9.5 matches nothing on the r side; 0.0/-0.0 match across sides.
            "fk": _shuffled(rng, [0.0, 1.5, -0.0, 2.5, None, 9.5, 1.5, 0.0]),
            "flag": _shuffled(rng, [True, False, None, True, False, None, True, False]),
        },
    )
    return {"main": main, "empty": empty, "l": left, "r": right, "t": third}


# ---------------------------------------------------------------------------
# File variants (public write entry points only)
# ---------------------------------------------------------------------------

VARIANTS = (
    "v1-none",
    "v1-zlib-dict",
    "v2-rg1-zlib",
    "v2-rg5-none-dict",
    "v2-rg23-zlib-dict",
    "v2-rg1000-none",
)


def _utf8_columns(table):
    return [c.name for c in table.schema.columns if c.type == "utf8"]


def write_variant(path, table, variant):
    if variant == "v1-none":
        write_file(path, table, compression="none")
    elif variant == "v1-zlib-dict":
        write_file(path, table, compression="zlib", dictionary_encoding=_utf8_columns(table))
    elif variant == "v2-rg1-zlib":
        write_partitioned_file(path, table, 1, compression="zlib")
    elif variant == "v2-rg5-none-dict":
        write_partitioned_file(path, table, 5, compression="none", dictionary_encoding=_utf8_columns(table))
    elif variant == "v2-rg23-zlib-dict":
        write_partitioned_file(path, table, 23, compression="zlib", dictionary_encoding=_utf8_columns(table))
    elif variant == "v2-rg1000-none":
        write_partitioned_file(path, table, 1000, compression="none")
    else:  # pragma: no cover - defensive
        raise AssertionError(f"unknown variant {variant!r}")


# Which file variant each join source uses per combination.
SOURCE_COMBOS = {
    "all-v1": {"l": "v1-none", "r": "v1-none", "t": "v1-none"},
    "all-v2": {"l": "v2-rg5-none-dict", "r": "v2-rg23-zlib-dict", "t": "v2-rg1-zlib"},
    "mixed": {"l": "v1-zlib-dict", "r": "v2-rg1-zlib", "t": "v2-rg1000-none"},
}

# ---------------------------------------------------------------------------
# Query specification DSL: one object renders the SQL text and drives the
# independent reference evaluator, so the two can never drift apart.
#
# Expression nodes (tuples):
#   ("lit", value)                        literal (bool/int/float/str)
#   ("col", name)                         column reference (qualified in joins)
#   ("gcol", name)                        GROUP BY column inside HAVING
#   ("agg", FUNC, arg|None, distinct)     aggregate call (None arg = COUNT(*))
#   ("cmp", op, l, r)                     = != < <= > >=
#   ("and"/"or", a, b) / ("not", a)
#   ("isnull", a) / ("isnotnull", a)
#   ("arith", op, l, r)                   + - * /
#   ("case", ((cond, res), ...), else|None)
#
# SELECT items: ("star",) | ("col", name) | ("expr", node, alias)
#               | ("agg", FUNC, arg|None, distinct)
# ORDER BY keys: (node, descending, nulls) where node is ("col", name),
#               ("alias", alias) or ("agg", ...); nulls is None/"first"/"last".
# ---------------------------------------------------------------------------


def LIT(v):
    return ("lit", v)


def COL(n):
    return ("col", n)


def GCOL(n):
    return ("gcol", n)


def AGG(func, arg=None, distinct=False):
    return ("agg", func, arg, distinct)


def CMP(op, l, r):
    return ("cmp", op, l, r)


def AND(a, b):
    return ("and", a, b)


def OR(a, b):
    return ("or", a, b)


def NOT(a):
    return ("not", a)


def ISNULL(a):
    return ("isnull", a)


def NOTNULL(a):
    return ("isnotnull", a)


def ARITH(op, l, r):
    return ("arith", op, l, r)


def CASE(whens, els=None):
    return ("case", tuple(whens), els)


def S_STAR():
    return ("star",)


def S_COL(n):
    return ("col", n)


def S_EXPR(node, alias):
    return ("expr", node, alias)


def S_AGG(func, arg=None, distinct=False):
    return ("agg", func, arg, distinct)


def O_COL(n, desc=False, nulls=None):
    return (("col", n), desc, nulls)


def O_ALIAS(a, desc=False, nulls=None):
    return (("alias", a), desc, nulls)


def O_AGG(func, arg=None, distinct=False, desc=False, nulls=None):
    return (("agg", func, arg, distinct), desc, nulls)


_JOIN_SQL = {
    "inner": "INNER JOIN",
    "left": "LEFT JOIN",
    "right": "RIGHT JOIN",
    "full": "FULL OUTER JOIN",
}


@dataclass(frozen=True)
class JoinSpec:
    kind: str          # "inner" | "left" | "right" | "full"
    table: str         # SQL name of the freshly introduced table
    dataset: str       # key into the datasets dict
    left_ref: str      # qualified key on the already-introduced side
    right_col: str     # bare join-key column of the new table


@dataclass(frozen=True)
class QuerySpec:
    qid: str
    from_table: str    # SQL table name ("input" for single-file statements)
    from_dataset: str  # key into the datasets dict
    select: tuple
    joins: tuple = ()
    distinct: bool = False
    where: tuple | None = None
    group_by: tuple = ()
    having: tuple | None = None
    order_by: tuple = ()
    limit: int | None = None


# ---------------------------------------------------------------------------
# SQL rendering (public grammar only)
# ---------------------------------------------------------------------------


def render_literal(v):
    if v is True:
        return "TRUE"
    if v is False:
        return "FALSE"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def render_agg(node):
    _tag, func, arg, distinct = node
    if arg is None:
        return f"{func}(*)"
    if distinct:
        return f"{func}(DISTINCT {arg})"
    return f"{func}({arg})"


def render_node(node):
    tag = node[0]
    if tag == "lit":
        return render_literal(node[1])
    if tag in ("col", "gcol"):
        return node[1]
    if tag == "agg":
        return render_agg(node)
    if tag == "cmp":
        return f"{render_node(node[2])} {node[1]} {render_node(node[3])}"
    if tag in ("and", "or"):
        return f"({render_node(node[1])} {tag.upper()} {render_node(node[2])})"
    if tag == "not":
        return f"NOT ({render_node(node[1])})"
    if tag == "isnull":
        return f"{render_node(node[1])} IS NULL"
    if tag == "isnotnull":
        return f"{render_node(node[1])} IS NOT NULL"
    if tag == "arith":
        return f"({render_node(node[2])} {node[1]} {render_node(node[3])})"
    if tag == "case":
        parts = ["CASE"]
        for cond, res in node[1]:
            parts.append(f"WHEN {render_node(cond)} THEN {render_node(res)}")
        if node[2] is not None:
            parts.append(f"ELSE {render_node(node[2])}")
        parts.append("END")
        return " ".join(parts)
    raise AssertionError(f"cannot render node {node!r}")  # pragma: no cover


def render_item(item):
    if item[0] == "star":
        return "*"
    if item[0] == "col":
        return item[1]
    if item[0] == "expr":
        return f"{render_node(item[1])} AS {item[2]}"
    return render_agg(item)


def render_order_key(order_key):
    node, descending, nulls = order_key
    if node[0] == "col":
        text = node[1]
    elif node[0] == "alias":
        text = node[1]
    else:
        text = render_agg(node)
    text += " DESC" if descending else " ASC"
    if nulls is not None:
        text += f" NULLS {nulls.upper()}"
    return text


def render_sql(spec):
    parts = ["SELECT"]
    if spec.distinct:
        parts.append("DISTINCT")
    parts.append(", ".join(render_item(item) for item in spec.select))
    parts.append(f"FROM {spec.from_table}")
    for join in spec.joins:
        parts.append(f"{_JOIN_SQL[join.kind]} {join.table} ON {join.left_ref} = {join.table}.{join.right_col}")
    if spec.where is not None:
        parts.append(f"WHERE {render_node(spec.where)}")
    if spec.group_by:
        parts.append("GROUP BY " + ", ".join(spec.group_by))
    if spec.having is not None:
        parts.append(f"HAVING {render_node(spec.having)}")
    if spec.order_by:
        parts.append("ORDER BY " + ", ".join(render_order_key(k) for k in spec.order_by))
    if spec.limit is not None:
        parts.append(f"LIMIT {spec.limit}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Independent row-level reference evaluator
# ---------------------------------------------------------------------------

_CMP_OPS = {
    "=": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def _ref_arith(op, left, right):
    """Mirror the documented arithmetic rules; scenarios never trigger them."""
    if op == "/" or isinstance(left, float) or isinstance(right, float):
        if op == "/" and right == 0:
            raise AssertionError("reference hit division by zero")
        a, b = float(left), float(right)
        result = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op]
        if not math.isfinite(result):
            raise AssertionError("reference produced a non-finite float64")
        return result
    result = {"+": left + right, "-": left - right, "*": left * right}[op]
    if not (INT64_MIN <= result <= INT64_MAX):
        raise AssertionError("reference overflowed the int64 range")
    return result


class Evaluator:
    """Three-valued expression evaluation plus static type/nullability."""

    def __init__(self, cols):
        # cols: name -> (type_name, nullable)
        self.cols = cols

    def eval(self, node, row, aggs=None):
        tag = node[0]
        if tag == "lit":
            return node[1]
        if tag in ("col", "gcol"):
            return row[node[1]]
        if tag == "agg":
            return aggs[(node[1], node[2], node[3])]
        if tag == "cmp":
            left = self.eval(node[2], row, aggs)
            right = self.eval(node[3], row, aggs)
            if left is None or right is None:
                return None
            return bool(_CMP_OPS[node[1]](left, right))
        if tag == "not":
            value = self.eval(node[1], row, aggs)
            return None if value is None else (not value)
        if tag == "and":
            left = self.eval(node[1], row, aggs)
            if left is False:
                return False
            right = self.eval(node[2], row, aggs)
            if right is False:
                return False
            return None if (left is None or right is None) else True
        if tag == "or":
            left = self.eval(node[1], row, aggs)
            if left is True:
                return True
            right = self.eval(node[2], row, aggs)
            if right is True:
                return True
            return None if (left is None or right is None) else False
        if tag == "isnull":
            return self.eval(node[1], row, aggs) is None
        if tag == "isnotnull":
            return self.eval(node[1], row, aggs) is not None
        if tag == "arith":
            left = self.eval(node[2], row, aggs)
            right = self.eval(node[3], row, aggs)
            if left is None or right is None:
                return None
            return _ref_arith(node[1], left, right)
        if tag == "case":
            value = None
            hit = False
            for cond, res in node[1]:
                if self.eval(cond, row, aggs) is True:
                    value = self.eval(res, row, aggs)
                    hit = True
                    break
            if not hit and node[2] is not None:
                value = self.eval(node[2], row, aggs)
            # int64/float64 results unify to float64; ints become floats.
            if (
                value is not None
                and self.type_of(node) == "float64"
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                return float(value)
            return value
        raise AssertionError(f"cannot evaluate node {node!r}")  # pragma: no cover

    def type_of(self, node):
        tag = node[0]
        if tag == "lit":
            value = node[1]
            if isinstance(value, bool):
                return "bool"
            if isinstance(value, int):
                return "int64"
            if isinstance(value, float):
                return "float64"
            return "utf8"
        if tag in ("col", "gcol"):
            return self.cols[node[1]][0]
        if tag == "agg":
            _tag, func, arg, _distinct = node
            if func == "COUNT":
                return "int64"
            if func == "AVG":
                return "float64"
            return self.cols[arg][0]  # SUM / MIN / MAX keep the argument type
        if tag in ("cmp", "and", "or", "not", "isnull", "isnotnull"):
            return "bool"
        if tag == "arith":
            left_t = self.type_of(node[2])
            right_t = self.type_of(node[3])
            if node[1] == "/" or "float64" in (left_t, right_t):
                return "float64"
            return "int64"
        if tag == "case":
            types = {self.type_of(res) for _cond, res in node[1]}
            if node[2] is not None:
                types.add(self.type_of(node[2]))
            if types <= {"int64", "float64"}:
                return "float64" if "float64" in types else "int64"
            return next(iter(types))
        raise AssertionError(f"cannot type node {node!r}")  # pragma: no cover

    def nullable_of(self, node):
        tag = node[0]
        if tag == "lit":
            return False
        if tag in ("col", "gcol"):
            return self.cols[node[1]][1]
        if tag == "agg":
            return node[1] != "COUNT"
        if tag == "arith":
            return self.nullable_of(node[2]) or self.nullable_of(node[3])
        if tag == "case":
            if node[2] is None:
                return True
            return any(self.nullable_of(res) for _cond, res in node[1]) or self.nullable_of(node[2])
        return False  # boolean nodes never appear as SELECT expressions here


def _walk(node):
    """Yield a HAVING/WHERE tree's nodes (no arith/case inside HAVING)."""
    yield node
    tag = node[0]
    if tag in ("and", "or"):
        yield from _walk(node[1])
        yield from _walk(node[2])
    elif tag in ("not", "isnull", "isnotnull"):
        yield from _walk(node[1])
    elif tag == "cmp":
        yield from _walk(node[2])
        yield from _walk(node[3])


def _distinct_values(values):
    """First-seen distinct non-NULL values; float64 0.0 == -0.0."""
    seen = set()
    out = []
    for value in values:
        key = (True, 0.0 if (isinstance(value, float) and value == 0) else value)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _compute_agg(key, rows, cols):
    func, arg, distinct = key
    if func == "COUNT" and arg is None:
        return len(rows)
    values = [row[arg] for row in rows if row[arg] is not None]
    if distinct:
        values = _distinct_values(values)
    if func == "COUNT":
        return len(values)
    if not values:
        return None
    if func == "MIN":
        return min(values)
    if func == "MAX":
        return max(values)
    if func == "SUM":
        if cols[arg][0] == "int64":
            total = sum(values)
            if not (INT64_MIN <= total <= INT64_MAX):
                raise AssertionError("reference SUM overflowed int64")
            return total
        total = math.fsum(values)
        if not math.isfinite(total):
            raise AssertionError("reference SUM produced a non-finite float64")
        return total
    # AVG
    result = math.fsum(values) / len(values)
    if not math.isfinite(result):
        raise AssertionError("reference AVG produced a non-finite float64")
    return result


def _compare_values(a, b, descending, nulls_first):
    if a is None or b is None:
        if a is None and b is None:
            return 0
        # NULL placement follows NULLS FIRST/LAST alone; the default keeps
        # NULLs at the end for both directions.
        none_before = -1 if nulls_first else 1
        return none_before if a is None else -none_before
    c = (a > b) - (a < b)
    return -c if descending else c


def _sort_rows(rows, key_of, order_by):
    def compare(x, y):
        for order_key in order_by:
            _node, descending, nulls = order_key
            c = _compare_values(key_of(order_key, x), key_of(order_key, y), descending, nulls == "first")
            if c:
                return c
        return 0

    return sorted(rows, key=cmp_to_key(compare))


def _lift_table(table, prefix):
    names = table.column_names
    columns = table.columns
    if prefix is None:
        schema = [(c.name, c.type, c.nullable) for c in table.schema.columns]
        key_of = lambda n: n
    else:
        schema = [(f"{prefix}.{c.name}", c.type, c.nullable) for c in table.schema.columns]
        key_of = lambda n: f"{prefix}.{n}"
    rows = [{key_of(n): columns[n][i] for n in names} for i in range(table.row_count)]
    return schema, rows


def _apply_join(schema, rows, new_table, join):
    """One equi-join step with the documented expansion and row order."""
    new_schema = [(f"{join.table}.{c.name}", c.type, c.nullable) for c in new_table.schema.columns]
    pad_left = join.kind in ("right", "full")
    pad_right = join.kind in ("left", "full")
    out_schema = [(n, t, True if pad_left else nu) for n, t, nu in schema] + [
        (n, t, True if pad_right else nu) for n, t, nu in new_schema
    ]
    new_names = new_table.column_names
    new_columns = new_table.columns
    new_rows = [
        {f"{join.table}.{c}": new_columns[c][i] for c in new_names}
        for i in range(new_table.row_count)
    ]
    right_key = f"{join.table}.{join.right_col}"

    # NULL keys never match; duplicate keys produce the full combination.
    index = {}
    for j, new_row in enumerate(new_rows):
        key = new_row[right_key]
        if key is not None:
            index.setdefault(key, []).append(j)
    matches_left = {}
    matches_right = {}
    for i, row in enumerate(rows):
        key = row[join.left_ref]
        if key is None:
            continue
        matches = index.get(key)
        if matches:
            matches_left[i] = matches
            for j in matches:
                matches_right.setdefault(j, []).append(i)

    left_pad = {n: None for n, _t, _nu in schema}
    right_pad = {n: None for n, _t, _nu in new_schema}
    out_rows = []
    if join.kind == "right":
        # Driven by the new table's file order; combinations of one new row
        # expand in the current row order.
        for j, new_row in enumerate(new_rows):
            left_matches = matches_right.get(j)
            if left_matches:
                out_rows.extend({**rows[i], **new_row} for i in left_matches)
            else:
                out_rows.append({**left_pad, **new_row})
        return out_schema, out_rows
    for i, row in enumerate(rows):
        right_matches = matches_left.get(i)
        if right_matches:
            out_rows.extend({**row, **new_rows[j]} for j in right_matches)
        elif join.kind in ("left", "full"):
            out_rows.append({**row, **right_pad})
    if join.kind == "full":
        for j, new_row in enumerate(new_rows):
            if j not in matches_right:
                out_rows.append({**left_pad, **new_row})
    return out_schema, out_rows


def _build_joined(spec, datasets):
    qualify = bool(spec.joins)
    schema, rows = _lift_table(datasets[spec.from_dataset], spec.from_table if qualify else None)
    for join in spec.joins:
        schema, rows = _apply_join(schema, rows, datasets[join.dataset], join)
    return schema, rows


def _expand_star(items, schema):
    out = []
    for item in items:
        if item[0] == "star":
            out.extend(("col", name) for name, _t, _nu in schema)
        else:
            out.append(item)
    return out


def _agg_label(func, arg, distinct):
    if arg is None:
        return "COUNT(*)"
    if distinct:
        return f"{func}(DISTINCT {arg})"
    return f"{func}({arg})"


def _run_plain(spec, schema, ev, rows):
    items = _expand_star(spec.select, schema)
    cols = ev.cols
    out_schema = []
    for item in items:
        if item[0] == "col":
            out_schema.append((item[1],) + cols[item[1]])
        else:
            out_schema.append((item[2], ev.type_of(item[1]), ev.nullable_of(item[1])))
    if spec.order_by:
        alias_exprs = {item[2]: item[1] for item in items if item[0] == "expr"}

        def key_of(order_key, row):
            node = order_key[0]
            if node[0] == "col":
                return row[node[1]]
            return ev.eval(alias_exprs[node[1]], row)

        rows = _sort_rows(rows, key_of, spec.order_by)
    if spec.limit is not None:
        rows = rows[: spec.limit]
    out_rows = [
        tuple(row[item[1]] if item[0] == "col" else ev.eval(item[1], row) for item in items)
        for row in rows
    ]
    return out_schema, out_rows


def _dedup_key_part(type_name, value):
    if value is None:
        return (False,)
    if type_name == "float64":
        return (True, 0.0 if value == 0 else value)
    return (True, value)


def _run_distinct(spec, schema, ev, rows):
    items = _expand_star(spec.select, schema)
    cols = ev.cols
    out_schema = []
    for item in items:
        if item[0] == "col":
            out_schema.append((item[1],) + cols[item[1]])
        else:
            out_schema.append((item[2], ev.type_of(item[1]), ev.nullable_of(item[1])))
    projected = [
        tuple(row[item[1]] if item[0] == "col" else ev.eval(item[1], row) for item in items)
        for row in rows
    ]
    seen = set()
    unique = []
    for tup in projected:
        key = tuple(_dedup_key_part(out_schema[i][1], value) for i, value in enumerate(tup))
        if key not in seen:
            seen.add(key)
            unique.append(tup)
    if spec.order_by:
        def key_of(order_key, tup):
            node = order_key[0]
            for i, item in enumerate(items):
                if node[0] == "col" and item[0] == "col" and item[1] == node[1]:
                    return tup[i]
                if node[0] == "alias" and item[0] == "expr" and item[2] == node[1]:
                    return tup[i]
            raise AssertionError(node)  # pragma: no cover

        unique = _sort_rows(unique, key_of, spec.order_by)
    if spec.limit is not None:
        unique = unique[: spec.limit]
    return out_schema, unique


def _run_aggregate(spec, schema, ev, rows):
    items = spec.select
    cols = ev.cols
    if spec.group_by:
        buckets = {}
        order = []
        for row in rows:
            key = tuple(row[g] for g in spec.group_by)
            bucket = buckets.get(key)
            if bucket is None:
                buckets[key] = bucket = []
                order.append(key)
            bucket.append(row)
        groups = [buckets[key] for key in order]
    else:
        # The whole filtered stream is one group, even when empty.
        groups = [rows]

    agg_keys = []

    def need(key):
        if key not in agg_keys:
            agg_keys.append(key)

    for item in items:
        if item[0] == "agg":
            need((item[1], item[2], item[3]))
    if spec.having is not None:
        for node in _walk(spec.having):
            if node[0] == "agg":
                need((node[1], node[2], node[3]))

    materialised = []
    for group_rows in groups:
        agg_values = {key: _compute_agg(key, group_rows, cols) for key in agg_keys}
        if spec.having is not None:
            representative = group_rows[0] if group_rows else {}
            if ev.eval(spec.having, representative, agg_values) is not True:
                continue
        materialised.append(
            tuple(
                group_rows[0][item[1]] if item[0] == "col" else agg_values[(item[1], item[2], item[3])]
                for item in items
            )
        )

    if spec.order_by:
        def key_of(order_key, tup):
            node = order_key[0]
            for i, item in enumerate(items):
                if node[0] == "col" and item[0] == "col" and item[1] == node[1]:
                    return tup[i]
                if node[0] == "agg" and item[0] == "agg" and item[1:] == node[1:]:
                    return tup[i]
            raise AssertionError(node)  # pragma: no cover

        materialised = _sort_rows(materialised, key_of, spec.order_by)
    if spec.limit is not None:
        materialised = materialised[: spec.limit]

    out_schema = []
    for item in items:
        if item[0] == "col":
            out_schema.append((item[1],) + cols[item[1]])
        else:
            _tag, func, arg, distinct = item
            out_schema.append((_agg_label(func, arg, distinct), ev.type_of(item), ev.nullable_of(item)))
    return out_schema, materialised


def run_reference(spec, datasets):
    """Independently evaluate ``spec`` over the in-memory generated rows."""
    schema, rows = _build_joined(spec, datasets)
    cols = {name: (t, nullable) for name, t, nullable in schema}
    ev = Evaluator(cols)
    if spec.where is not None:
        rows = [row for row in rows if ev.eval(spec.where, row) is True]
    is_aggregate = (
        bool(spec.group_by)
        or any(item[0] == "agg" for item in spec.select)
        or (spec.having is not None and any(n[0] == "agg" for n in _walk(spec.having)))
    )
    if is_aggregate:
        return _run_aggregate(spec, schema, ev, rows)
    if spec.distinct:
        return _run_distinct(spec, schema, ev, rows)
    return _run_plain(spec, schema, ev, rows)


# ---------------------------------------------------------------------------
# Independent export renderers (documented CSV / JSONL byte layout)
# ---------------------------------------------------------------------------


def _csv_text(text):
    if any(ch in text for ch in (",", '"', "\r", "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


def _csv_scalar(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '""' if value == "" else _csv_text(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def render_export(fmt, schema, rows):
    if fmt == "csv":
        lines = [",".join(_csv_text(name) for name, _t, _nu in schema)]
        for row in rows:
            lines.append(",".join(_csv_scalar(value) for value in row))
        return "".join(line + "\n" for line in lines).encode("utf-8")
    names = [name for name, _t, _nu in schema]
    lines = [
        json.dumps(
            {names[i]: row[i] for i in range(len(names))},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        for row in rows
    ]
    return "".join(line + "\n" for line in lines).encode("utf-8")


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


def _single(qid, **kwargs):
    return QuerySpec(qid=qid, from_table="input", from_dataset="main", **kwargs)


_BAND_CASE = CASE(
    [
        (ISNULL(COL("score")), LIT("none")),
        (CMP(">=", COL("score"), LIT(1.5)), LIT("high")),
    ],
    LIT("low"),
)

S1 = _single(
    "where-tvl-mixed-pushdown",
    select=(S_COL("id"), S_COL("name"), S_COL("flag")),
    # id >= 20 / id <= 400 are statistics-pushable AND leaves; the OR/NOT
    # branch is not, so v2 pruning and the full row filter must agree.
    where=AND(
        CMP(">=", COL("id"), LIT(20)),
        AND(
            CMP("<=", COL("id"), LIT(400)),
            OR(NOTNULL(COL("name")), NOT(CMP("=", COL("flag"), LIT(False)))),
        ),
    ),
    order_by=(O_COL("flag", desc=True, nulls="first"), O_COL("id")),
    limit=15,
)

S2 = _single(
    "projection-case-expr",
    select=(
        S_COL("id"),
        S_EXPR(ARITH("+", ARITH("*", COL("id"), LIT(2)), LIT(1)), "calc"),
        S_EXPR(_BAND_CASE, "band"),
        S_EXPR(CASE([(CMP("<", COL("score"), LIT(0.0)), LIT("neg"))]), "sign"),
        S_EXPR(CASE([(CMP("=", COL("flag"), LIT(True)), LIT(1))], LIT(2.5)), "numcase"),
    ),
    # An arithmetic operand makes this leaf non-pushable.
    where=CMP("<", ARITH("*", COL("id"), LIT(3)), LIT(200)),
    order_by=(O_ALIAS("calc", desc=True), O_COL("id")),
    limit=12,
)

S3A = _single(
    "zero-rows",
    select=(S_COL("id"), S_COL("score")),
    where=CMP(">", COL("id"), LIT(1000000)),
    order_by=(O_COL("id"),),
)

S3B = _single(
    "limit-zero",
    select=(S_COL("id"), S_COL("flag")),
    where=ISNULL(COL("flag")),
    limit=0,
)

S4 = _single(
    "global-aggregates",
    select=(
        S_AGG("COUNT"),
        S_AGG("COUNT", "score"),
        S_AGG("SUM", "id"),
        S_AGG("AVG", "score"),
        S_AGG("MIN", "name"),
        S_AGG("MAX", "big"),
        S_AGG("COUNT", "name", distinct=True),
        S_AGG("SUM", "score", distinct=True),
    ),
    where=OR(NOTNULL(COL("flag")), CMP("<", COL("id"), LIT(50))),
)

S5 = _single(
    "group-having-distinct-agg",
    select=(
        S_COL("flag"),
        S_AGG("COUNT"),
        S_AGG("COUNT", "name", distinct=True),
        S_AGG("AVG", "score"),
    ),
    group_by=("flag",),
    # SUM(id) is HAVING-only (never projected); AVG(score) may be NULL for a
    # group, exercising three-valued HAVING.
    having=AND(
        CMP(">=", AGG("COUNT"), LIT(2)),
        OR(CMP(">", AGG("AVG", "score"), LIT(0.0)), ISNULL(AGG("SUM", "id"))),
    ),
    order_by=(O_COL("flag", nulls="first"), O_AGG("COUNT", desc=True)),
)

S6A = _single(
    "group-nullable-utf8",
    select=(S_COL("name"), S_AGG("COUNT")),
    group_by=("name",),
    order_by=(O_AGG("COUNT", desc=True), O_COL("name", nulls="last")),
)

S6B = _single(
    "group-all-null",
    select=(S_COL("all_null"), S_AGG("COUNT")),
    group_by=("all_null",),
)

S7A = _single(
    "distinct-first-appearance",
    select=(S_COL("flag"), S_COL("name")),
    distinct=True,
    where=CMP("<=", COL("id"), LIT(250)),
)

S7B = _single(
    "distinct-expr-order",
    select=(S_EXPR(_BAND_CASE, "band"), S_COL("flag")),
    distinct=True,
    where=CMP("<", COL("id"), LIT(300)),
    order_by=(O_ALIAS("band"), O_COL("flag", desc=True, nulls="first")),
    limit=6,
)

S7C = _single(
    "distinct-float-signed-zero",
    select=(S_COL("score"),),
    distinct=True,
    where=NOTNULL(COL("score")),
    order_by=(O_COL("score"),),
)

S8 = _single(
    "order-by-alias-nulls",
    select=(S_COL("id"), S_EXPR(ARITH("+", COL("score"), LIT(1.0)), "s")),
    order_by=(O_ALIAS("s", nulls="first"), O_COL("id", desc=True)),
    limit=10,
)

S9A = _single(
    "zero-match-global-agg",
    select=(S_AGG("COUNT"), S_AGG("SUM", "id"), S_AGG("AVG", "score"), S_AGG("MIN", "name")),
    where=CMP(">", COL("id"), LIT(1000000)),
)

S9B = _single(
    "zero-match-having-drops",
    select=(S_AGG("COUNT"), S_AGG("SUM", "id")),
    where=CMP(">", COL("id"), LIT(1000000)),
    having=CMP(">", AGG("COUNT"), LIT(0)),
)

S9C = _single(
    "zero-match-having-keeps",
    select=(S_AGG("COUNT"),),
    where=CMP(">", COL("id"), LIT(1000000)),
    having=CMP("=", AGG("COUNT"), LIT(0)),
)

S10 = _single(
    "case-as-where",
    select=(S_COL("id"),),
    where=CASE(
        [
            (ISNULL(COL("score")), COL("flag")),
            (CMP(">", COL("score"), LIT(1.0)), LIT(True)),
        ],
        LIT(False),
    ),
    order_by=(O_COL("id"),),
    limit=9,
)

S11 = _single(
    "arith-projection-types",
    select=(
        S_COL("id"),
        S_EXPR(ARITH("/", COL("id"), LIT(2)), "half"),
        S_EXPR(ARITH("*", COL("score"), LIT(2.0)), "ds"),
    ),
    # IS NOT NULL is a pushable leaf; id >= 1 is a pushable comparison.
    where=AND(NOTNULL(COL("score")), CMP(">=", COL("id"), LIT(1))),
    order_by=(O_COL("id"),),
    limit=8,
)

S12 = _single(
    "bool-case-projection",
    select=(
        S_COL("id"),
        S_EXPR(CASE([(ISNULL(COL("score")), LIT(True))], COL("flag")), "sflag"),
    ),
    order_by=(O_COL("id"),),
    limit=11,
)

SINGLE_SCENARIOS = [S1, S2, S3A, S3B, S4, S5, S6A, S6B, S7A, S7B, S7C, S8, S9A, S9B, S9C, S10, S11, S12]

# The same statements against the all-empty table (zero input rows).
EMPTY_SCENARIOS = [
    replace(spec, from_dataset="empty", qid=f"{spec.qid}-empty")
    for spec in (S1, S4, S5, S7A, S9A)
]

SINGLE_SCENARIOS = SINGLE_SCENARIOS + EMPTY_SCENARIOS


def _join(qid, joins, **kwargs):
    return QuerySpec(qid=qid, from_table="l", from_dataset="l", joins=joins, **kwargs)


J1 = _join(
    "inner-dup-expansion",
    [JoinSpec("inner", "r", "r", "l.k", "k")],
    select=(S_STAR(),),
    order_by=(O_COL("l.id"), O_COL("r.rid")),
)

J2 = _join(
    "left-join-padded-where",
    [JoinSpec("left", "r", "r", "l.k", "k")],
    select=(S_COL("l.id"), S_COL("r.rid"), S_COL("r.val")),
    where=OR(ISNULL(COL("r.rid")), CMP(">=", COL("r.val"), LIT(0.0))),
    order_by=(O_COL("l.id"), O_COL("r.rid", nulls="first")),
)

J3 = _join(
    "right-join",
    [JoinSpec("right", "r", "r", "l.k", "k")],
    select=(S_COL("l.id"), S_COL("l.name"), S_COL("r.rid")),
    order_by=(O_COL("r.rid", desc=True), O_COL("l.id", nulls="first")),
    limit=14,
)

J4 = _join(
    "full-outer-natural-order",
    [JoinSpec("full", "r", "r", "l.k", "k")],
    select=(S_COL("l.id"), S_COL("r.rid")),
)

J5 = _join(
    "three-table-inner-float-keys",
    [JoinSpec("inner", "r", "r", "l.k", "k"), JoinSpec("inner", "t", "t", "r.fk", "fk")],
    select=(S_COL("l.id"), S_COL("r.rid"), S_COL("t.tid")),
    order_by=(O_COL("l.id"), O_COL("r.rid"), O_COL("t.tid")),
)

J6 = _join(
    "left-left-chain",
    [JoinSpec("left", "r", "r", "l.k", "k"), JoinSpec("left", "t", "t", "r.fk", "fk")],
    select=(S_COL("l.id"), S_COL("r.rid"), S_COL("t.tid")),
    order_by=(O_COL("l.id"), O_COL("r.rid", nulls="first"), O_COL("t.tid", nulls="first")),
    limit=25,
)

J7 = _join(
    "inner-full-chain",
    [JoinSpec("inner", "r", "r", "l.k", "k"), JoinSpec("full", "t", "t", "r.fk", "fk")],
    select=(S_COL("l.id"), S_COL("r.rid"), S_COL("t.tid")),
    order_by=(O_COL("t.tid", nulls="first"), O_COL("l.id", nulls="first"), O_COL("r.rid", nulls="first")),
)

J8 = _join(
    "join-group-having",
    [JoinSpec("left", "r", "r", "l.k", "k")],
    select=(S_COL("l.name"), S_AGG("COUNT"), S_AGG("SUM", "r.rid")),
    group_by=("l.name",),
    having=CMP(">=", AGG("COUNT"), LIT(1)),
    order_by=(O_COL("l.name", nulls="last"),),
)

J9 = _join(
    "join-distinct",
    [JoinSpec("inner", "r", "r", "l.k", "k")],
    select=(S_COL("l.name"), S_COL("r.tag")),
    distinct=True,
)

J10 = _join(
    "join-zero-match",
    [JoinSpec("inner", "r", "r", "l.k", "k")],
    select=(S_COL("l.id"), S_COL("r.rid")),
    where=CMP(">", COL("l.id"), LIT(100000)),
    order_by=(O_COL("l.id"),),
)

J11 = _join(
    "full-outer-aggregates",
    [JoinSpec("full", "r", "r", "l.k", "k")],
    select=(
        S_AGG("COUNT"),
        S_AGG("COUNT", "r.rid"),
        S_AGG("MIN", "l.id"),
        S_AGG("MAX", "r.val"),
        S_AGG("COUNT", "r.tag", distinct=True),
    ),
)

J12 = _join(
    "right-right-chain",
    [JoinSpec("right", "r", "r", "l.k", "k"), JoinSpec("right", "t", "t", "r.fk", "fk")],
    select=(S_COL("l.id"), S_COL("r.rid"), S_COL("t.tid")),
    order_by=(O_COL("t.tid", nulls="first"), O_COL("r.rid", nulls="first"), O_COL("l.id", nulls="first")),
    limit=30,
)

JOIN_SCENARIOS = [J1, J2, J3, J4, J5, J6, J7, J8, J9, J10, J11, J12]

EXPORT_SINGLE_SCENARIOS = [S1, S2, S5, S8]
EXPORT_JOIN_SCENARIOS = [J1, J6]

# ---------------------------------------------------------------------------
# Fixtures and comparison helpers (public Table attributes only)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    datasets = build_datasets()
    root = tmp_path_factory.mktemp("differential")
    files = {key: {} for key in datasets}
    for key, table in datasets.items():
        for variant in VARIANTS:
            path = root / f"{key}-{variant}.caef"
            write_variant(path, table, variant)
            files[key][variant] = path
    return SimpleNamespace(datasets=datasets, files=files)


def public_result(table):
    """Schema triples and row tuples read via public Table attributes."""
    names = table.column_names
    columns = table.columns
    schema = [(c.name, c.type, c.nullable) for c in table.schema.columns]
    rows = [tuple(columns[n][i] for n in names) for i in range(table.row_count)]
    return schema, rows


def values_match(expected, actual):
    if expected is None or actual is None:
        return expected is None and actual is None
    if type(expected) is not type(actual):
        return False
    if isinstance(expected, float):
        # Bit-exact: 0.0 and -0.0 must not be confused.
        return struct.pack("<d", expected) == struct.pack("<d", actual)
    return expected == actual


def assert_result(actual, expected, ctx):
    exp_schema, exp_rows = expected
    act_schema, act_rows = actual
    assert act_schema == exp_schema, (
        f"{ctx}\nschema mismatch:\nexpected={exp_schema}\nactual  ={act_schema}"
    )
    assert len(act_rows) == len(exp_rows), (
        f"{ctx}\nrow count: expected {len(exp_rows)}, got {len(act_rows)}"
    )
    for i, (exp_row, act_row) in enumerate(zip(exp_rows, act_rows)):
        for j, (exp_value, act_value) in enumerate(zip(exp_row, act_row)):
            assert values_match(exp_value, act_value), (
                f"{ctx}\nrow {i} column {exp_schema[j][0]!r}: "
                f"expected {exp_value!r}, got {act_value!r}"
            )


def _sources_for(spec, env, combo):
    sources = {spec.from_table: env.files[spec.from_dataset][combo[spec.from_dataset]]}
    for join in spec.joins:
        sources[join.table] = env.files[join.dataset][combo[join.dataset]]
    return sources


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_generated_data_covers_required_shapes(env):
    """The fixed seed must keep producing every shape the scenarios rely on."""
    ds = env.datasets
    main = ds["main"]
    cols = main.columns
    assert main.row_count == 120
    assert ds["empty"].row_count == 0
    assert {v for v in cols["flag"] if v is not None} == {True, False}
    assert any(v is None for v in cols["flag"])
    zero_signs = {
        math.copysign(1.0, v) for v in cols["score"] if isinstance(v, float) and v == 0.0
    }
    assert zero_signs == {1.0, -1.0}  # both +0.0 and -0.0 present
    assert any(v is None for v in cols["score"])
    assert any(isinstance(v, str) and not v.isascii() for v in cols["name"])
    assert "" in cols["name"]
    assert any(v is None for v in cols["name"])
    assert len({v for v in cols["name"] if v is not None}) < len([v for v in cols["name"] if v is not None])
    assert any(v is not None and v > 2**62 for v in cols["big"])
    assert any(v is not None and v < -(2**62) for v in cols["big"])
    assert all(v is None for v in cols["all_null"])

    left, right, third = ds["l"].columns, ds["r"].columns, ds["t"].columns
    l_keys = [v for v in left["k"] if v is not None]
    r_keys = [v for v in right["k"] if v is not None]
    assert len(set(l_keys)) < len(l_keys)  # duplicate join keys on the left
    assert len(set(r_keys)) < len(r_keys)  # duplicate join keys on the right
    assert any(v is None for v in left["k"]) and any(v is None for v in right["k"])
    assert set(l_keys) - set(r_keys)  # unmatched left keys
    assert set(r_keys) - set(l_keys)  # unmatched right keys
    for side in (right["fk"], third["fk"]):
        signs = {math.copysign(1.0, v) for v in side if isinstance(v, float) and v == 0.0}
        assert signs == {1.0, -1.0}
        assert any(v is None for v in side)
    r_fk = {v for v in right["fk"] if v is not None}
    t_fk = {v for v in third["fk"] if v is not None}
    assert r_fk & t_fk  # the float-key chain produces matches
    assert t_fk - r_fk  # unmatched new-side keys for FULL OUTER chains


@pytest.mark.parametrize("spec", SINGLE_SCENARIOS, ids=lambda s: s.qid)
def test_single_file_layouts_match_reference(env, spec):
    sql = render_sql(spec)
    expected = run_reference(spec, env.datasets)
    for variant in VARIANTS:
        path = env.files[spec.from_dataset][variant]
        for run in range(REPEATS):
            actual = public_result(query_file(path, sql))
            assert_result(
                actual,
                expected,
                f"seed={SEED} scenario={spec.qid} variant={variant} run={run} sql={sql}",
            )


@pytest.mark.parametrize("spec", JOIN_SCENARIOS, ids=lambda s: s.qid)
@pytest.mark.parametrize("combo", sorted(SOURCE_COMBOS))
@pytest.mark.parametrize("strategy", [None, "hash", "sort_merge"])
def test_join_paths_match_reference(env, spec, combo, strategy):
    sql = render_sql(spec)
    expected = run_reference(spec, env.datasets)
    sources = _sources_for(spec, env, SOURCE_COMBOS[combo])
    for run in range(REPEATS):
        actual = public_result(query_files(sources, sql, strategy))
        assert_result(
            actual,
            expected,
            f"seed={SEED} scenario={spec.qid} combo={combo} strategy={strategy} "
            f"run={run} sql={sql}",
        )


@pytest.mark.parametrize("spec", EXPORT_SINGLE_SCENARIOS, ids=lambda s: s.qid)
@pytest.mark.parametrize("fmt", ["csv", "jsonl"])
def test_single_file_exports_byte_identical(env, spec, fmt, tmp_path):
    sql = render_sql(spec)
    schema, rows = run_reference(spec, env.datasets)
    expected_bytes = render_export(fmt, schema, rows)
    for variant in VARIANTS:
        for run in range(2):
            dest = tmp_path / f"{spec.qid}-{variant}-{run}.{fmt}"
            written = export_query_file(env.files[spec.from_dataset][variant], sql, dest, format=fmt)
            ctx = f"seed={SEED} scenario={spec.qid} variant={variant} run={run} fmt={fmt} sql={sql}"
            assert written == len(rows), ctx
            assert dest.read_bytes() == expected_bytes, ctx


@pytest.mark.parametrize("spec", EXPORT_JOIN_SCENARIOS, ids=lambda s: s.qid)
@pytest.mark.parametrize("fmt", ["csv", "jsonl"])
def test_join_exports_byte_identical(env, spec, fmt, tmp_path):
    sql = render_sql(spec)
    schema, rows = run_reference(spec, env.datasets)
    expected_bytes = render_export(fmt, schema, rows)
    for combo_name, combo in sorted(SOURCE_COMBOS.items()):
        sources = _sources_for(spec, env, combo)
        for strategy in (None, "hash", "sort_merge"):
            dest = tmp_path / f"{spec.qid}-{combo_name}-{strategy or 'default'}.{fmt}"
            written = export_query_files(sources, sql, dest, format=fmt, join_strategy=strategy)
            ctx = (
                f"seed={SEED} scenario={spec.qid} combo={combo_name} strategy={strategy} "
                f"fmt={fmt} sql={sql}"
            )
            assert written == len(rows), ctx
            assert dest.read_bytes() == expected_bytes, ctx
