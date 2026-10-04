"""Bound expression IR and its row-wise / batched evaluation.

The binder (:mod:`columnar_analytics.binder`) turns parsed tuple trees
into the :class:`_Expr` hierarchy defined here; every later stage --
type and nullability derivation, dependency collection, plan rendering,
predicate pushdown and the evaluators -- walks these bound nodes.  This
module owns the per-row semantics (three-valued logic, NULL propagation,
int64 / float64 error rules) and the batched column-vector evaluation
that shares them exactly.
"""

from __future__ import annotations

import math

from .errors import QueryValidationError
from .parser import _INT64_MAX, _INT64_MIN

# ---------------------------------------------------------------------------
# Bound expression IR -- one node hierarchy for every later stage
# ---------------------------------------------------------------------------
#
# Binding resolves a parsed tuple tree against one :class:`Schema` and
# produces :class:`_Expr` nodes.  Each node carries, once and for all:
#
# * ``type``     -- its static result type (``int64`` / ``float64`` /
#                   ``bool`` / ``utf8``); predicates are ``bool``;
# * ``nullable`` -- whether the node may yield NULL, derived while binding
#                   from the participating columns and the node kind;
# * ``columns()``-- every bound source-column index referenced anywhere in
#                   the subtree (multiplicity and in-tree order preserved);
# * ``children`` -- the node's direct bound subexpressions in fixed order,
#                   so generic traversals never re-switch on a node tag;
# * ``to_json()``-- its explain-plan condition / projection tree.
#
# Single-table statements and join chains bind against the same kind of
# (combined) :class:`Schema`, then share these exact nodes, rules and
# evaluators; adding an expression node later means adding one class here
# rather than teaching several parallel tuple-shape branches about it.


class _Expr:
    """One bound scalar value or boolean condition."""

    __slots__ = ("type", "nullable", "children")

    # Value node (literal / column / aggregate / unary / arith / case);
    # boolean predicates set this to False.
    is_value: bool = True
    # Accepted as a HAVING comparison / IS NULL operand (leaf references).
    is_having_leaf: bool = False

    def __init__(self, type_name: str, nullable: bool, children: tuple = ()):
        self.type = type_name
        self.nullable = nullable
        self.children = children

    def columns(self) -> list:
        """Bound source-column indices referenced in this subtree, in order."""
        indices: list = []
        for child in self.children:
            indices.extend(child.columns())
        return indices

    def to_json(self) -> dict:
        raise NotImplementedError  # pragma: no cover - concrete nodes override

    def eval(self, row: tuple, agg_values: tuple | None):
        """Evaluate on one row (HAVING also passes the group's agg values)."""
        raise NotImplementedError  # pragma: no cover - concrete nodes override

    def eval_batch(self, source_columns, batch, agg_values: tuple | None) -> list:
        """Evaluate on one index batch; defaults to the row-wise semantics."""
        return [
            self.eval(_row_values(source_columns, i), agg_values) for i in batch
        ]


class _Literal(_Expr):
    """A typed literal constant."""

    __slots__ = ("value",)

    is_having_leaf = True

    def __init__(self, value, type_name: str):
        super().__init__(type_name, False)
        self.value = value

    def to_json(self) -> dict:
        return {"kind": "literal", "type": self.type, "value": self.value}

    def eval(self, row: tuple, agg_values: tuple | None):
        return self.value

    def eval_batch(self, source_columns, batch, agg_values: tuple | None) -> list:
        return [self.value] * len(batch)


class _Column(_Expr):
    """A column reference resolved to a combined-schema column index."""

    __slots__ = ("name", "index")

    is_having_leaf = True

    def __init__(self, name: str, index: int, type_name: str, nullable: bool):
        super().__init__(type_name, nullable)
        self.name = name
        self.index = index

    def columns(self) -> list:
        return [self.index]

    def to_json(self) -> dict:
        return {"kind": "column", "name": self.name}

    def eval(self, row: tuple, agg_values: tuple | None):
        return row[self.index]

    def eval_batch(self, source_columns, batch, agg_values: tuple | None) -> list:
        column = source_columns[self.index]
        return [column[i] for i in batch]


class _Aggregate(_Expr):
    """A HAVING aggregate leaf pointing at one shared aggregate slot.

    The aggregate itself is registered (and computed) with SELECT and
    ORDER BY; this node only reads the slot's value for the bound group.
    """

    __slots__ = ("func", "slot", "arg_name", "distinct_flag")

    is_having_leaf = True

    def __init__(
        self,
        func: str,
        slot: int,
        arg_name: str | None,
        distinct: bool,
        type_name: str,
        nullable: bool,
    ):
        super().__init__(type_name, nullable)
        self.func = func
        self.slot = slot
        self.arg_name = arg_name
        self.distinct_flag = distinct

    def to_json(self) -> dict:
        leaf = {
            "kind": "aggregate",
            "function": self.func,
            "argument": self.arg_name,
            "type": self.type,
            "nullable": self.nullable,
        }
        if self.distinct_flag:
            # Kept last, exactly as the historical leaf ordering.
            leaf["distinct"] = True
        return leaf

    def eval(self, row: tuple, agg_values: tuple | None):
        return agg_values[self.slot]

    def eval_batch(self, source_columns, batch, agg_values: tuple | None) -> list:
        value = agg_values[self.slot]
        return [value] * len(batch)


class _Unary(_Expr):
    """Unary ``+`` (kept) or ``-`` (type-preserving) on a numeric operand."""

    __slots__ = ("negate", "operand")

    def __init__(self, operand: _Expr, negate: bool):
        super().__init__(operand.type, operand.nullable, (operand,))
        self.negate = negate
        self.operand = operand

    def to_json(self) -> dict:
        return {
            "kind": "unary",
            "operator": "-" if self.negate else "+",
            "operands": [self.operand.to_json()],
        }

    def eval(self, row: tuple, agg_values: tuple | None):
        value = self.operand.eval(row, agg_values)
        if value is None:
            return None
        if not self.negate:  # unary plus keeps the value
            return value
        return _negate_value(value)

    def eval_batch(self, source_columns, batch, agg_values: tuple | None) -> list:
        values = self.operand.eval_batch(source_columns, batch, agg_values)
        if not self.negate:  # unary plus keeps the value
            return values
        return [None if value is None else _negate_value(value) for value in values]


class _Arith(_Expr):
    """Binary ``+`` / ``-`` / ``*`` / ``/`` with the bound result type."""

    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: _Expr, right: _Expr, type_name: str):
        super().__init__(
            type_name, left.nullable or right.nullable, (left, right)
        )
        self.op = op
        self.left = left
        self.right = right

    def to_json(self) -> dict:
        return {
            "kind": "arithmetic",
            "operator": self.op,
            "operands": [self.left.to_json(), self.right.to_json()],
        }

    def eval(self, row: tuple, agg_values: tuple | None):
        left = self.left.eval(row, agg_values)
        right = self.right.eval(row, agg_values)
        if left is None or right is None:
            return None
        return _arith_value(self.op, left, right)

    def eval_batch(self, source_columns, batch, agg_values: tuple | None) -> list:
        left = self.left.eval_batch(source_columns, batch, agg_values)
        right = self.right.eval_batch(source_columns, batch, agg_values)
        op = self.op
        return [
            None if lval is None or rval is None else _arith_value(op, lval, rval)
            for lval, rval in zip(left, right)
        ]


class _Case(_Expr):
    """A searched CASE: ordered (condition, result) branches and an ELSE."""

    __slots__ = ("branches", "else_node")

    def __init__(self, branches: tuple, else_node: _Expr | None, type_name: str):
        children = tuple(
            child for condition, result in branches for child in (condition, result)
        )
        if else_node is not None:
            children += (else_node,)
        nullable = else_node is None or any(
            result.nullable for _condition, result in branches
        ) or (else_node is not None and else_node.nullable)
        super().__init__(type_name, nullable, children)
        self.branches = branches
        self.else_node = else_node

    def to_json(self) -> dict:
        # Written order is preserved; an omitted ELSE is rendered as null,
        # which also marks the implicit result nullable in the output schema.
        return {
            "kind": "case",
            "cases": [
                {"when": condition.to_json(), "then": result.to_json()}
                for condition, result in self.branches
            ],
            "else": None if self.else_node is None else self.else_node.to_json(),
        }

    def eval(self, row: tuple, agg_values: tuple | None):
        # Conditions are tried in written order; only TRUE matches.  FALSE
        # and UNKNOWN fall through, so unhit results are never evaluated:
        # their division-by-zero, int64 overflow or non-finite float64
        # cannot raise.  The chosen result alone is evaluated.
        for condition, result in self.branches:
            if condition.eval(row, agg_values) is True:
                value = result.eval(row, agg_values)
                break
        else:
            value = (
                None
                if self.else_node is None
                else self.else_node.eval(row, agg_values)
            )
        if value is None:
            return None
        if self.type == "float64" and isinstance(value, int) and not isinstance(
            value, bool
        ):
            return float(value)
        return value

    def eval_batch(self, source_columns, batch, agg_values: tuple | None) -> list:
        # Each result expression is evaluated only on the sub-batch of rows
        # routed to it, so an error inside an unhit branch never raises;
        # rows no branch claimed fall through to ELSE (or NULL).
        result: list = [None] * len(batch)
        remaining = list(range(len(batch)))
        for condition, result_node in self.branches:
            if not remaining:
                break
            cond_values = condition.eval_batch(
                source_columns, [batch[k] for k in remaining], agg_values
            )
            taken = [k for k, cond in zip(remaining, cond_values) if cond is True]
            remaining = [
                k for k, cond in zip(remaining, cond_values) if cond is not True
            ]
            if taken:
                values = result_node.eval_batch(
                    source_columns, [batch[k] for k in taken], agg_values
                )
                for k, value in zip(taken, values):
                    result[k] = value
        if remaining and self.else_node is not None:
            values = self.else_node.eval_batch(
                source_columns, [batch[k] for k in remaining], agg_values
            )
            for k, value in zip(remaining, values):
                result[k] = value
        if self.type == "float64":
            # int64/float64 results unify to float64; ints become floats.
            return [self._unify_float(value) for value in result]
        return result

    @staticmethod
    def _unify_float(value):
        if value is None:
            return None
        # int64/float64 results unify to float64; ints become floats.
        if isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        return value


class _Predicate(_Expr):
    """Base class for the three-valued boolean conditions."""

    __slots__ = ()

    is_value = False

    def __init__(self, nullable: bool, children: tuple = ()):
        super().__init__("bool", nullable, children)


class _Cmp(_Predicate):
    """A comparison of two value nodes (UNKNOWN when either side is NULL)."""

    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: _Expr, right: _Expr):
        super().__init__(left.nullable or right.nullable, (left, right))
        self.op = op
        self.left = left
        self.right = right

    def to_json(self) -> dict:
        return {
            "kind": "comparison",
            "operator": self.op,
            "operands": [self.left.to_json(), self.right.to_json()],
        }

    def eval(self, row: tuple, agg_values: tuple | None):
        left = self.left.eval(row, agg_values)
        right = self.right.eval(row, agg_values)
        if left is None or right is None:
            return None
        return _compare_values(self.op, left, right)

    def eval_batch(self, source_columns, batch, agg_values: tuple | None) -> list:
        left = self.left.eval_batch(source_columns, batch, agg_values)
        right = self.right.eval_batch(source_columns, batch, agg_values)
        op = self.op
        return [
            None if lval is None or rval is None else _compare_values(op, lval, rval)
            for lval, rval in zip(left, right)
        ]


class _IsNull(_Predicate):
    """IS NULL / IS NOT NULL; never itself UNKNOWN."""

    __slots__ = ("operand", "negated")

    def __init__(self, operand: _Expr, negated: bool):
        super().__init__(False, (operand,))
        self.operand = operand
        self.negated = negated

    def to_json(self) -> dict:
        return {
            "kind": "is_null",
            "operator": "IS NOT NULL" if self.negated else "IS NULL",
            "operands": [self.operand.to_json()],
        }

    def eval(self, row: tuple, agg_values: tuple | None):
        value = self.operand.eval(row, agg_values)
        result = value is None
        return not result if self.negated else result

    def eval_batch(self, source_columns, batch, agg_values: tuple | None) -> list:
        values = self.operand.eval_batch(source_columns, batch, agg_values)
        if self.negated:  # IS NOT NULL
            return [value is not None for value in values]
        return [value is None for value in values]


class _Not(_Predicate):
    __slots__ = ("operand",)

    def __init__(self, operand: _Expr):
        super().__init__(operand.nullable, (operand,))
        self.operand = operand

    def to_json(self) -> dict:
        return {
            "kind": "not",
            "operator": "NOT",
            "operands": [self.operand.to_json()],
        }

    def eval(self, row: tuple, agg_values: tuple | None):
        value = self.operand.eval(row, agg_values)
        return None if value is None else (not value)

    def eval_batch(self, source_columns, batch, agg_values: tuple | None) -> list:
        values = self.operand.eval_batch(source_columns, batch, agg_values)
        return [None if value is None else (not value) for value in values]


class _Logic(_Predicate):
    """AND / OR; UNKNOWN propagates through either side."""

    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left: _Expr, right: _Expr):
        super().__init__(left.nullable or right.nullable, (left, right))
        self.op = op
        self.left = left
        self.right = right

    def to_json(self) -> dict:
        return {
            "kind": "logic",
            "operator": self.op.upper(),
            "operands": [self.left.to_json(), self.right.to_json()],
        }

    def eval(self, row: tuple, agg_values: tuple | None):
        # Short-circuit SQL semantics; UNKNOWN propagates only when needed.
        left = self.left.eval(row, agg_values)
        if self.op == "and":
            if left is False:
                return False
            right = self.right.eval(row, agg_values)
            if right is False:
                return False
            if left is None or right is None:
                return None
            return True
        if left is True:
            return True
        right = self.right.eval(row, agg_values)
        if right is True:
            return True
        if left is None or right is None:
            return None
        return False

    def eval_batch(self, source_columns, batch, agg_values: tuple | None) -> list:
        left = self.left.eval_batch(source_columns, batch, agg_values)
        # Rows the left operand already decided (FALSE for AND, TRUE for
        # OR) never evaluate the right operand, exactly as row-wise
        # short-circuiting; the rest form the sub-batch evaluated next.
        if self.op == "and":
            undecided = [k for k, value in enumerate(left) if value is not False]
        else:
            undecided = [k for k, value in enumerate(left) if value is not True]
        result = list(left)
        if undecided:
            right = self.right.eval_batch(
                source_columns,
                [batch[k] for k in undecided],
                agg_values,
            )
            for k, rval in zip(undecided, right):
                lval = left[k]
                if self.op == "and":
                    result[k] = (
                        False
                        if rval is False
                        else (None if lval is None or rval is None else True)
                    )
                else:
                    result[k] = (
                        True
                        if rval is True
                        else (None if lval is None or rval is None else False)
                    )
        return result


# ---------------------------------------------------------------------------
# Three-valued-logic evaluation
# ---------------------------------------------------------------------------
#
# The per-row semantics live on the bound expression nodes themselves
# (:meth:`_Expr.eval`); a batch failure falls back to that same row-at-a-
# time method (see :func:`_eval_batch`) so the reported first error keeps
# its historical row order.  The batched semantics
# (:meth:`_Expr.eval_batch`) visit exactly the same
# (row, subexpression) pairs.


def _compare_values(op: str, left, right) -> bool:
    """The boolean result of one comparison on two non-NULL values."""
    if op == "=":
        result = left == right
    elif op == "!=":
        result = left != right
    elif op == "<":
        result = left < right
    elif op == "<=":
        result = left <= right
    elif op == ">":
        result = left > right
    else:  # ">="
        result = left >= right
    return bool(result)


def _negate_value(value):
    # Unary minus: the type is preserved; only int64 can overflow.
    if isinstance(value, int):
        if value == _INT64_MIN:
            raise QueryValidationError(
                "negating the int64 minimum overflows the int64 range"
            )
        return -value
    return -value


def _arith_value(op: str, left, right):
    # int64 op int64 stays int64 for + - *; any float64 operand or a
    # division produces float64.  NULLs are handled by the caller.
    if op == "/" or isinstance(left, float) or isinstance(right, float):
        if op == "/" and right == 0:
            raise QueryValidationError("division by zero")
        a = float(left)
        b = float(right)
        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        elif op == "*":
            result = a * b
        else:
            result = a / b
        if not math.isfinite(result):
            raise QueryValidationError(
                "float64 arithmetic produced a non-finite value"
            )
        return result
    if op == "+":
        result = left + right
    elif op == "-":
        result = left - right
    else:
        result = left * right
    if not (_INT64_MIN <= result <= _INT64_MAX):
        raise QueryValidationError("int64 arithmetic overflowed the int64 range")
    return result


# ---------------------------------------------------------------------------
# Batched column-vector evaluation
# ---------------------------------------------------------------------------
#
# The row-processing stages -- the WHERE filter, the ORDER BY sort keys,
# the SELECT expressions and the projection feeding SELECT DISTINCT -- share
# one batched evaluator.  Each stage hands the next a set of valid row
# indices; an expression is evaluated one *batch* of rows at a time against
# the source columns and yields one vector of result values aligned with
# the batch, instead of re-entering the row-at-a-time control flow for
# every row.  The vectorised walk is the node's own ``eval_batch``; its
# semantics are exactly those of :meth:`_Expr.eval` (NULL propagation,
# three-valued logic, per-row CASE short-circuit, the int64 / float64
# error rules).  Because the batched walk visits the same
# (row, subexpression) pairs as the row-wise walk, any batch that contains
# a failing row raises; the driver then re-runs that batch row by row so
# the reported error is byte-identical to the historical one.

_EVAL_BATCH_SIZE = 2048


def _iter_batches(indices, size=_EVAL_BATCH_SIZE):
    """Yield ``indices`` sliced into consecutive batches of at most ``size``."""
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


def _eval_batch(node: _Expr, source_columns, batch, agg_values: tuple | None = None) -> list:
    """Evaluate one batch of a bound expression, returning a value vector.

    ``batch`` is the current valid-row set (row indices into
    ``source_columns``); the returned list holds one value per batch row, in
    batch order.  A :class:`QueryValidationError` anywhere in the batch is
    re-raised through the row-wise reference path so the exact historical
    error (and its row order) surfaces.
    """
    try:
        return node.eval_batch(source_columns, batch, agg_values)
    except QueryValidationError:
        # The batched walk visits the same (row, subexpression) pairs as
        # the row-wise one, so this re-run raises the same first error the
        # pre-vectorised engine reported for these rows.
        return [node.eval(_row_values(source_columns, i), agg_values) for i in batch]


def _eval_expr_selection(node, source_columns, selected) -> list:
    """Evaluate ``node`` over the row indices ``selected``, batch by batch.

    Returns one list of values aligned with ``selected``; shared by the
    WHERE filter, the ORDER BY sort keys and the SELECT projections.
    """
    values: list = []
    for batch in _iter_batches(selected):
        values.extend(_eval_batch(node, source_columns, batch))
    return values


def _filter_rows(where: _Expr, source_columns, row_count: int) -> list[int]:
    """The row indices whose WHERE condition evaluates to TRUE.

    Batched counterpart of the historical row loop: rows are evaluated
    batch by batch in file order, and FALSE and UNKNOWN both drop the row.
    """
    selected: list[int] = []
    for batch in _iter_batches(list(range(row_count))):
        mask = _eval_batch(where, source_columns, batch)
        selected.extend(i for i, keep in zip(batch, mask) if keep is True)
    return selected


def _row_values(source_columns: tuple[tuple, ...], i: int) -> tuple:
    return tuple(col[i] for col in source_columns)

