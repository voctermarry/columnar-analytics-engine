"""Single-file SQL query layer.

Public API:

* :func:`query_file` -- run a ``SELECT ... FROM input [WHERE ...]`` query
  against one columnar file and return a :class:`~columnar_analytics.format.Table`
* :func:`query_files` -- run a statement against a table-name to path
  mapping, optionally chaining several equi-joins
* :func:`explain_file` / :func:`explain_files` -- parse, bind and plan the
  same statements without executing them; only file metadata is read
* :class:`QuerySyntaxError` -- lexical / grammatical errors
* :class:`QueryValidationError` -- unknown columns, wrong table name,
  type-incompatible comparisons, invalid aggregate use

The accepted grammar (keywords case-insensitive)::

    query       := SELECT [DISTINCT] select_item (',' select_item)*
                   FROM ident [WHERE expr] [GROUP BY ident (',' ident)*]
                   [HAVING expr]
                   [ORDER BY order_item (',' order_item)*]
                   [LIMIT uint]
    select_item := '*' | ident | agg_name '(' [DISTINCT] ('*' | ident) ')'
                   | scalar_expr AS alias
    order_item  := (ident | agg_name '(' [DISTINCT] ('*' | ident) ')')
                   [ASC | DESC] [NULLS FIRST | NULLS LAST]
    expr       := or_expr
    or_expr     := and_expr (OR and_expr)*
    and_expr    := cmp_expr (AND cmp_expr)*
    cmp_expr    := not_factor (cmp_op not_factor)?
    not_factor  := NOT not_factor | postfix
    postfix     := scalar_expr (IS [NOT] NULL)?
    scalar_expr := term (('+' | '-') term)*
    term        := factor (('*' | '/') factor)*
    factor      := ('+' | '-') factor | atom
    atom        := '(' expr ')' | operand
                | CASE WHEN expr THEN scalar_expr
                  (WHEN expr THEN scalar_expr)*
                  (ELSE scalar_expr)? END
    operand     := ident | literal
    literal     := TRUE | FALSE | [+-]? int64 | [+-]? finite float64
                   | single-quoted utf8

The supported aggregates are ``COUNT(*)``, ``COUNT(ident)`` and
``SUM`` / ``AVG`` / ``MIN`` / ``MAX`` applied to one column.  Each of
``COUNT`` / ``SUM`` / ``AVG`` / ``MIN`` / ``MAX`` also accepts
``DISTINCT`` before its single column argument (``COUNT(DISTINCT ident)``
etc.): rows are first selected by WHERE, then within each group (or the
single global group) NULL arguments are ignored and the remaining values
are deduplicated by the column's type -- float64 ``0.0`` and ``-0.0``
count as the same value -- before the aggregate runs.  ``COUNT(DISTINCT
x)`` returns the number of distinct non-NULL values (0 on empty input,
a non-nullable int64); the other DISTINCT aggregates return NULL when no
distinct non-NULL value remains and otherwise keep the result type,
nullability, comparison rules and overflow / non-finite
:class:`QueryValidationError` behaviour of their plain counterparts.
DISTINCT here only deduplicates the aggregate's argument values; it does
not change ``SELECT DISTINCT`` row deduplication, and the argument stays
a single column reference (no ``*``, expressions, multiple arguments or
nested aggregates).  A misplaced DISTINCT keyword, a missing argument,
``*``, an expression or multiple arguments inside a DISTINCT call raise
:class:`QuerySyntaxError` before any file is opened.  DISTINCT
aggregates may appear wherever plain aggregates may (SELECT, HAVING,
ORDER BY), and textually identical calls share one computation.  Without
GROUP BY the projection may contain aggregate expressions only; with
GROUP BY it may additionally contain the grouped columns.  A bare ``*``
must not be mixed with aggregates or grouping.  Aggregates are not
allowed inside WHERE, may not be nested, and aliases are not supported
for bare columns or aggregate calls.

Scalar arithmetic (parentheses, unary ``+``/``-``, binary ``+`` ``-``
``*`` ``/``; precedence: parentheses, unary, ``*``/``/``, ``+``/``-``)
is available over int64 / float64 columns and numeric literals in SELECT
and WHERE.  A computed SELECT expression must be named with ``AS`` (the
alias follows the usual identifier rules and must be unique among the
output names); bare columns and aggregate calls keep their existing
result names.  int64 ``+``/``-``/``*`` on two int64 operands yields
int64; any float64 operand or a division yields float64, and unary
operators preserve the operand type.  Any NULL operand makes the result
NULL; the output column is nullable iff a participating column is
nullable (pure constants are not).  int64 overflow, division by zero and
non-finite float64 results raise :class:`QueryValidationError`.  A
numeric expression used directly as a boolean condition is rejected with
:class:`QueryValidationError` as well.  Non-aggregate queries execute
WHERE, then compute the sort keys, stably sort, apply LIMIT and only
then evaluate the SELECT expressions, so filtered or truncated rows
never trigger division-by-zero or overflow in the projection; a
DISTINCT query instead evaluates the projection right after WHERE,
deduplicates the result rows, and only then sorts and applies LIMIT.  ORDER BY
may name a SELECT alias (an alias shadows an input column of the same
name) and sorts on the expression's result type with the usual NULL
placement and stability rules.  Aggregate queries do not accept scalar
expressions: GROUP BY keys and aggregate arguments stay plain column
references, and mixing a scalar expression into an aggregate query
raises :class:`QueryValidationError`.

``SELECT DISTINCT`` removes duplicate rows from a non-aggregate query:
the projection may be a star, plain columns or ``AS``-aliased scalar
expressions exactly as without DISTINCT, and deduplication compares the
full projected result row column by column after WHERE.  Two NULLs in
the same result column compare equal, NULL never equals a non-NULL
value, bool / utf8 / numeric values keep their existing comparison
semantics, and float64 ``0.0`` and ``-0.0`` compare equal.  Without
ORDER BY the surviving rows keep the order of the first occurrence of
each distinct result row; ORDER BY then sorts the deduplicated rows and
LIMIT keeps the first N.  The projection expressions are evaluated
only for rows that passed WHERE (and, with ORDER BY, before the sort),
so a division by zero, int64 overflow or non-finite float64 result that
actually occurs still raises :class:`QueryValidationError`.  ORDER BY
of a DISTINCT query may name only bare columns or explicit aliases that
are part of the projection; an unprojected or otherwise unknown name
raises :class:`QueryValidationError`, and the sort direction, NULL
placement and stability rules are unchanged.  DISTINCT combined with
GROUP BY, HAVING or any aggregate projection raises
:class:`QueryValidationError`; argument-level ``DISTINCT`` inside an
aggregate call (``COUNT(DISTINCT ...)`` etc.) is a separate feature
described above and stays an aggregate query.  A missing projection, a
repeated DISTINCT or a misplaced DISTINCT keyword raises
:class:`QuerySyntaxError` before any file is opened.

The row-processing stages (WHERE, the ORDER BY sort keys, the SELECT
expressions and the projection feeding SELECT DISTINCT) share one batched
column-vector evaluator: each stage hands the next a set of valid row
indices, and an expression is evaluated one batch of column values at a
time instead of row by row.  The batched semantics are exactly the
per-row semantics described above -- three-valued logic, NULL propagation,
per-row CASE short-circuit, the int64 / float64 promotion and error rules
and the stage order (so filtered or LIMIT-cut rows never evaluate the
expressions of a later stage) are unchanged.  Joins, grouping, aggregation
and HAVING keep their existing result organisation; the data they hand to
the expression stages follows the same batched semantics.

Searched CASE expressions
(``CASE WHEN cond THEN result [WHEN ...] [ELSE result] END``; the simple
``CASE value WHEN ...`` form is not supported and at least one WHEN is
required) are scalar operands and may be nested inside more arithmetic,
comparisons and other CASE expressions.  For each row the WHEN conditions
are tested in written order and only TRUE matches; FALSE and UNKNOWN fall
through, an explicit ELSE supplies the fallback and a missing ELSE yields
NULL.  Only the chosen result is evaluated, so division by zero, int64
overflow or a non-finite float64 inside an unhit branch never raises; a
chosen result that errors still raises :class:`QueryValidationError`.
Every reachable THEN and an explicit ELSE must share one type family:
int64 and float64 may mix and unify to float64, while bool and utf8 must
match each other exactly (any other mix raises
:class:`QueryValidationError`).  The result column is nullable if any
reachable result is nullable or the ELSE is omitted.  WHEN conditions use
the same boolean type check and three-valued logic as WHERE; CASE stays a
scalar expression, so it is rejected in aggregate arguments, GROUP BY and
the projection of an aggregate query just like the other scalar
expressions.

Operator precedence (highest first) is NOT, comparison, AND, OR; IS [NOT]
NULL is a postfix of its scalar operand.  Comparison operands may be
column references, literals or scalar arithmetic expressions.  WHERE
follows SQL three-valued logic: a normal
comparison against NULL yields UNKNOWN, UNKNOWN propagates through the
logical operators, and only TRUE rows are returned.

After WHERE, rows are grouped in GROUP BY column order (a NULL key forms
its own group); without an explicit ORDER BY groups come out in the order
of their first selected row.  An optional HAVING clause between GROUP BY
and ORDER BY filters the formed groups with the same boolean expression
grammar and three-valued logic as WHERE: its conditions combine grouping
columns, aggregate calls and type-compatible literals with NOT / AND / OR
/ comparisons / IS [NOT] NULL, only groups whose condition is TRUE are
kept, and aggregates named in HAVING need not appear in SELECT.  Without
GROUP BY HAVING filters the single global aggregate row (still produced
when WHERE selected no rows); with GROUP BY an empty selection yields
zero groups.  HAVING may not name ungrouped plain columns, ``*``, CASE,
scalar arithmetic or nested aggregates; it is rejected on a completely
non-aggregate query.  ORDER BY may only name a selected result; ASC is the
default, NULLs sort last regardless of direction unless NULLS FIRST /
NULLS LAST is given, and groups equal on every sort key keep that
first-row order.  LIMIT then keeps the first N groups.  Without GROUP BY
and aggregates an aggregate query over zero selected rows still yields one
output row (COUNT 0, the other aggregates NULL), unless HAVING removes it;
with GROUP BY it yields zero rows.

Multi-file queries (:func:`query_files`) add a deterministic chain of zero
or more equi-joins to the grammar::

    query := SELECT ... FROM from_table
             { join_kind JOIN next_table
               ON (earlier_table.col = next_table.col
                   | next_table.col = earlier_table.col) } ...
    join_kind := INNER | LEFT | RIGHT | FULL OUTER

``sources`` maps table names to file paths; only the tables referenced by
the statement are read.  Each join step introduces exactly one not-yet-used
table (aliases and repeated tables are rejected) and its single ON equality
connects a qualified column of that new table with a qualified column of any
table introduced earlier (FROM or a previous join); the two sides of the
equality are interchangeable.  Compound or non-equality ON conditions,
other join types and ``FULL`` without ``OUTER`` are syntax errors raised
before any file is opened.  In any statement that contains a JOIN every
column reference outside ``COUNT(*)`` must be qualified as ``table.column``
(either part may be a double-quoted identifier).  ``SELECT *`` emits the
referenced schemas in FROM/JOIN order with ``table.column`` names; explicit
projections keep their qualified names and aggregates keep the uppercase
``FUNC(table.column)`` labels.

Each step is evaluated with the materialised intermediate result as its
left input and the newly read table as its right input.  Join keys may
share a type or mix int64 with float64; NULL keys never match.  INNER JOIN
emits every matching combination, LEFT JOIN also emits unmatched left rows
with the right-side values set to NULL (the new table's result columns are
nullable), RIGHT JOIN emits all new-table rows in new-file order (one new
row's matches expand in intermediate/current row order) padding the whole
intermediate side with NULL (every earlier table's result columns become
nullable), and FULL OUTER JOIN emits the LEFT JOIN output and then appends
the unmatched new rows in new-file order (every result column is nullable).
INNER / LEFT / FULL expand in the current intermediate row order, with the
matching new rows in new-file order within one current row; RIGHT expands
in new-file row order with the matches in current row order.  Nullability
is derived per step: once a step makes an earlier table's columns
nullable, later steps keep them nullable.  WHERE / GROUP BY / HAVING /
projection / DISTINCT / stable sorting / LIMIT run only after the whole
chain has been built, with their usual semantics.

The multi-file entry points (:func:`query_files`, :func:`explain_files`)
take an optional ``join_strategy`` of ``"hash"`` or ``"sort_merge"``; an
explicit strategy is applied to every join step.  Both strategies accept
the same statements and return identical column descriptions, values, NULL
placement and row order; they differ only in how matching right rows are
located (a right-key hash index versus sorting both sides on the key and
merging equal-key runs).  With the argument omitted the historical
default join path is used and the explain plan carries no strategy field;
with an explicit strategy every plan Join operator gains a ``strategy``
field of ``HASH`` or ``SORT_MERGE``.  A JOIN-less statement accepts either
strategy but gains no extra operator, and an invalid strategy raises
:class:`ValueError` before any source file is opened.

The algorithm selection is decoupled from everything described here:
:mod:`columnar_analytics.join` owns the strategy registry and validation,
the per-algorithm match lookup, key semantics, match expansion, outer NULL
padding, deterministic row order and post-step nullability.  Each algorithm
only produces a match relation; the join layer assembles the result and
this module keeps parsing, qualified-name binding, key/type validation,
grouping, sorting, plan description and export.  Adding an algorithm is a
registry entry there -- no part of the query, explain or export path is
copied or changed.

The explain plan lists the referenced sources and their Scan operators in
FROM/JOIN order with ``required_columns`` attributed to each source, then
one Join operator per step in the same order, recording the step type and
the two normalised qualified keys (the earlier table on the left, the
freshly introduced table on the right).

Row-group-partitioned (v2) files are read selectively: every query decodes
only the column blocks its plan actually references (a join chain
additionally reads each table's ON keys).  Single-file statements
(:func:`query_file` and JOIN-less :func:`query_files`) and multi-file
statements whose whole join chain is INNER push the AND-connected,
type-compatible comparisons and IS [NOT] NULL conditions of WHERE down to
the per-group statistics.  In a join chain only a top-level AND leaf that
references exactly one source (a qualified column of that source compared
to a type-compatible literal, on either side, or that column's IS [NOT]
NULL) is attributed to that source; each source's own leaves are combined
with AND independently.  A chain containing any LEFT / RIGHT / FULL OUTER
step performs no row-group pruning at all (an OUTER step can pad a source
with NULLs, so no group can be proven absent), v1 sources are always read
in full, and in a mixed v1/v2 all-INNER chain only the v2 sources are
pruned.

A row group is skipped only when a source's statistics prove its pushed
leaves cannot all be TRUE at once for any of the group's rows; OR, NOT,
CASE, arithmetic, column-to-column or cross-source comparisons and
undecidable ranges are never pushed, and neither block nor replace the
eligible leaves next to them.  Surviving rows are still filtered row by
row with the full WHERE condition over the complete join result, so NULL
three-valued logic, duplicate key combinations, aggregates, DISTINCT,
ordering, LIMIT and the deterministic unordered row order are unchanged.
When every group is excluded the statement simply sees zero rows, so
empty results, ``COUNT(*)``, other global aggregates and HAVING keep
their usual semantics.  Both format versions return identical column
descriptions, values, row order, sorting, LIMIT, join and export results
for the same data and statement.  Only the blocks of the columns and row
groups actually selected are decompressed or decoded: a block belonging
to an excluded group is never read even when corrupt, while a corrupt
block of a selected group raises
:class:`~columnar_analytics.format.ColumnarFormatError`.

In the explain plan a v2 Scan operator of an eligible statement
(single-source, or an all-INNER chain) carries, right after
``required_columns``, the fields ``row_groups_total``,
``row_groups_selected`` and ``pushed_condition`` (the leaves attributed
to that source combined, in SQL appearance order, as one condition tree
with qualified column names, or null when no leaf was pushed; with no
pushed leaf selected equals total).  v1 Scan operators and the Scan
operators of a chain containing an OUTER step keep their historical
shape; the reported counts always match the groups execution reads.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any

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
from .join import (
    _derive_step_schema,
    _execute_join,
    _strategy_label,
    _validate_join_strategy,
)

__all__ = [
    "QuerySyntaxError",
    "QueryValidationError",
    "explain_file",
    "explain_files",
    "query_file",
    "query_files",
]

_KEYWORDS = frozenset(
    (
        "select",
        "distinct",
        "from",
        "where",
        "as",
        "not",
        "and",
        "or",
        "is",
        "null",
        "true",
        "false",
        "order",
        "by",
        "asc",
        "desc",
        "nulls",
        "first",
        "last",
        "limit",
        "group",
        "having",
        "inner",
        "left",
        "join",
        "on",
        "case",
        "when",
        "then",
        "else",
        "end",
    )
)

# Aggregate function names are ordinary (case-insensitive) identifiers that
# gain call syntax only in the projection and ORDER BY.
_AGG_NAMES = frozenset(("count", "sum", "avg", "min", "max"))
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

# Optional join strategies and their validation live in the join execution
# layer (:mod:`columnar_analytics.join`), together with the algorithm
# registry and explain-plan labels; ``_strategy_label`` /
# ``_validate_join_strategy`` are imported above and reused by the query and
# explain entry points.


class QuerySyntaxError(Exception):
    """Raised when a query is lexically or grammatically invalid."""


class QueryValidationError(Exception):
    """Raised when a syntactically valid query is incompatible with the schema."""


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Token:
    kind: str  # one of: keyword, ident, number, string, op, star, eof
    value: Any
    text: str


_NUMBER_RE = re.compile(
    r"""
    \d+(?:\.\d*)?(?:[eE][+-]?\d+)?   # 1, 1., 1.5, 1e3, 1.5e-2
    | \.\d+(?:[eE][+-]?\d+)?         # .5
    """,
    re.VERBOSE,
)
# Bare identifiers: Unicode letters/underscore first, then word characters.
_IDENT_RE = re.compile(r"[^\W\d]\w*", re.UNICODE)


def _tokenize(sql: str) -> list[_Token]:
    if not isinstance(sql, str):
        raise QuerySyntaxError("SQL statement must be a string")
    tokens: list[_Token] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "*":
            tokens.append(_Token("star", None, ch))
            i += 1
            continue
        if ch in ",()+-/":
            tokens.append(_Token("op", ch, ch))
            i += 1
            continue
        if ch in "=<>!":
            if ch == "!" and i + 1 < n and sql[i + 1] == "=":
                tokens.append(_Token("op", "!=", "!="))
                i += 2
                continue
            if ch in "<>" and i + 1 < n and sql[i + 1] == "=":
                op = ch + "="
                tokens.append(_Token("op", op, op))
                i += 2
                continue
            if ch in "=<>":
                tokens.append(_Token("op", ch, ch))
                i += 1
                continue
            raise QuerySyntaxError(f"unexpected character '!' at position {i}")
        if ch == "'":
            value, new_i = _tokenize_string(sql, i)
            tokens.append(_Token("string", value, sql[i:new_i]))
            i = new_i
            continue
        if ch == '"':
            value, new_i = _tokenize_quoted_ident(sql, i)
            tokens.append(_token_from_word(value, quoted=True))
            i = new_i
            continue
        if ch.isdigit() or ch == ".":
            match = _NUMBER_RE.match(sql, i)
            if match:
                text = match.group(0)
                tokens.append(_Token("number", _parse_number(text), text))
                i = match.end()
                continue
            # A '.' that does not start a number is the qualifier separator
            # used by multi-table queries ("table.column").
            tokens.append(_Token("op", ".", ch))
            i += 1
            continue
        match = _IDENT_RE.match(sql, i)
        if match:
            text = match.group(0)
            tokens.append(_token_from_word(text, quoted=False))
            i = match.end()
            continue
        raise QuerySyntaxError(f"unexpected character {ch!r} at position {i}")
    tokens.append(_Token("eof", None, ""))
    return tokens


def _token_from_word(word: str, *, quoted: bool) -> _Token:
    lowered = word.lower()
    if not quoted and lowered in _KEYWORDS:
        if lowered == "true":
            return _Token("number", True, word)
        if lowered == "false":
            return _Token("number", False, word)
        return _Token("keyword", lowered, word)
    return _Token("qident" if quoted else "ident", word, word)


def _tokenize_string(sql: str, start: int) -> tuple[str, int]:
    parts: list[str] = []
    i = start + 1
    n = len(sql)
    while True:
        j = sql.find("'", i)
        if j == -1:
            raise QuerySyntaxError("unterminated string literal")
        parts.append(sql[i:j])
        if j + 1 < n and sql[j + 1] == "'":
            parts.append("'")
            i = j + 2
            continue
        value = "".join(parts)
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise QuerySyntaxError("string literal is not valid UTF-8") from None
        return value, j + 1


def _tokenize_quoted_ident(sql: str, start: int) -> tuple[str, int]:
    parts: list[str] = []
    i = start + 1
    n = len(sql)
    while True:
        j = sql.find('"', i)
        if j == -1:
            raise QuerySyntaxError("unterminated quoted identifier")
        parts.append(sql[i:j])
        if j + 1 < n and sql[j + 1] == '"':
            parts.append('"')
            i = j + 2
            continue
        value = "".join(parts)
        if not value:
            raise QuerySyntaxError("quoted identifier must not be empty")
        return value, j + 1


def _parse_number(text: str) -> int | float:
    try:
        if any(ch in text for ch in ".eE"):
            value: int | float = float(text)
        else:
            value = int(text, 10)
    except ValueError:
        raise QuerySyntaxError(f"invalid numeric literal {text!r}") from None
    if isinstance(value, int):
        # The magnitude 2**63 is accepted unsigned; the signed range check
        # happens after the optional leading sign is applied.
        if not (0 <= value <= 2**63):
            raise QuerySyntaxError(f"integer literal {text} is outside the int64 range")
        return value
    if not math.isfinite(value):
        raise QuerySyntaxError(f"float literal {text} must be finite")
    return value


# ---------------------------------------------------------------------------
# Parsed expression tree
#
# The parser emits plain tuples; qualifiers are resolved textually by the
# rewrite pass before binding.  Node tuples (pre-binding shape):
#   ("literal", python_value, type_name)
#   ("column", name, table|None, table_quoted)
#   ("arith", op, left_node, right_node)              -- + - * /
#   ("unary", operand, negate)
#   ("case", ((cond, result), ...), else_node|None)
#   ("cmp", op, left_node, right_node)
#   ("isnull", operand, negate)
#   ("not", operand)
#   ("and"|"or", left, right)
# The HAVING grammar additionally produces its own aggregate leaf:
#   ("hagg", func_upper, arg_name|"" , arg_table|None, arg_table_quoted, distinct)
# Predicate nodes are boolean-typed (three-valued at evaluation time);
# "literal"/"column"/"arith"/"unary"/"case" nodes are value nodes.
#
# Binding turns those tuples into the single bound expression IR defined
# below (the :class:`_Expr` hierarchy).  Every later stage -- type and
# nullability derivation, dependency collection, plan rendering, predicate
# pushdown, the row-wise and the batched evaluator -- walks one bound node
# through the node's own declared fields, so a single bound expression is
# the consistent source of truth in single-table and join queries alike.
#
# Projection / ORDER BY reference items:
#   ("column_ref", name)                 -- a plain column name
#   ("agg", func_upper, arg_name|None)   -- an aggregate call; arg None = '*'
#   ("expr", alias)                      -- a computed scalar expression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RefItem:
    """One projection element or ORDER BY key as parsed."""

    kind: str  # "star" | "column" | "agg" | "expr"
    name: str | None = None  # column name, or the AS alias for kind == "expr"
    func: str | None = None  # uppercase function name for kind == "agg"
    arg: str | None = None  # aggregate column argument; "" stands for '*'
    distinct: bool = False  # kind == "agg": DISTINCT before the argument
    expr: tuple | None = None  # parsed scalar expression for kind == "expr"
    descending: bool = False
    nulls_first: bool | None = None  # None -> default (NULLs last)
    # Table qualifiers (multi-table queries only; None when unqualified):
    table: str | None = None
    table_quoted: bool = False
    arg_table: str | None = None
    arg_table_quoted: bool = False


@dataclass(frozen=True)
class _Join:
    """One equi-join step of a multi-table query.

    A statement may chain several steps in FROM/JOIN order; each step takes
    the intermediate result built so far as its left input and one freshly
    introduced table as its right input.
    """

    kind: str  # "inner" | "left" | "right" | "full"
    table: str  # right table name as spelled
    table_quoted: bool
    left_key: tuple  # (table, table_quoted, column)
    right_key: tuple  # (table, table_quoted, column)


@dataclass
class _Select:
    items: tuple[_RefItem, ...]  # projection; may be a single ("star",) item
    table: str
    table_quoted: bool
    where: tuple | None
    # GROUP BY entries are (table|None, table_quoted, name) triples.
    group_by: tuple[tuple, ...] | None
    # Post-grouping boolean expression (aggregate queries only); None absent.
    having: tuple | None
    order_by: tuple[_RefItem, ...] | None
    limit: int | None
    # Join steps in FROM/JOIN order; empty for a single-table statement.
    joins: tuple[_Join, ...] = ()
    # SELECT DISTINCT: deduplicate the projected rows (non-aggregate only).
    distinct: bool = False

    @property
    def star(self) -> bool:
        return len(self.items) == 1 and self.items[0].kind == "star"

    @property
    def has_aggregate(self) -> bool:
        return any(item.kind == "agg" for item in self.items)

    @property
    def is_aggregate_query(self) -> bool:
        return self.has_aggregate or self.group_by is not None or self.having is not None


class _Parser:
    _CMP_OPS = frozenset(("=", "!=", "<", "<=", ">", ">="))

    def __init__(self, tokens: list[_Token], allow_join: bool = False):
        self.tokens = tokens
        self.pos = 0
        # Multi-table statements (qualified names + join clauses) are only
        # enabled for query_files; single-file entry points keep rejecting
        # them as syntax errors.
        self._allow_join = allow_join

    def parse(self) -> _Select:
        select = self._parse_select()
        if self._peek().kind != "eof":
            tok = self._peek()
            raise QuerySyntaxError(f"unexpected trailing input {tok.text!r}")
        return select

    def _parse_select(self) -> _Select:
        self._expect_keyword("select")
        distinct = self._accept_keyword("distinct")
        items = tuple(self._parse_projection())
        self._expect_keyword("from")
        table_tok = self._expect_table_name()
        joins: tuple[_Join, ...] = ()
        if self._allow_join:
            joins = tuple(self._parse_join_clauses())
        where = None
        if self._accept_keyword("where"):
            where = self._parse_or()
        group_by = None
        if self._accept_keyword("group"):
            self._expect_keyword("by")
            names = [self._parse_group_name()]
            while self._accept_op(","):
                names.append(self._parse_group_name())
            group_by = tuple(names)
        having = None
        if self._accept_keyword("having"):
            having = self._parse_having_or()
        order_by = None
        if self._accept_keyword("order"):
            self._expect_keyword("by")
            order_items = [self._parse_order_item()]
            while self._accept_op(","):
                order_items.append(self._parse_order_item())
            order_by = tuple(order_items)
        limit = None
        if self._accept_keyword("limit"):
            limit = self._parse_limit()
        return _Select(
            items=items,
            table=table_tok.value,
            table_quoted=table_tok.kind == "qident",
            where=where,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=limit,
            joins=joins,
            distinct=distinct,
        )

    def _parse_join_clauses(self) -> list[_Join]:
        # Zero or more join clauses may follow FROM; each introduces exactly
        # one new table with a single equi-ON condition.  "right" / "full" /
        # "outer" stay ordinary identifiers away from this clause (they are
        # matched with _accept_word), so columns carrying those names keep
        # working outside join parsing.
        joins: list[_Join] = []
        while True:
            if self._accept_keyword("inner"):
                self._expect_keyword("join")
                joins.append(self._parse_join("inner"))
            elif self._accept_keyword("left"):
                self._expect_keyword("join")
                joins.append(self._parse_join("left"))
            elif self._accept_word("right"):
                self._expect_keyword("join")
                joins.append(self._parse_join("right"))
            elif self._accept_word("full"):
                # Only the explicit FULL OUTER JOIN spelling is accepted.
                self._expect_word("outer")
                self._expect_keyword("join")
                joins.append(self._parse_join("full"))
            else:
                return joins

    def _parse_join(self, kind: str) -> _Join:
        table_tok = self._expect_table_name()
        self._expect_keyword("on")
        left_key = self._parse_on_key()
        self._expect_op("=")
        right_key = self._parse_on_key()
        return _Join(
            kind=kind,
            table=table_tok.value,
            table_quoted=table_tok.kind == "qident",
            left_key=left_key,
            right_key=right_key,
        )

    def _parse_on_key(self) -> tuple:
        tok = self._peek()
        if tok.kind not in ("ident", "qident"):
            raise QuerySyntaxError(
                f"ON requires table-qualified columns, got {tok.text!r}"
            )
        self._next()
        if not self._accept_op("."):
            raise QuerySyntaxError(
                "ON requires table-qualified columns (table.column)"
            )
        nxt = self._peek()
        if nxt.kind not in ("ident", "qident"):
            raise QuerySyntaxError(
                f"expected column name after '.' in ON, got {nxt.text!r}"
            )
        self._next()
        return (tok.value, tok.kind == "qident", nxt.value)

    def _parse_projection(self) -> list[_RefItem]:
        items = [self._parse_select_item()]
        while self._accept_op(","):
            items.append(self._parse_select_item())
        # Aliases are only supported for computed scalar expressions (which
        # consume their own "AS alias" in _parse_select_item); a trailing
        # identifier (including the spelling "as") where FROM is expected
        # names one on a bare column or aggregate.  Plain queries keep their
        # historical QuerySyntaxError classification; aggregate queries treat
        # the unsupported alias as a validation error.
        tok = self._peek()
        if tok.kind in ("ident", "qident") or (
            tok.kind == "keyword" and tok.value == "as"
        ):
            grouped_ahead = any(
                t.kind == "keyword" and t.value == "group"
                for t in self.tokens[self.pos :]
            )
            if any(item.kind == "agg" for item in items) or grouped_ahead:
                raise QueryValidationError(
                    f"aliases are not supported; unexpected {tok.text!r} after projection item"
                )
            raise QuerySyntaxError(f"expected FROM, got {tok.text!r}")
        return items

    def _parse_select_item(self) -> _RefItem:
        tok = self._peek()
        if tok.kind == "star":
            self._next()
            return _RefItem("star")
        if (
            tok.kind == "ident"
            and tok.value.lower() in _AGG_NAMES
            and self.tokens[self.pos + 1].kind == "op"
            and self.tokens[self.pos + 1].value == "("
        ):
            item = self._parse_column_or_agg()
            if self._arith_op_ahead():
                raise QueryValidationError(
                    "aggregate calls cannot be combined with scalar arithmetic"
                )
            return item
        node = self._parse_arith()
        if node[0] == "column":
            # A bare (possibly parenthesised) column reference keeps its
            # existing projection behaviour and needs no alias.
            return _RefItem(
                "column", name=node[1], table=node[2], table_quoted=node[3]
            )
        # A computed scalar expression must be named with AS.
        self._expect_keyword("as")
        alias = self._peek()
        if alias.kind not in ("ident", "qident"):
            raise QuerySyntaxError(
                f"AS requires a result name, got {alias.text!r}"
            )
        self._next()
        return _RefItem("expr", name=alias.value, expr=node)

    def _arith_op_ahead(self) -> bool:
        tok = self._peek()
        return tok.kind == "star" or (
            tok.kind == "op" and tok.value in ("+", "-", "/")
        )

    def _parse_order_item(self) -> _RefItem:
        item = self._parse_column_or_agg()
        descending = False
        if self._accept_keyword("asc"):
            descending = False
        elif self._accept_keyword("desc"):
            descending = True
        nulls_first: bool | None = None
        if self._accept_keyword("nulls"):
            if self._accept_keyword("first"):
                nulls_first = True
            elif self._accept_keyword("last"):
                nulls_first = False
            else:
                tok = self._peek()
                raise QuerySyntaxError(
                    f"expected FIRST or LAST after NULLS, got {tok.text!r}"
                )
        return _RefItem(
            item.kind,
            name=item.name,
            func=item.func,
            arg=item.arg,
            distinct=item.distinct,
            descending=descending,
            nulls_first=nulls_first,
            table=item.table,
            table_quoted=item.table_quoted,
            arg_table=item.arg_table,
            arg_table_quoted=item.arg_table_quoted,
        )

    def _qualifier_ahead(self) -> bool:
        nxt = self.tokens[self.pos + 1]
        return nxt.kind == "op" and nxt.value == "."

    def _parse_qualified_column(self) -> _RefItem:
        # Current token starts "table.column"; consumes all three tokens.
        table_tok = self._next()
        self._expect_op(".")
        name_tok = self._peek()
        if name_tok.kind not in ("ident", "qident"):
            raise QuerySyntaxError(
                f"expected identifier after '.', got {name_tok.text!r}"
            )
        self._next()
        return _RefItem(
            "column",
            name=name_tok.value,
            table=table_tok.value,
            table_quoted=table_tok.kind == "qident",
        )

    def _parse_column_or_agg(self) -> _RefItem:
        tok = self._peek()
        if tok.kind not in ("ident", "qident"):
            raise QuerySyntaxError(f"expected identifier, got {tok.text!r}")
        if self._allow_join and self._qualifier_ahead():
            return self._parse_qualified_column()
        # A quoted identifier, or a bare word not spelled like an aggregate,
        # can only be a plain column reference.
        if tok.kind == "qident" or tok.value.lower() not in _AGG_NAMES:
            self._next()
            return _RefItem("column", name=tok.value)
        func = tok.value.lower()
        self._next()  # consume the function name
        if not self._accept_op("("):
            # e.g. a column literally named "count" -- the name token is
            # already consumed, so it is a plain column reference.
            return _RefItem("column", name=tok.value)
        arg, arg_table, arg_table_quoted, distinct = self._parse_aggregate_args(func)
        self._expect_op(")")
        return _RefItem(
            "agg",
            func=func.upper(),
            arg=arg,
            distinct=distinct,
            arg_table=arg_table,
            arg_table_quoted=arg_table_quoted,
        )

    def _parse_aggregate_args(self, func: str) -> tuple:
        # Exactly one argument: '*' for COUNT, otherwise one (possibly
        # table-qualified) identifier, optionally preceded by DISTINCT.
        # Returns (name, table, table_quoted, distinct); name "" stands
        # for '*'.  DISTINCT only deduplicates the argument values of this
        # call; it accepts a single column reference only, and any other
        # use (a star, a missing argument, an expression, extra arguments
        # or a nested call) is a syntax error raised before any file is
        # opened.
        tok = self._peek()
        if tok.kind == "keyword" and tok.value == "case":
            raise QueryValidationError(
                f"{func.upper()} argument must be a column reference, not a CASE expression"
            )
        if tok.kind == "keyword" and tok.value == "distinct":
            self._next()
            tok = self._peek()
            if tok.kind == "star":
                raise QuerySyntaxError(
                    f"{func.upper()}(DISTINCT ...) does not accept '*'"
                )
            if tok.kind not in ("ident", "qident"):
                raise QuerySyntaxError(
                    f"{func.upper()}(DISTINCT ...) requires one column argument, "
                    f"got {tok.text!r}"
                )
            if self._allow_join and self._qualifier_ahead():
                item = self._parse_qualified_column()
                name, table, table_quoted = item.name, item.table, item.table_quoted
            else:
                self._next()
                name, table, table_quoted = tok.value, None, False
            nxt = self._peek()
            if nxt.kind == "op" and nxt.value == "(":
                raise QuerySyntaxError(
                    f"{func.upper()}(DISTINCT ...) argument must be a column name, "
                    "not a function call"
                )
            if self._arith_op_ahead():
                raise QuerySyntaxError(
                    f"{func.upper()}(DISTINCT ...) argument must be a column "
                    "reference, not an expression"
                )
            return (name, table, table_quoted, True)
        if tok.kind == "star":
            if func != "count":
                raise QuerySyntaxError(f"{func.upper()} does not accept '*'")
            self._next()
            return ("", None, False, False)
        if tok.kind not in ("ident", "qident"):
            raise QuerySyntaxError(
                f"{func.upper()} requires one column argument, got {tok.text!r}"
            )
        if self._allow_join and self._qualifier_ahead():
            item = self._parse_qualified_column()
            nxt = self._peek()
            if nxt.kind == "op" and nxt.value == "(":
                raise QuerySyntaxError(
                    f"{func.upper()} argument must be a column name, not a function call"
                )
            self._reject_expression_in_agg_arg(func)
            return (item.name, item.table, item.table_quoted, False)
        self._next()
        nxt = self._peek()
        if nxt.kind == "op" and nxt.value == "(":
            if tok.kind == "ident" and tok.value.lower() in _AGG_NAMES:
                raise QueryValidationError(
                    f"aggregate functions must not be nested: "
                    f"{func.upper()}({tok.value.upper()}(...))"
                )
            raise QuerySyntaxError(
                f"{func.upper()} argument must be a column name, not a function call"
            )
        self._reject_expression_in_agg_arg(func)
        return (tok.value, None, False, False)

    def _reject_expression_in_agg_arg(self, func: str) -> None:
        # Aggregate arguments stay limited to plain column references; a
        # scalar expression inside an aggregate query is a validation error.
        if self._arith_op_ahead():
            raise QueryValidationError(
                f"{func.upper()} argument must be a column reference, not an expression"
            )

    def _parse_limit(self) -> int:
        tok = self._peek()
        if (
            tok.kind != "number"
            or isinstance(tok.value, bool)
            or not isinstance(tok.value, int)
        ):
            raise QuerySyntaxError("LIMIT requires an unsigned integer literal")
        self._next()
        if not (0 <= tok.value <= 2**63 - 1):
            raise QuerySyntaxError(
                f"LIMIT value {tok.value} is outside the allowed range"
            )
        return tok.value

    def _expect_table_name(self) -> _Token:
        tok = self._peek()
        if tok.kind in ("ident", "qident"):
            return self._next()
        raise QuerySyntaxError(f"expected table name, got {tok.text!r}")

    def _parse_group_name(self) -> tuple:
        # Returns a (table|None, table_quoted, name) triple.
        tok = self._peek()
        if tok.kind == "keyword" and tok.value == "case":
            raise QueryValidationError(
                "GROUP BY only accepts column references, not CASE expressions"
            )
        if tok.kind not in ("ident", "qident"):
            raise QuerySyntaxError(f"expected identifier, got {tok.text!r}")
        if self._allow_join and self._qualifier_ahead():
            item = self._parse_qualified_column()
            result = (item.table, item.table_quoted, item.name)
        else:
            self._next()
            result = (None, False, tok.value)
        # GROUP BY keeps accepting column references only; a scalar
        # expression in an aggregate query is a validation error.
        if self._arith_op_ahead():
            raise QueryValidationError(
                "GROUP BY only accepts column references, not expressions"
            )
        return result

    # HAVING expression grammar ------------------------------------------------
    #
    # HAVING shares the WHERE boolean structure (NOT / AND / OR / comparison /
    # IS [NOT] NULL, parentheses, typed literals) but its value operands are
    # restricted to grouping columns, aggregate calls and literals; CASE,
    # scalar arithmetic and nested aggregates are rejected.

    def _parse_having_or(self) -> tuple:
        node = self._parse_having_and()
        while self._accept_keyword("or"):
            node = ("or", node, self._parse_having_and())
        return node

    def _parse_having_and(self) -> tuple:
        node = self._parse_having_comparison()
        while self._accept_keyword("and"):
            node = ("and", node, self._parse_having_comparison())
        return node

    def _parse_having_comparison(self) -> tuple:
        left = self._parse_having_not_factor()
        tok = self._peek()
        if tok.kind == "op" and tok.value in self._CMP_OPS:
            self._next()
            right = self._parse_having_not_factor()
            return ("cmp", tok.value, left, right)
        return left

    def _parse_having_not_factor(self) -> tuple:
        if self._accept_keyword("not"):
            return ("not", self._parse_having_not_factor())
        return self._parse_having_postfix()

    def _parse_having_postfix(self) -> tuple:
        node = self._parse_having_operand()
        # HAVING operands stay scalar references; arithmetic on top of a
        # group column or aggregate result is a validation error.
        if self._arith_op_ahead():
            raise QueryValidationError(
                "HAVING does not allow scalar arithmetic expressions"
            )
        if self._accept_keyword("is"):
            negate = self._accept_keyword("not")
            self._expect_keyword("null")
            node = ("isnull", node, negate)
        return node

    def _parse_having_operand(self) -> tuple:
        tok = self._peek()
        if self._accept_op("("):
            node = self._parse_having_or()
            self._expect_op(")")
            return node
        if tok.kind == "op" and tok.value in ("+", "-"):
            # Only a signed numeric literal is accepted; unary +/- around a
            # column or aggregate is scalar arithmetic and stays rejected.
            nxt = self.tokens[self.pos + 1]
            if nxt.kind == "number" and not isinstance(nxt.value, bool):
                self._next()
                self._next()
                value = nxt.value
                if isinstance(value, int):
                    if tok.value == "-":
                        value = -value
                    if not (_INT64_MIN <= value <= _INT64_MAX):
                        raise QuerySyntaxError(
                            "integer literal is outside the int64 range"
                        )
                    return ("literal", value, "int64")
                return ("literal", -value if tok.value == "-" else value, "float64")
            if nxt.kind == "eof":
                raise QuerySyntaxError(
                    f"expected a numeric literal after {tok.value!r} in HAVING"
                )
            raise QueryValidationError(
                "HAVING does not allow scalar arithmetic expressions"
            )
        if tok.kind == "star":
            raise QuerySyntaxError("'*' is not allowed in HAVING")
        if tok.kind == "number":
            self._next()
            value = tok.value
            if isinstance(value, bool):
                return ("literal", value, "bool")
            return (
                "literal",
                value,
                "int64" if isinstance(value, int) else "float64",
            )
        if tok.kind == "string":
            self._next()
            return ("literal", tok.value, "utf8")
        if tok.kind in ("ident", "qident"):
            if (
                tok.kind == "ident"
                and tok.value.lower() in _AGG_NAMES
                and self.tokens[self.pos + 1].kind == "op"
                and self.tokens[self.pos + 1].value == "("
            ):
                func = tok.value.lower()
                self._next()  # function name
                self._expect_op("(")
                arg, arg_table, arg_table_quoted, distinct = self._parse_aggregate_args(
                    func
                )
                self._expect_op(")")
                return (
                    "hagg",
                    func.upper(),
                    arg,
                    arg_table,
                    arg_table_quoted,
                    distinct,
                )
            if self._allow_join and self._qualifier_ahead():
                item = self._parse_qualified_column()
                return ("column", item.name, item.table, item.table_quoted)
            if tok.kind == "qident" or tok.value.lower() not in _AGG_NAMES:
                self._next()
                return ("column", tok.value, None, False)
            # An aggregate-style name not followed by '(' is a plain column.
            self._next()
            return ("column", tok.value, None, False)
        if tok.kind == "keyword":
            if tok.value == "case":
                raise QueryValidationError("CASE expressions are not allowed in HAVING")
            raise QuerySyntaxError(
                f"unexpected keyword {tok.text.upper()!r} in HAVING expression"
            )
        raise QuerySyntaxError(
            f"unexpected token {tok.text!r} in HAVING expression"
        )

    # WHERE expression grammar ------------------------------------------------

    def _parse_or(self) -> tuple:
        node = self._parse_and()
        while self._accept_keyword("or"):
            node = ("or", node, self._parse_and())
        return node

    def _parse_and(self) -> tuple:
        node = self._parse_comparison()
        while self._accept_keyword("and"):
            node = ("and", node, self._parse_comparison())
        return node

    def _parse_comparison(self) -> tuple:
        left = self._parse_not_factor()
        tok = self._peek()
        if tok.kind == "op" and tok.value in self._CMP_OPS:
            self._next()
            right = self._parse_not_factor()
            return ("cmp", tok.value, left, right)
        return left

    def _parse_not_factor(self) -> tuple:
        if self._accept_keyword("not"):
            return ("not", self._parse_not_factor())
        return self._parse_postfix()

    def _parse_postfix(self) -> tuple:
        node = self._parse_arith()
        if self._accept_keyword("is"):
            negate = self._accept_keyword("not")
            self._expect_keyword("null")
            node = ("isnull", node, negate)
        return node

    # Arithmetic expressions (numeric scalar operands) ------------------------
    # Precedence (highest first): parentheses, unary +/-, * /, + -.

    def _parse_arith(self) -> tuple:
        node = self._parse_term()
        while True:
            tok = self._peek()
            if tok.kind == "op" and tok.value in ("+", "-"):
                self._next()
                node = ("arith", tok.value, node, self._parse_term())
            else:
                return node

    def _parse_term(self) -> tuple:
        node = self._parse_factor()
        while True:
            tok = self._peek()
            if tok.kind == "star":
                self._next()
                node = ("arith", "*", node, self._parse_factor())
            elif tok.kind == "op" and tok.value == "/":
                self._next()
                node = ("arith", "/", node, self._parse_factor())
            else:
                return node

    def _parse_factor(self) -> tuple:
        tok = self._peek()
        if tok.kind == "op" and tok.value in ("+", "-"):
            negate = tok.value == "-"
            self._next()
            nxt = self._peek()
            if nxt.kind == "number" and not isinstance(nxt.value, bool):
                # A signed numeric literal is folded at parse time so the
                # signed int64 range check keeps its historical meaning
                # (-9223372036854775808 is valid, the bare magnitude is not).
                self._next()
                value = nxt.value
                if isinstance(value, int):
                    if negate:
                        value = -value
                    if not (-(2**63) <= value <= 2**63 - 1):
                        raise QuerySyntaxError(
                            "integer literal is outside the int64 range"
                        )
                    return ("literal", value, "int64")
                if negate:
                    value = -value
                return ("literal", value, "float64")
            if nxt.kind not in ("ident", "qident") and not (
                nxt.kind == "op" and nxt.value == "("
            ) and not (nxt.kind == "keyword" and nxt.value == "case"):
                raise QuerySyntaxError(
                    "sign must be followed by a numeric literal or expression"
                )
            return ("unary", self._parse_factor(), negate)
        node = self._parse_atom()
        if (
            node[0] == "literal"
            and isinstance(node[1], int)
            and not isinstance(node[1], bool)
            and node[1] > 2**63 - 1
        ):
            # The magnitude 2**63 is only reachable through an explicit
            # negation; a bare literal must fit the signed int64 range.
            raise QuerySyntaxError("integer literal is outside the int64 range")
        return node

    def _parse_atom(self) -> tuple:
        if self._accept_op("("):
            node = self._parse_or()
            self._expect_op(")")
            return node
        if self._accept_keyword("case"):
            return self._parse_case()
        tok = self._peek()
        if tok.kind == "number":
            self._next()
            value = tok.value
            if isinstance(value, bool):
                return ("literal", value, "bool")
            type_name = "int64" if isinstance(value, int) else "float64"
            return ("literal", value, type_name)
        if tok.kind == "string":
            self._next()
            return ("literal", tok.value, "utf8")
        if tok.kind in ("ident", "qident"):
            if self._allow_join and self._qualifier_ahead():
                item = self._parse_qualified_column()
                return ("column", item.name, item.table, item.table_quoted)
            nxt = self.tokens[self.pos + 1]
            if (
                tok.kind == "ident"
                and tok.value.lower() in _AGG_NAMES
                and nxt.kind == "op"
                and nxt.value == "("
            ):
                raise QueryValidationError(
                    f"aggregate {tok.value.upper()}(...) is not allowed here"
                )
            self._next()
            return ("column", tok.value, None, False)
        if tok.kind == "keyword":
            raise QuerySyntaxError(f"unexpected keyword {tok.text.upper()!r} in expression")
        raise QuerySyntaxError(f"unexpected token {tok.text!r} in expression")

    def _parse_case(self) -> tuple:
        # A searched CASE: at least one "WHEN cond THEN result", an optional
        # "ELSE result", terminated by END.  Conditions are boolean
        # expressions; results are scalar (numeric, bool or utf8) and may
        # themselves contain nested CASE expressions.  The simple form
        # (CASE value WHEN ...) is not supported.
        branches: list[tuple] = []
        while True:
            self._expect_keyword("when")
            condition = self._parse_or()
            self._expect_keyword("then")
            result = self._parse_arith()
            branches.append((condition, result))
            tok = self._peek()
            if not (tok.kind == "keyword" and tok.value == "when"):
                break
        else_node = None
        if self._accept_keyword("else"):
            else_node = self._parse_arith()
        self._expect_keyword("end")
        return ("case", tuple(branches), else_node)

    # Token helpers -----------------------------------------------------------

    def _peek(self) -> _Token:
        return self.tokens[self.pos]

    def _next(self) -> _Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _accept(self, kind: str) -> bool:
        if self._peek().kind == kind:
            self.pos += 1
            return True
        return False

    def _accept_op(self, value: str) -> bool:
        tok = self._peek()
        if tok.kind == "op" and tok.value == value:
            self.pos += 1
            return True
        return False

    def _accept_keyword(self, value: str) -> bool:
        tok = self._peek()
        if tok.kind == "keyword" and tok.value == value:
            self.pos += 1
            return True
        return False

    def _accept_word(self, value: str) -> bool:
        # Match a reserved word that is otherwise lexed as an ordinary
        # (non-keyword) identifier; used for RIGHT / FULL / OUTER, which keep
        # their meaning as column names outside the join clause.
        tok = self._peek()
        if tok.kind == "ident" and tok.value.lower() == value:
            self.pos += 1
            return True
        return False

    def _expect_word(self, value: str) -> _Token:
        tok = self._peek()
        if tok.kind != "ident" or tok.value.lower() != value:
            raise QuerySyntaxError(f"expected {value.upper()}, got {tok.text!r}")
        return self._next()

    def _expect_keyword(self, value: str) -> _Token:
        tok = self._peek()
        if tok.kind != "keyword" or tok.value != value:
            raise QuerySyntaxError(f"expected {value.upper()}, got {tok.text!r}")
        return self._next()

    def _expect_op(self, value: str) -> _Token:
        tok = self._peek()
        if tok.kind != "op" or tok.value != value:
            raise QuerySyntaxError(f"expected {value!r}, got {tok.text!r}")
        return self._next()


# ---------------------------------------------------------------------------
# Binding: schema resolution, projection checks and type compatibility
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BoundItem:
    """A validated projection element."""

    kind: str  # "column" | "agg" | "expr"
    output_name: str
    # kind == "column":
    col_index: int = -1
    # kind == "agg":
    func: str = ""
    arg_index: int = -1  # -1 means COUNT(*)
    arg_type: str = ""
    distinct: bool = False  # argument values are deduplicated per group
    # kind == "expr":
    expr: tuple | None = None  # bound scalar expression
    out_type: str = ""
    nullable: bool = True


def _bind_select(select: _Select, schema: Schema, expected_table="input") -> dict:
    if expected_table is not None:
        table_matches = (
            select.table == expected_table
            if select.table_quoted
            else select.table.lower() == expected_table.lower()
        )
        if not table_matches:
            raise QueryValidationError(
                f"unknown table {select.table!r}; only {expected_table!r} is supported"
            )

    where = _bind_expr(select.where, schema) if select.where is not None else None
    if where is not None and where.type != "bool":
        raise QueryValidationError(
            f"WHERE clause must be boolean, got {where.type}"
        )

    if select.distinct and select.is_aggregate_query:
        raise QueryValidationError(
            "DISTINCT is not allowed with GROUP BY, HAVING or aggregate projections"
        )

    if not select.is_aggregate_query:
        return _bind_plain(select, schema, where)
    return _bind_aggregate(select, schema, where)


def _bind_plain(select: _Select, schema: Schema, where) -> dict:
    # Non-aggregate path: '*' or a comma-separated list of plain columns and
    # computed scalar expressions (each named with AS).
    bound_items: list[_BoundItem] = []
    if select.star:
        for i, col in enumerate(schema.columns):
            bound_items.append(
                _BoundItem(
                    "column",
                    output_name=col.name,
                    col_index=i,
                    out_type=col.type,
                    nullable=col.nullable,
                )
            )
    else:
        output_seen: set[str] = set()
        for item in select.items:
            if item.kind == "star":
                # A star mixed into a plain projection remains grammatical.
                raise QuerySyntaxError("'*' cannot be mixed with other projection items")
            if item.kind == "column":
                name = item.name
                if name in output_seen:
                    raise QueryValidationError(
                        f"duplicate column in projection: {name!r}"
                    )
                try:
                    col_index = schema.index(name)
                except KeyError:
                    raise QueryValidationError(f"unknown column: {name!r}") from None
                col = schema.columns[col_index]
                output_seen.add(col.name)
                bound_items.append(
                    _BoundItem(
                        "column",
                        output_name=col.name,
                        col_index=col_index,
                        out_type=col.type,
                        nullable=col.nullable,
                    )
                )
            else:  # "expr"
                expr = _bind_expr(item.expr, schema)
                out_type = expr.type
                # Arithmetic scalar expressions stay numeric-only; a CASE
                # expression may additionally yield bool or utf8 results
                # (its own WHEN/ELSE type consistency is checked at binding).
                if isinstance(expr, _Case):
                    allowed_types = ("int64", "float64", "bool", "utf8")
                else:
                    allowed_types = ("int64", "float64")
                if out_type not in allowed_types:
                    raise QueryValidationError(
                        f"SELECT expression must be numeric, got {out_type}"
                    )
                alias = item.name
                if alias in output_seen:
                    raise QueryValidationError(f"duplicate result column: {alias!r}")
                output_seen.add(alias)
                bound_items.append(
                    _BoundItem(
                        "expr",
                        output_name=alias,
                        expr=expr,
                        out_type=out_type,
                        nullable=expr.nullable,
                    )
                )
    aliases = {
        item.output_name: item.expr for item in bound_items if item.kind == "expr"
    }
    projected = {item.output_name: i for i, item in enumerate(bound_items)}
    order_by = _bind_plain_order_by(
        select.order_by, schema, aliases, select.distinct, projected
    )
    return {
        "mode": "plain",
        "items": tuple(bound_items),
        "aggregates": (),
        "where": where,
        "having": None,
        "order_by": order_by,
        "distinct": select.distinct,
    }


def _bind_plain_order_by(
    select_order_by, schema: Schema, aliases: dict, distinct: bool, projected: dict
):
    if select_order_by is None:
        return None
    bound: list[tuple] = []
    order_seen: set[str] = set()
    for item in select_order_by:
        if item.kind == "agg":
            raise QueryValidationError(
                f"aggregate {item.func}({_arg_label(item)}) may appear in ORDER BY "
                "only as part of an aggregate query"
            )
        name = item.name
        if name in order_seen:
            raise QueryValidationError(
                f"duplicate column in ORDER BY: {name!r}"
            )
        order_seen.add(name)
        nulls_first = item.nulls_first if item.nulls_first is not None else False
        if distinct:
            # A DISTINCT query sorts its deduplicated result rows, so every
            # ORDER BY name must be one of the projected outputs (a bare
            # projected column or an explicit AS alias).
            if name not in projected:
                try:
                    schema.index(name)
                except KeyError:
                    raise QueryValidationError(
                        f"unknown ORDER BY column: {name!r}"
                    ) from None
                raise QueryValidationError(
                    f"ORDER BY column {name!r} is not part of the SELECT DISTINCT results"
                )
            bound.append(("out", projected[name], item.descending, nulls_first))
            continue
        # A SELECT alias shadows an input column of the same name.
        if name in aliases:
            bound.append(("expr", aliases[name], item.descending, nulls_first))
            continue
        try:
            col_index = schema.index(name)
        except KeyError:
            raise QueryValidationError(
                f"unknown ORDER BY column: {name!r}"
            ) from None
        bound.append(("col", col_index, item.descending, nulls_first))
    return tuple(bound)


def _bind_aggregate(select: _Select, schema: Schema, where) -> dict:
    if select.star:
        raise QueryValidationError("'*' cannot be combined with aggregates or GROUP BY")

    if (
        select.having is not None
        and select.group_by is None
        and not select.has_aggregate
        and not _having_tree_has_aggregate(select.having)
    ):
        # HAVING filters formed groups; without aggregation or GROUP BY the
        # statement stays a non-aggregate query and HAVING has no meaning.
        raise QueryValidationError(
            "HAVING is only valid with GROUP BY or an aggregate"
        )

    # Resolve GROUP BY columns first: order matters for the grouping key, and
    # duplicates / unknown columns are rejected here.  Entries are
    # (table, table_quoted, name) triples; names reaching the binder are
    # already canonical (single-file queries never carry a qualifier).
    group_indices: list[int] = []
    group_seen: set[str] = set()
    for _table, _table_quoted, name in select.group_by or ():
        if name in group_seen:
            raise QueryValidationError(f"duplicate column in GROUP BY: {name!r}")
        group_seen.add(name)
        try:
            group_indices.append(schema.index(name))
        except KeyError:
            raise QueryValidationError(f"unknown GROUP BY column: {name!r}") from None
    group_index_set = set(group_indices)

    # One registry of distinct aggregates shared by SELECT, HAVING and ORDER
    # BY, keyed by (function, argument index, DISTINCT flag) and kept in
    # first-reference order.  HAVING-only aggregates are computed but never
    # projected.
    agg_registry: dict[tuple, int] = {}
    agg_order: list[_BoundItem] = []

    def require_aggregate(func: str, arg_name: str, distinct: bool = False) -> int:
        arg_index, arg_col, label, out_type, nullable = _resolve_agg_call(
            func, arg_name, schema, distinct
        )
        key = (func, arg_index, distinct)
        slot = agg_registry.get(key)
        if slot is None:
            slot = len(agg_order)
            agg_registry[key] = slot
            agg_order.append(
                _BoundItem(
                    "agg",
                    output_name=label,
                    func=func,
                    arg_index=arg_index,
                    arg_type=arg_col.type if arg_col is not None else "",
                    distinct=distinct,
                    out_type=out_type,
                    nullable=nullable,
                )
            )
        return slot

    # Resolve the projection.  Plain columns must be grouped; output names
    # (group column names and canonical "FUNC(arg)" labels) must be unique.
    bound_items: list[_BoundItem] = []
    output_seen: set[str] = set()
    has_plain = False
    for item in select.items:
        if item.kind == "star":
            raise QueryValidationError(
                "'*' cannot be combined with aggregates or GROUP BY"
            )
        if item.kind == "expr":
            raise QueryValidationError(
                "scalar expressions are not supported in aggregate queries"
            )
        if item.kind == "column":
            has_plain = True
            name = item.name
            try:
                col_index = schema.index(name)
            except KeyError:
                raise QueryValidationError(f"unknown column: {name!r}") from None
            if col_index not in group_index_set:
                raise QueryValidationError(
                    f"column {name!r} must appear in GROUP BY or be wrapped in an aggregate"
                )
            label = name
            if label in output_seen:
                raise QueryValidationError(f"duplicate result column: {name!r}")
            output_seen.add(label)
            col = schema.columns[col_index]
            bound_items.append(
                _BoundItem(
                    "column",
                    output_name=label,
                    col_index=col_index,
                    out_type=col.type,
                    nullable=col.nullable,
                )
            )
        else:
            slot = require_aggregate(item.func, item.arg, item.distinct)
            agg_item = agg_order[slot]
            if agg_item.output_name in output_seen:
                raise QueryValidationError(
                    f"duplicate result column: {agg_item.output_name!r}"
                )
            output_seen.add(agg_item.output_name)
            bound_items.append(agg_item)

    if select.group_by is None and has_plain:
        # No grouping: every projected column must be an aggregate.
        raise QueryValidationError(
            "without GROUP BY, the projection may contain aggregates only"
        )

    # HAVING filters the formed groups; it may reuse projected aggregates and
    # may introduce further aggregates that never reach the projection.
    having = None
    if select.having is not None:
        having = _bind_having(
            select.having, schema, group_index_set, agg_order, require_aggregate
        )
        _require_boolean(having, "HAVING clause")

    order_by = _bind_aggregate_order_by(
        select.order_by, bound_items, schema
    )
    return {
        "mode": "aggregate",
        "group_indices": tuple(group_indices),
        "items": tuple(bound_items),
        "aggregates": tuple(agg_order),
        "where": where,
        "having": having,
        "order_by": order_by,
        "distinct": False,
    }


def _having_tree_has_aggregate(node: tuple) -> bool:
    """Whether a parsed (unbound) HAVING tree names at least one aggregate."""
    tag = node[0]
    if tag == "hagg":
        return True
    if tag in ("literal", "column"):
        return False
    if tag in ("not", "isnull"):
        return _having_tree_has_aggregate(node[1])
    if tag == "cmp":
        return _having_tree_has_aggregate(node[2]) or _having_tree_has_aggregate(
            node[3]
        )
    return _having_tree_has_aggregate(node[1]) or _having_tree_has_aggregate(node[2])


def _bind_having(
    node: tuple,
    schema: Schema,
    group_index_set: set[int],
    agg_order: list[_BoundItem],
    require_aggregate,
) -> _Expr:
    tag = node[0]
    if tag == "literal":
        return _Literal(node[1], node[2])
    if tag == "column":
        name = node[1]
        try:
            col_index = schema.index(name)
        except KeyError:
            raise QueryValidationError(f"unknown column: {name!r}") from None
        if col_index not in group_index_set:
            raise QueryValidationError(
                f"HAVING column {name!r} must appear in GROUP BY or be wrapped in an aggregate"
            )
        col = schema.columns[col_index]
        return _Column(name, col_index, col.type, col.nullable)
    if tag == "hagg":
        slot = require_aggregate(node[1], node[2], node[5])
        agg_item = agg_order[slot]
        arg_name = (
            None
            if agg_item.arg_index < 0
            else schema.columns[agg_item.arg_index].name
        )
        return _Aggregate(
            agg_item.func,
            slot,
            arg_name,
            agg_item.distinct,
            agg_item.out_type,
            agg_item.nullable,
        )
    if tag == "not":
        operand = _bind_having(
            node[1], schema, group_index_set, agg_order, require_aggregate
        )
        _require_boolean(operand, "NOT")
        return _Not(operand)
    if tag in ("and", "or"):
        left = _bind_having(
            node[1], schema, group_index_set, agg_order, require_aggregate
        )
        right = _bind_having(
            node[2], schema, group_index_set, agg_order, require_aggregate
        )
        _require_boolean(left, tag.upper())
        _require_boolean(right, tag.upper())
        return _Logic(tag, left, right)
    if tag == "isnull":
        operand = _bind_having(
            node[1], schema, group_index_set, agg_order, require_aggregate
        )
        if not operand.is_having_leaf:
            raise QuerySyntaxError(
                "IS NULL operand must be a group column, an aggregate or a literal"
            )
        return _IsNull(operand, node[2])
    if tag == "cmp":
        op = node[1]
        left = _bind_having(
            node[2], schema, group_index_set, agg_order, require_aggregate
        )
        right = _bind_having(
            node[3], schema, group_index_set, agg_order, require_aggregate
        )
        if not (left.is_having_leaf and right.is_having_leaf):
            raise QuerySyntaxError(
                "comparison operands must be group columns, aggregates or literals"
            )
        _check_comparison_types(op, left, right)
        return _Cmp(op, left, right)
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


def _agg_result_type(func: str, arg_col: ColumnSchema | None) -> tuple[str, bool]:
    """Static (type, nullable) of an aggregate call; rejects illegal args."""
    if func == "COUNT":
        return "int64", False
    if func == "SUM":
        if arg_col.type not in ("int64", "float64"):
            raise QueryValidationError(
                f"SUM requires an int64 or float64 argument, got {arg_col.type}"
            )
        return arg_col.type, True
    if func == "AVG":
        if arg_col.type not in ("int64", "float64"):
            raise QueryValidationError(
                f"AVG requires an int64 or float64 argument, got {arg_col.type}"
            )
        return "float64", True
    # MIN / MAX accept all four existing types.
    return arg_col.type, True


def _resolve_agg_call(
    func: str, arg_name: str, schema: Schema, distinct: bool = False
) -> tuple[int, ColumnSchema | None, str, str, bool]:
    """Resolve one aggregate call to (arg_index, arg_col|None, label, type, nullable)."""
    if arg_name == "":
        if func != "COUNT":
            raise QuerySyntaxError(f"{func} does not accept '*'")
        out_type, nullable = _agg_result_type("COUNT", None)
        return -1, None, "COUNT(*)", out_type, nullable
    try:
        arg_index = schema.index(arg_name)
    except KeyError:
        raise QueryValidationError(f"unknown column: {arg_name!r}") from None
    arg_col = schema.columns[arg_index]
    out_type, nullable = _agg_result_type(func, arg_col)
    # The result label uses the column name as spelled in the schema.
    if distinct:
        return (
            arg_index,
            arg_col,
            f"{func}(DISTINCT {arg_col.name})",
            out_type,
            nullable,
        )
    return arg_index, arg_col, f"{func}({arg_col.name})", out_type, nullable


def _arg_label(item: _RefItem) -> str:
    return "*" if item.arg == "" else item.arg


def _bind_aggregate_order_by(
    select_order_by,
    bound_items: list[_BoundItem],
    schema: Schema,
):
    if select_order_by is None:
        return None

    # Map every SELECT result to its output position.  ORDER BY in an
    # aggregate query may only name those selected results.
    selected: dict[tuple, int] = {}
    for i, bound in enumerate(bound_items):
        if bound.kind == "column":
            selected[("column", schema.columns[bound.col_index].name)] = i
        else:
            arg_key = "*" if bound.arg_index == -1 else schema.columns[bound.arg_index].name
            selected[("agg", f"{bound.func}|{arg_key}|{bound.distinct}")] = i

    bound_order: list[tuple[int, bool, bool]] = []
    order_seen: set[str] = set()
    for item in select_order_by:
        if item.kind == "column":
            key = ("column", item.name)
            label = item.name
            if key not in selected:
                # Keep the same "unknown column" wording for names the schema
                # does not know at all; anything else is an unselected result.
                try:
                    schema.index(item.name)
                except KeyError:
                    raise QueryValidationError(
                        f"unknown ORDER BY column: {item.name!r}"
                    ) from None
                raise QueryValidationError(
                    f"ORDER BY column {item.name!r} is not part of the selected results"
                )
        else:
            if item.arg == "":
                arg_key = "*"
                label = "COUNT(*)"
            else:
                try:
                    real_name = schema.columns[schema.index(item.arg)].name
                except KeyError:
                    raise QueryValidationError(
                        f"unknown ORDER BY column: {item.arg!r}"
                    ) from None
                arg_key = real_name
                if item.distinct:
                    label = f"{item.func}(DISTINCT {real_name})"
                else:
                    label = f"{item.func}({real_name})"
            key = ("agg", f"{item.func}|{arg_key}|{item.distinct}")
            if key not in selected:
                raise QueryValidationError(
                    f"ORDER BY aggregate {label} is not part of the selected results"
                )
        if label in order_seen:
            raise QueryValidationError(f"duplicate column in ORDER BY: {label!r}")
        order_seen.add(label)
        nulls_first = item.nulls_first if item.nulls_first is not None else False
        bound_order.append((selected[key], item.descending, nulls_first))
    return tuple(bound_order)


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


def _bind_expr(node: tuple, schema: Schema) -> _Expr:
    tag = node[0]
    if tag == "literal":
        return _Literal(node[1], node[2])
    if tag == "column":
        name = node[1]
        try:
            index = schema.index(name)
        except KeyError:
            raise QueryValidationError(f"unknown column: {name!r}") from None
        col = schema.columns[index]
        return _Column(name, index, col.type, col.nullable)
    if tag == "not":
        operand = _bind_expr(node[1], schema)
        _require_boolean(operand, "NOT")
        return _Not(operand)
    if tag in ("and", "or"):
        left = _bind_expr(node[1], schema)
        right = _bind_expr(node[2], schema)
        _require_boolean(left, tag.upper())
        _require_boolean(right, tag.upper())
        return _Logic(tag, left, right)
    if tag == "isnull":
        operand = _bind_expr(node[1], schema)
        if not operand.is_value:
            raise QuerySyntaxError(
                "IS NULL operand must be a column reference or a literal"
            )
        return _IsNull(operand, node[2])
    if tag == "unary":
        operand = _bind_expr(node[1], schema)
        type_name = operand.type
        if type_name not in ("int64", "float64"):
            raise QueryValidationError(
                f"unary +/- requires a numeric operand, got {type_name}"
            )
        return _Unary(operand, node[2])
    if tag == "arith":
        left = _bind_expr(node[2], schema)
        right = _bind_expr(node[3], schema)
        left_t = left.type
        right_t = right.type
        if left_t not in ("int64", "float64") or right_t not in ("int64", "float64"):
            raise QueryValidationError(
                f"arithmetic requires numeric operands, got {left_t} and {right_t}"
            )
        op = node[1]
        out_type = (
            "float64"
            if op == "/" or "float64" in (left_t, right_t)
            else "int64"
        )
        return _Arith(op, left, right, out_type)
    if tag == "case":
        bound_branches = []
        result_types: list[str] = []
        for cond_node, result_node in node[1]:
            cond = _bind_expr(cond_node, schema)
            _require_boolean(cond, "WHEN")
            result = _bind_expr(result_node, schema)
            bound_branches.append((cond, result))
            result_types.append(result.type)
        else_bound = None
        if node[2] is not None:
            else_bound = _bind_expr(node[2], schema)
            result_types.append(else_bound.type)
        out_type = _unify_case_types(result_types)
        return _Case(tuple(bound_branches), else_bound, out_type)
    if tag == "cmp":
        op = node[1]
        left = _bind_expr(node[2], schema)
        right = _bind_expr(node[3], schema)
        if not (left.is_value and right.is_value):
            raise QuerySyntaxError(
                "comparison operands must be column references or literals"
            )
        _check_comparison_types(op, left, right)
        return _Cmp(op, left, right)
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


def _check_comparison_types(op: str, left: _Expr, right: _Expr) -> None:
    """Validate the static operand types of one comparison (WHERE and HAVING)."""
    left_t = left.type
    right_t = right.type
    if "bool" in (left_t, right_t):
        if left_t != "bool" or right_t != "bool":
            raise QueryValidationError(
                f"bool can only be compared to bool, got {left_t} and {right_t}"
            )
        if op not in ("=", "!="):
            raise QueryValidationError(f"bool only supports = and !=, not {op!r}")
    elif "utf8" in (left_t, right_t):
        if left_t != "utf8" or right_t != "utf8":
            raise QueryValidationError(
                f"utf8 can only be compared to utf8, got {left_t} and {right_t}"
            )
    elif not {left_t, right_t} <= {"int64", "float64"}:
        raise QueryValidationError(
            f"cannot compare values of type {left_t} and {right_t}"
        )


def _unify_case_types(types: list[str]) -> str:
    """Unify the static types of every reachable CASE result.

    int64 and float64 may mix and unify to float64; bool and utf8 must
    match every other result exactly.  Anything else is incompatible.
    """
    unique = set(types)
    if unique <= {"int64", "float64"}:
        return "float64" if "float64" in unique else "int64"
    if len(unique) == 1:
        return next(iter(unique))
    raise QueryValidationError(
        "CASE results must have consistent types: "
        + ", ".join(sorted(unique))
    )


def _require_boolean(node: _Expr, context: str) -> None:
    if node.type != "bool":
        raise QueryValidationError(
            f"{context} requires a boolean operand, got {node.type}"
        )


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


# ---------------------------------------------------------------------------
# Row-group statistics pushdown (v2 files)
# ---------------------------------------------------------------------------
#
# For row-group-partitioned (v2) files the AND-connected, type-compatible
# comparisons and IS [NOT] NULL conditions of a WHERE clause are evaluated
# against each group's per-column statistics.  A group is skipped only when
# its statistics prove the pushed condition can never be TRUE for any of its
# rows; anything else (OR, NOT, CASE, arithmetic, column-to-column or
# cross-source comparisons, or simply undecidable ranges) keeps the group,
# and the surviving rows are still filtered row by row with the full WHERE
# condition.
#
# In a multi-table statement only a chain made entirely of INNER JOINs is
# eligible: every source row that can reach the WHERE result must reach it
# through matches on both sides, so a condition referencing one source
# alone may prune that source's groups exactly as in the single-file case.
# Any OUTER step can emit padded rows, so no group pruning is applied to
# such chains.  Bound pushable leaves are attributed to a source through
# the combined schema's column layout, which is robust against table names
# sharing dotted prefixes.

_FLIP_CMP_OP = {"=": "=", "!=": "!=", "<": ">", "<=": ">=", ">": "<", ">=": "<="}


def _is_pushable_leaf(node: _Expr) -> bool:
    """Whether a bound WHERE leaf can be checked against column statistics."""
    if isinstance(node, _IsNull):
        return isinstance(node.operand, _Column)
    if isinstance(node, _Cmp):
        left, right = node.left, node.right
        return (isinstance(left, _Column) and isinstance(right, _Literal)) or (
            isinstance(left, _Literal) and isinstance(right, _Column)
        )
    return False


def _walk_top_and_leaves(node: _Expr, sink) -> None:
    """Feed the top-level AND-connected pushable leaves to ``sink``.

    Only the AND spine at the root is unfolded; leaves nested under OR,
    NOT or any other non-AND node are never visited.  Non-pushable leaves
    sitting directly on the AND spine are simply skipped, so they neither
    prune groups nor block the pushdown of their eligible siblings.
    """
    if isinstance(node, _Logic) and node.op == "and":
        _walk_top_and_leaves(node.left, sink)
        _walk_top_and_leaves(node.right, sink)
    elif _is_pushable_leaf(node):
        sink(node)


def _extract_pushable(where: _Expr | None) -> list:
    """The AND-connected pushable leaves of a bound WHERE tree, in order."""
    conditions: list = []
    if where is not None:
        _walk_top_and_leaves(where, conditions.append)
    return conditions


def _condition_possible(cond: _Expr, group: Mapping, col_index: int) -> bool:
    """Whether ``cond`` could be TRUE for some row of ``group``.

    Only a provable "cannot be TRUE" returns False; anything undecidable
    keeps the group.  ``col_index`` is the column's position in the
    source's local schema (the condition's bound column index is already
    local in the single-file case and translated by the caller for joins).
    """
    group_rows = group["row_count"]
    if isinstance(cond, _IsNull):
        stats = group["columns"][col_index]
        if cond.negated:  # IS NOT NULL
            return stats["null_count"] < group_rows
        return stats["null_count"] > 0
    # A column-vs-literal comparison (normalised to column OP literal).
    op = cond.op
    left, right = cond.left, cond.right
    if not isinstance(left, _Column):
        op = _FLIP_CMP_OP[op]
    stats = group["columns"][col_index]
    if stats["null_count"] == group_rows:
        # All values NULL: a comparison is never TRUE.
        return False
    literal = right.value if isinstance(left, _Column) else left.value
    minimum = stats["min"]
    maximum = stats["max"]
    if op == "=":
        return minimum <= literal <= maximum
    if op == "!=":
        return not (minimum == maximum == literal)
    if op == "<":
        return minimum < literal
    if op == "<=":
        return minimum <= literal
    if op == ">":
        return maximum > literal
    return maximum >= literal  # ">="


def _leaf_col_index(cond: _Expr) -> int:
    """The bound column index carried by one pushable leaf."""
    if isinstance(cond, _IsNull):
        return cond.operand.index
    column = cond.left if isinstance(cond.left, _Column) else cond.right
    return column.index


def _select_row_groups(groups: list, pushed: list, col_indices: list) -> list[int]:
    """Indices of the row groups whose statistics do not rule them out.

    ``col_indices`` parallels ``pushed``: the local column position of
    each pushed leaf in this source's schema.
    """
    if not pushed:
        return list(range(len(groups)))
    return [
        index
        for index, group in enumerate(groups)
        if all(
            _condition_possible(cond, group, col_index)
            for cond, col_index in zip(pushed, col_indices)
        )
    ]


def _pushed_col_indices(pushed: list) -> list[int]:
    """The bound column index carried by each pushable leaf."""
    return [_leaf_col_index(cond) for cond in pushed]


def _pushed_condition_json(pushed: list) -> dict | None:
    """Render the pushed-down condition set as one condition tree (or null)."""
    if not pushed:
        return None
    node: _Expr = pushed[0]
    for cond in pushed[1:]:
        node = _Logic("and", node, cond)
    return node.to_json()


def _source_column_spans(schemas: Mapping, table_keys: tuple) -> dict:
    """Each source's ``[start, end)`` column-index range in the combined schema.

    The combined schema concatenates the sources' own columns in
    FROM/JOIN order, so the ranges are contiguous and non-overlapping.
    """
    spans: dict = {}
    start = 0
    for key in table_keys:
        width = len(schemas[key].columns)
        spans[key] = (start, start + width)
        start += width
    return spans


def _pushable_leaves_by_source(
    bound_where: tuple | None, spans: Mapping, table_keys: tuple
) -> dict:
    """Top-level AND pushable leaves attributed to one source each.

    A leaf's bound column index falls into exactly one source's combined
    schema range; the leaves keep their SQL appearance order within each
    source.  Column-to-column comparisons never reach this point (they are
    not pushable), so no leaf can reference two sources.
    """
    by_source: dict = {key: [] for key in table_keys}

    def sink(node: tuple) -> None:
        index = _leaf_col_index(node)
        for key in table_keys:
            start, end = spans[key]
            if start <= index < end:
                by_source[key].append(node)
                return

    if bound_where is not None:
        _walk_top_and_leaves(bound_where, sink)
    return by_source


def _source_scan_info(groups: list, leaves: list, span: tuple) -> dict:
    """One v2 Scan's pushdown state.

    ``leaves`` are the bound leaves attributed to this source; ``span`` is
    its combined-schema range, so global bound column indices translate to
    the source's local block positions.  The returned dict carries the
    selected group indices (for the reader) plus the total count and the
    pushed leaves (for the explain plan); :func:`_scan_pushdown_fields`
    projects its JSON-visible fields.
    """
    start = span[0]
    local_indices = [_leaf_col_index(cond) - start for cond in leaves]
    selected = _select_row_groups(groups, leaves, local_indices)
    return {
        "groups_total": len(groups),
        "selected_groups": selected,
        "pushed_leaves": leaves,
    }


def _scan_pushdown_fields(info: Mapping) -> dict:
    """The three JSON fields appended to an eligible v2 Scan operator."""
    leaves = info["pushed_leaves"]
    return {
        "row_groups_total": info["groups_total"],
        "row_groups_selected": len(info["selected_groups"]),
        "pushed_condition": _pushed_condition_json(leaves),
    }


def _scan_pushdown_info(groups: list, bound: Mapping, schema: Schema) -> dict:
    """The single-source v2 Scan statistics (the span starts at zero)."""
    pushed = _extract_pushable(bound["where"])
    return _source_scan_info(groups, pushed, (0, len(schema.columns)))


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


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def query_file(path: Any, sql: str) -> Table:
    """Run ``sql`` against the single columnar file ``path``.

    Returns a :class:`~columnar_analytics.format.Table` with columns in
    projection order.  Rows are filtered by WHERE, then grouped (aggregate
    queries may drop groups via HAVING) or sorted by ORDER BY, then capped
    by LIMIT, then projected; without ORDER BY the file's original row
    order (or the first-selected-row group order) is kept.  The statement
    is parsed before the file is touched, so purely grammatical errors
    surface as :class:`QuerySyntaxError` regardless of whether ``path``
    exists.  Unknown columns, duplicate result columns, the wrong table
    name, type-incompatible predicates, ungrouped columns, illegal
    aggregate arguments, invalid HAVING conditions or int64 SUM overflow
    raise :class:`QueryValidationError`;
    malformed files raise :class:`~columnar_analytics.format.ColumnarFormatError`;
    other I/O failures propagate as :class:`OSError`.
    """
    tokens = _tokenize(sql)
    select = _Parser(tokens).parse()
    metadata = inspect_file(path)
    if metadata["format_version"] == FORMAT_VERSION_PARTITIONED:
        # Row-group-partitioned file: decode only the referenced columns
        # and the row groups their statistics cannot exclude.
        schema = _schema_from_metadata(metadata)
        return _query_partitioned(path, select, schema, _SINGLE_TABLE_NAME)
    table = read_file(path)
    return _run_query(table, select)


def query_table(table: Table, sql: str) -> Table:
    """Apply ``sql`` (``FROM input``) to an in-memory table."""
    tokens = _tokenize(sql)
    select = _Parser(tokens).parse()
    return _run_query(table, select)


def query_files(sources: Any, sql: str, join_strategy: Any = None) -> Table:
    """Run ``sql`` against the tables named by ``sources``.

    ``sources`` maps table names to columnar file paths; only the tables
    referenced by the statement are read.  The FROM table may be followed by
    a deterministic chain of zero or more equi-joins (``INNER JOIN`` /
    ``LEFT JOIN`` / ``RIGHT JOIN`` / ``FULL OUTER JOIN``); each step joins
    the intermediate result to one freshly introduced table on a single
    ``a.col = b.col`` condition.  See the module docstring for the exact
    grammar and semantics.

    ``join_strategy`` optionally selects the join algorithm applied to
    every step: ``"hash"`` (a right-key hash lookup) or ``"sort_merge"``
    (sort both sides on the join key and merge the runs).  Both strategies
    accept the same statements and return identical columns, values and row
    order; when omitted the implicit default join path is used.  A statement
    without a JOIN accepts either strategy, which then adds no operator and
    changes nothing.  A non-string or otherwise invalid ``join_strategy``
    raises :class:`ValueError` before any file is opened.

    A non-mapping or empty ``sources``, non-string keys or non-path values
    raise :class:`ValueError` before any file is touched.  Lexical and
    grammatical errors (including join-keyword, ON and alias problems)
    raise :class:`QuerySyntaxError` before any file is read.  Unknown or
    duplicate tables, ON conditions that do not connect a new table to an
    earlier one, unknown or unqualified columns, and type-incompatible join
    keys raise :class:`QueryValidationError`; malformed files raise
    :class:`~columnar_analytics.format.ColumnarFormatError`; other I/O
    failures propagate as :class:`OSError`.
    """
    paths, select, strategy, from_key, steps = _resolve_statement(
        sources, sql, join_strategy
    )
    if not steps:
        rewritten = _rewrite_select(select, _single_table_resolver(from_key))
        metadata = inspect_file(paths[from_key])
        if metadata["format_version"] == FORMAT_VERSION_PARTITIONED:
            # Row-group-partitioned file: decode only the referenced
            # columns and the row groups their statistics cannot exclude.
            schema = _schema_from_metadata(metadata)
            return _query_partitioned(
                paths[from_key], rewritten, schema, expected_table=None
            )
        table = read_file(paths[from_key])
        return _run_query(table, rewritten, expected_table=None)

    table_keys = (from_key, *(step.new_key for step in steps))
    # Qualifier checks (unqualified / unknown-table column references) do
    # not need the files and run before any read.
    rewritten = _rewrite_select(select, _multi_table_resolver(table_keys))

    # The partitioned read path applies as soon as one source is a v2
    # file; the version peek only reads the 5-byte prefix and treats an
    # unreadable or unrecognisable prefix as v1, so all-v1 chains keep the
    # historical read order and error behaviour exactly.
    if any(
        _peek_format_version(paths[key]) == FORMAT_VERSION_PARTITIONED
        for key in table_keys
    ):
        return _run_join_chain_partitioned(
            paths, rewritten, from_key, steps, strategy, table_keys
        )

    # Each step takes the materialised intermediate result as its left input
    # and the one freshly introduced table as its right input; WHERE /
    # GROUP BY / HAVING / projection / DISTINCT / ORDER BY / LIMIT run only
    # after the whole chain has been built.  The FROM table is qualified up
    # front so every step sees uniformly "table.column" column names.
    combined = _qualify_table(read_file(paths[from_key]), from_key)
    for step in steps:
        new_table = read_file(paths[step.new_key])
        combined = _execute_join_step(combined, new_table, step, strategy)
    return _run_query(combined, rewritten, expected_table=None)


def _peek_format_version(path: Any) -> int | None:
    """Best-effort read of a file's format version byte (no validation).

    Returns ``None`` for unreadable or unrecognisable files; callers treat
    that as "not v2" so the legacy read path reports the proper error.
    """
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(5)
    except OSError:
        return None
    if len(prefix) < 5 or prefix[:4] != b"CAEF":
        return None
    return prefix[4]


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


def _qualify_table(table: Table, key: str) -> Table:
    """Re-wrap ``table`` naming every column ``key.column`` (values unchanged)."""
    qualified = Schema(
        tuple(
            ColumnSchema(f"{key}.{col.name}", col.type, col.nullable)
            for col in table.schema.columns
        )
    )
    return Table._from_storage(qualified, table._columns)


def _validate_sources(sources: Any) -> dict:
    if not isinstance(sources, Mapping):
        raise ValueError("sources must be a mapping of table names to file paths")
    if not sources:
        raise ValueError("sources must not be empty")
    paths = {}
    for key, value in sources.items():
        if not isinstance(key, str) or not key:
            raise ValueError("sources keys must be non-empty strings")
        if not isinstance(value, (str, os.PathLike)):
            raise ValueError(f"sources[{key!r}] must be a file path")
        paths[key] = value
    return paths


@dataclass(frozen=True)
class _JoinStep:
    """One validated join step, normalised to intermediate-vs-new sides."""

    kind: str  # "inner" | "left" | "right" | "full"
    prior_key: str  # canonical table of the ON side already introduced
    prior_col: str
    new_key: str  # canonical table freshly introduced by this step
    new_col: str


def _resolve_statement(sources: Any, sql: str, join_strategy: Any = None) -> tuple:
    """Validate arguments and resolve the statement's table/ON references.

    No file is ever opened: only argument validation, parsing and name
    resolution run here.  Returns ``(paths, select, strategy, from_key,
    steps)`` with ``steps`` a tuple of :class:`_JoinStep` in FROM/JOIN
    order (empty for a single-table statement).
    """
    strategy = _validate_join_strategy(join_strategy)
    paths = _validate_sources(sources)
    tokens = _tokenize(sql)
    select = _Parser(tokens, allow_join=True).parse()
    all_keys = tuple(paths)
    from_key = _resolve_table_ref(select.table, select.table_quoted, all_keys)
    introduced = [from_key]
    steps: list[_JoinStep] = []
    for join in select.joins:
        new_key = _resolve_table_ref(join.table, join.table_quoted, all_keys)
        if new_key in introduced:
            raise QueryValidationError(f"duplicate table {new_key!r} in join")
        candidates = (*introduced, new_key)
        left_table = _resolve_table_ref(
            join.left_key[0], join.left_key[1], candidates
        )
        right_table = _resolve_table_ref(
            join.right_key[0], join.right_key[1], candidates
        )
        left_is_new = left_table == new_key
        right_is_new = right_table == new_key
        # Exactly one ON side introduces this step's new table; the other
        # must name a table introduced by FROM or an earlier join.
        if left_is_new == right_is_new:
            raise QueryValidationError(
                "ON keys must connect the newly joined table with an earlier table"
            )
        if left_is_new:
            prior_key, prior_col = right_table, join.right_key[2]
            new_col = join.left_key[2]
        else:
            prior_key, prior_col = left_table, join.left_key[2]
            new_col = join.right_key[2]
        steps.append(
            _JoinStep(join.kind, prior_key, prior_col, new_key, new_col)
        )
        introduced.append(new_key)
    return paths, select, strategy, from_key, tuple(steps)


def _referenced_source_paths(sources: Any, sql: str, join_strategy: Any = None) -> tuple:
    """Resolve the file paths of the tables the statement actually reads.

    Parsing-only counterpart of :func:`query_files`: no file is ever opened.
    Returns the paths in ``FROM`` / ``JOIN`` order (one entry for a
    single-table statement, one per introduced table otherwise).  Source and
    join-strategy validation raise :class:`ValueError`; lexical / grammatical
    errors raise :class:`QuerySyntaxError`; unknown or ambiguous table names
    and other join-shape problems raise :class:`QueryValidationError`.
    """
    paths, _select, _strategy, from_key, steps = _resolve_statement(
        sources, sql, join_strategy
    )
    return (paths[from_key], *(paths[step.new_key] for step in steps))


def _resolve_table_ref(name: str, quoted: bool, candidates: tuple) -> str:
    """Resolve a table reference against the available canonical names.

    Quoted references match exactly; bare references match exactly first,
    then case-insensitively when that is unambiguous.
    """
    if name in candidates:
        return name
    if not quoted:
        lowered = name.lower()
        matches = [c for c in candidates if c.lower() == lowered]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise QueryValidationError(f"ambiguous table name {name!r}")
    raise QueryValidationError(f"unknown table {name!r}")


def _single_table_resolver(table_key: str):
    def resolve(table, table_quoted, column):
        if table is not None:
            _resolve_table_ref(table, table_quoted, (table_key,))
        return column

    return resolve


def _multi_table_resolver(table_keys: tuple):
    def resolve(table, table_quoted, column):
        if table is None:
            raise QueryValidationError(
                f"column reference {column!r} must be qualified with a table name"
            )
        key = _resolve_table_ref(table, table_quoted, table_keys)
        return f"{key}.{column}"

    return resolve


def _rewrite_select(select: _Select, resolve) -> _Select:
    """Resolve every column reference to its canonical output name."""
    items = tuple(_rewrite_ref_item(item, resolve) for item in select.items)
    # ORDER BY names that match a SELECT alias stay unresolved here; the
    # binder resolves them against the aliased expressions first.
    aliases = {item.name for item in items if item.kind == "expr"}
    group_by = None
    if select.group_by is not None:
        group_by = tuple(
            (None, False, resolve(table, quoted, name))
            for table, quoted, name in select.group_by
        )
    order_by = None
    if select.order_by is not None:
        order_by = tuple(
            item
            if item.kind == "column" and item.table is None and item.name in aliases
            else _rewrite_ref_item(item, resolve)
            for item in select.order_by
        )
    where = _rewrite_expr(select.where, resolve) if select.where is not None else None
    having = (
        _rewrite_having_expr(select.having, resolve)
        if select.having is not None
        else None
    )
    return _Select(
        items=items,
        table=select.table,
        table_quoted=select.table_quoted,
        where=where,
        group_by=group_by,
        having=having,
        order_by=order_by,
        limit=select.limit,
        joins=select.joins,
        distinct=select.distinct,
    )


def _rewrite_ref_item(item: _RefItem, resolve) -> _RefItem:
    if item.kind == "star":
        return item
    if item.kind == "expr":
        return _RefItem(
            "expr",
            name=item.name,
            expr=_rewrite_expr(item.expr, resolve),
            descending=item.descending,
            nulls_first=item.nulls_first,
        )
    if item.kind == "column":
        return _RefItem(
            "column",
            name=resolve(item.table, item.table_quoted, item.name),
            descending=item.descending,
            nulls_first=item.nulls_first,
        )
    arg = item.arg
    if arg:
        arg = resolve(item.arg_table, item.arg_table_quoted, arg)
    return _RefItem(
        "agg",
        func=item.func,
        arg=arg,
        distinct=item.distinct,
        descending=item.descending,
        nulls_first=item.nulls_first,
    )


def _rewrite_expr(node: tuple, resolve) -> tuple:
    tag = node[0]
    if tag == "literal":
        return node
    if tag == "column":
        return ("column", resolve(node[2], node[3], node[1]), None, False)
    if tag == "unary":
        return ("unary", _rewrite_expr(node[1], resolve), node[2])
    if tag == "arith":
        return ("arith", node[1], _rewrite_expr(node[2], resolve), _rewrite_expr(node[3], resolve))
    if tag == "case":
        return (
            "case",
            tuple(
                (_rewrite_expr(cond, resolve), _rewrite_expr(result, resolve))
                for cond, result in node[1]
            ),
            _rewrite_expr(node[2], resolve) if node[2] is not None else None,
        )
    if tag == "not":
        return ("not", _rewrite_expr(node[1], resolve))
    if tag in ("and", "or"):
        return (tag, _rewrite_expr(node[1], resolve), _rewrite_expr(node[2], resolve))
    if tag == "isnull":
        return ("isnull", _rewrite_expr(node[1], resolve), node[2])
    if tag == "cmp":
        return ("cmp", node[1], _rewrite_expr(node[2], resolve), _rewrite_expr(node[3], resolve))
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


def _rewrite_having_expr(node: tuple, resolve) -> tuple:
    """Rewrite column and aggregate references inside a parsed HAVING tree."""
    tag = node[0]
    if tag == "literal":
        return node
    if tag == "column":
        return ("column", resolve(node[2], node[3], node[1]), None, False)
    if tag == "hagg":
        arg = node[2]
        if arg:
            arg = resolve(node[3], node[4], arg)
        return ("hagg", node[1], arg, None, False, node[5])
    if tag == "not":
        return ("not", _rewrite_having_expr(node[1], resolve))
    if tag in ("and", "or"):
        return (
            tag,
            _rewrite_having_expr(node[1], resolve),
            _rewrite_having_expr(node[2], resolve),
        )
    if tag == "isnull":
        return ("isnull", _rewrite_having_expr(node[1], resolve), node[2])
    if tag == "cmp":
        return (
            "cmp",
            node[1],
            _rewrite_having_expr(node[2], resolve),
            _rewrite_having_expr(node[3], resolve),
        )
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


def _build_joined_schema(
    prior_schema: Schema,
    new_schema: Schema,
    step: _JoinStep,
) -> Schema:
    """Validate one step's ON keys and derive the combined post-step schema.

    Name resolution, key-type compatibility and duplicate result names are
    query-layer concerns and stay here (keeping their
    :class:`QueryValidationError` classification); the resulting column
    layout and outer-join nullability are derived by the shared join layer
    so execution and the explain plan can never disagree.
    """
    prior_name = f"{step.prior_key}.{step.prior_col}"
    try:
        prior_idx = prior_schema.index(prior_name)
    except KeyError:
        raise QueryValidationError(f"unknown column: {prior_name!r}") from None
    try:
        new_idx = new_schema.index(step.new_col)
    except KeyError:
        raise QueryValidationError(
            f"unknown column: {f'{step.new_key}.{step.new_col}'!r}"
        ) from None
    prior_type = prior_schema.columns[prior_idx].type
    new_type = new_schema.columns[new_idx].type
    if prior_type != new_type and not {prior_type, new_type} <= {"int64", "float64"}:
        raise QueryValidationError(
            f"join key types are incompatible: {prior_type} and {new_type}"
        )

    schema = _derive_step_schema(
        prior_schema,
        new_schema,
        new_table=step.new_key,
        kind=step.kind,
    )
    names = [col.name for col in schema.columns]
    if len(set(names)) != len(names):
        raise QueryValidationError("joined tables produce duplicate column names")
    return schema


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


def _row_values(source_columns: tuple[tuple, ...], i: int) -> tuple:
    return tuple(col[i] for col in source_columns)


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


# ---------------------------------------------------------------------------
# EXPLAIN: parse, bind and plan without touching the data section
# ---------------------------------------------------------------------------


def explain_file(path: Any, sql: str) -> dict:
    """Produce the query plan for ``sql`` against one columnar file.

    Like :func:`query_file` for parsing and binding, but only the file
    metadata is read: the data section is never read, decompressed or
    decoded, so data CRCs and value-level statistics are not verified.
    The returned value is a JSON-serialisable ordered dict with the fixed
    top-level keys ``sources``, ``operators`` and ``output``.

    :class:`QuerySyntaxError` is raised before the file is touched;
    binding problems raise :class:`QueryValidationError`; malformed
    metadata or a declared-size mismatch raise
    :class:`~columnar_analytics.format.ColumnarFormatError`; other I/O
    failures propagate as :class:`OSError`.
    """
    tokens = _tokenize(sql)
    select = _Parser(tokens).parse()
    metadata = inspect_file(path)
    schema = _schema_from_metadata(metadata)
    sources = ((_SINGLE_TABLE_NAME, metadata, schema),)
    bound = _bind_select(select, schema, expected_table=_SINGLE_TABLE_NAME)
    scan_extras = None
    if metadata["format_version"] == FORMAT_VERSION_PARTITIONED:
        info = _scan_pushdown_info(inspect_row_groups(path), bound, schema)
        scan_extras = {_SINGLE_TABLE_NAME: _scan_pushdown_fields(info)}
    return _build_explain(
        sources, schema, select, bound, steps=(), scan_extras=scan_extras
    )


def explain_files(sources: Any, sql: str, join_strategy: Any = None) -> dict:
    """Produce the query plan for ``sql`` against the mapped tables.

    Like :func:`query_files` for parsing, source resolution and binding,
    but only the metadata of the referenced files is read: unreferenced
    sources are never opened and no data section is ever touched.  The
    returned value is a JSON-serialisable ordered dict with the fixed
    top-level keys ``sources``, ``operators`` and ``output``.

    ``join_strategy`` accepts the same ``"hash"`` / ``"sort_merge"``
    values as :func:`query_files`; an explicit strategy applies to every
    join step and each of the plan's ``Join`` operators carries a
    ``strategy`` field (``HASH`` / ``SORT_MERGE``) matching the strategy
    the query and export entries would use.  Sources and Scans are listed
    in FROM/JOIN order, one Join operator follows per step, and each
    source's ``required_columns`` cover only its own referenced columns.
    In a single-source statement or an all-INNER chain every v2 Scan
    additionally carries, right after ``required_columns``,
    ``row_groups_total``, ``row_groups_selected`` and
    ``pushed_condition`` (the statistics-pushed leaves attributed to that
    source, combined in SQL appearance order with qualified names, or
    null; without a pushed leaf selected equals total); v1 Scans and the
    Scans of a chain containing an OUTER step keep their three-field
    shape, and the counts mirror exactly the groups execution would read.
    A statement without a JOIN accepts either strategy but gains no join
    operator.  An invalid ``join_strategy`` raises :class:`ValueError`
    before any file is touched.

    A non-mapping or empty ``sources``, non-string keys or non-path
    values raise :class:`ValueError` before any file is touched;
    :class:`QuerySyntaxError` is raised before files are read as well.
    Unknown tables or columns and other binding problems raise
    :class:`QueryValidationError`; malformed metadata or a declared-size
    mismatch raise :class:`~columnar_analytics.format.ColumnarFormatError`;
    other I/O failures propagate as :class:`OSError`.
    """
    paths, select, strategy, from_key, steps = _resolve_statement(
        sources, sql, join_strategy
    )
    table_keys = (from_key, *(step.new_key for step in steps))
    if not steps:
        rewritten = _rewrite_select(select, _single_table_resolver(from_key))
        from_metadata = inspect_file(paths[from_key])
        from_schema = _schema_from_metadata(from_metadata)
        sources = ((from_key, from_metadata, from_schema),)
        bound = _bind_select(rewritten, from_schema, expected_table=None)
        scan_extras = None
        if from_metadata["format_version"] == FORMAT_VERSION_PARTITIONED:
            info = _scan_pushdown_info(
                inspect_row_groups(paths[from_key]), bound, from_schema
            )
            scan_extras = {from_key: _scan_pushdown_fields(info)}
        return _build_explain(
            sources, from_schema, rewritten, bound, steps=(), scan_extras=scan_extras
        )

    # Qualifier checks (unqualified / unknown-table column references) do
    # not need the files and run before any metadata is read.
    rewritten = _rewrite_select(select, _multi_table_resolver(table_keys))

    # Metadata is read for the referenced sources only, in FROM/JOIN order;
    # each step validates its ON keys against the intermediate schema and
    # derives the combined schema exactly as execution would.
    source_entries = []
    from_metadata = inspect_file(paths[from_key])
    current_schema = _schema_from_metadata(from_metadata)
    source_entries.append((from_key, from_metadata, current_schema))
    current_schema = Schema(
        tuple(
            ColumnSchema(f"{from_key}.{col.name}", col.type, col.nullable)
            for col in current_schema.columns
        )
    )
    for step in steps:
        metadata = inspect_file(paths[step.new_key])
        new_schema = _schema_from_metadata(metadata)
        source_entries.append((step.new_key, metadata, new_schema))
        current_schema = _build_joined_schema(current_schema, new_schema, step)

    bound = _bind_select(rewritten, current_schema, expected_table=None)

    # Statistics pushdown is restricted to all-INNER chains; a chain with
    # any OUTER step keeps every Scan operator's historical shape.  In an
    # eligible chain each v2 Scan reports its group counts and the leaves
    # attributed to that source (selected == total, pushed_condition null
    # when no leaf qualifies); v1 Scans stay unchanged.
    scan_extras = None
    if all(step.kind == "inner" for step in steps):
        bare_schemas = {key: source_schema for key, _m, source_schema in source_entries}
        spans = _source_column_spans(bare_schemas, table_keys)
        leaves_by_source = _pushable_leaves_by_source(
            bound["where"], spans, table_keys
        )
        scan_extras = {}
        for key, metadata, _source_schema in source_entries:
            if metadata["format_version"] == FORMAT_VERSION_PARTITIONED:
                groups = inspect_row_groups(paths[key])
                info = _source_scan_info(groups, leaves_by_source[key], spans[key])
                scan_extras[key] = _scan_pushdown_fields(info)
    return _build_explain(
        tuple(source_entries),
        current_schema,
        rewritten,
        bound,
        steps=steps,
        strategy=strategy,
        scan_extras=scan_extras,
    )


_SINGLE_TABLE_NAME = "input"


def _schema_from_metadata(metadata: Mapping) -> Schema:
    return Schema(
        tuple(
            ColumnSchema(entry["name"], entry["type"], entry["nullable"])
            for entry in metadata["columns"]
        )
    )


def _source_description(key: str, metadata: Mapping) -> dict:
    return {
        "name": key,
        "row_count": metadata["row_count"],
        "columns": [
            {
                "name": entry["name"],
                "type": entry["type"],
                "nullable": entry["nullable"],
            }
            for entry in metadata["columns"]
        ],
    }


def _collect_expr_columns(node: _Expr, indices: set) -> None:
    """Add every source-column index referenced by one bound expression.

    One traversal for WHERE, SELECT expressions, ORDER BY aliases and the
    HAVING tree: the node hierarchy itself owns the walk.  HAVING
    aggregate leaves reference no source column here -- their arguments
    are scanned separately through the aggregate registry.
    """
    indices.update(node.columns())


def _collect_required_indices(bound: Mapping) -> set:
    """Combined-schema column indices referenced anywhere in the query."""
    indices: set = set(bound.get("group_indices", ()))
    if bound["where"] is not None:
        _collect_expr_columns(bound["where"], indices)
    for item in bound["items"]:
        if item.kind == "column":
            indices.add(item.col_index)
        elif item.kind == "agg":
            if item.arg_index >= 0:
                indices.add(item.arg_index)
        else:  # "expr"
            _collect_expr_columns(item.expr, indices)
    # Aggregates introduced only by HAVING still have to be scanned.
    for item in bound.get("aggregates", ()):
        if item.arg_index >= 0:
            indices.add(item.arg_index)
    if bound.get("having") is not None:
        _collect_expr_columns(bound["having"], indices)
    for entry in bound["order_by"] or ():
        if entry[0] == "col":
            indices.add(entry[1])
        elif entry[0] == "expr":
            _collect_expr_columns(entry[1], indices)
        # Aggregate ORDER BY entries index a selected output column, which
        # is already covered by the projection scan above.
    return indices


def _expr_json(node: _Expr) -> dict:
    """Render a bound expression as its explain-plan JSON tree.

    Every bound node -- WHERE conditions, SELECT expressions and the HAVING
    tree including its aggregate leaves -- owns its rendering via
    :meth:`_Expr.to_json`, so plan JSON and the executable expression can
    never disagree.
    """
    return node.to_json()


def _aggregate_json(item, schema: Schema) -> dict:
    """Render one bound aggregate of the Aggregate operator as JSON.

    DISTINCT aggregates gain a ``distinct: true`` field; plain aggregates
    keep the historical three-field shape.
    """
    entry = {
        "function": item.func,
        "argument": (
            None if item.arg_index < 0 else schema.columns[item.arg_index].name
        ),
        "output": item.output_name,
    }
    if item.distinct:
        entry["distinct"] = True
    return entry


def _build_explain(
    sources: tuple,
    schema: Schema,
    select: _Select,
    bound: Mapping,
    steps: tuple[_JoinStep, ...] = (),
    strategy: str | None = None,
    scan_extras: Mapping | None = None,
) -> dict:
    referenced = _collect_required_indices(bound)
    for step in steps:
        # The ON keys feed the join even when neither is projected.
        referenced.add(schema.index(f"{step.prior_key}.{step.prior_col}"))
        referenced.add(schema.index(f"{step.new_key}.{step.new_col}"))

    operators: list = []
    referenced_names = {schema.columns[i].name for i in referenced}
    for key, _metadata, source_schema in sources:
        if not steps:
            required = [col.name for col in schema.columns if col.name in referenced_names]
        else:
            prefix = f"{key}."
            required = [
                col.name
                for col in source_schema.columns
                if f"{prefix}{col.name}" in referenced_names
            ]
        scan_operator = {"operator": "Scan", "source": key, "required_columns": required}
        if scan_extras is not None and key in scan_extras:
            # An eligible v2 (row-group-partitioned) scan reports the
            # statistics pushdown right after required_columns: total /
            # selected row groups and the pushed condition tree (null when
            # nothing was pushed).  v1 scans and scans of chains containing
            # an OUTER step keep their historical three-field shape.
            scan_operator.update(scan_extras[key])
        operators.append(scan_operator)

    # One Join operator per step, in FROM/JOIN order; each reports the two
    # qualified ON keys normalised to prior-table (intermediate) left and
    # freshly introduced table right.  An explicitly requested strategy is
    # reported on every Join operator; the implicit path (join_strategy
    # omitted) keeps the operator shape with no strategy field.
    for step in steps:
        join_operator = {
            "operator": "Join",
            "type": step.kind.upper(),
            "left": {"table": step.prior_key, "column": step.prior_col},
            "right": {"table": step.new_key, "column": step.new_col},
        }
        if strategy is not None:
            join_operator["strategy"] = _strategy_label(strategy)
        operators.append(join_operator)

    if bound["where"] is not None:
        operators.append({"operator": "Filter", "condition": _expr_json(bound["where"])})

    if bound["mode"] == "aggregate":
        # The Aggregate node computes every distinct aggregate the SELECT,
        # ORDER BY or HAVING needs, in first-reference order.  Only
        # DISTINCT aggregates carry a "distinct" field; plain aggregate
        # entries keep their historical shape.
        operators.append(
            {
                "operator": "Aggregate",
                "group_keys": [
                    schema.columns[index].name for index in bound["group_indices"]
                ],
                "aggregates": [
                    _aggregate_json(item, schema)
                    for item in bound["aggregates"]
                ],
            }
        )
        if bound["having"] is not None:
            operators.append(
                {
                    "operator": "Having",
                    "condition": _expr_json(bound["having"]),
                }
            )

    projections = []
    for item in bound["items"]:
        if item.kind == "column":
            expression = {"kind": "column", "name": schema.columns[item.col_index].name}
        elif item.kind == "agg":
            expression = {
                "kind": "aggregate",
                "function": item.func,
                "argument": (
                    None
                    if item.arg_index < 0
                    else schema.columns[item.arg_index].name
                ),
            }
            if item.distinct:
                expression["distinct"] = True
        else:  # "expr"
            expression = _expr_json(item.expr)
        projections.append({"expression": expression, "output": item.output_name})
    project_operator = {"operator": "Project", "expressions": projections}

    if bound.get("distinct"):
        # DISTINCT evaluates the projection right after the scan/join/filter
        # stages, then deduplicates the result rows before Sort and Limit.
        operators.append(project_operator)
        operators.append(
            {
                "operator": "Distinct",
                "keys": [item.output_name for item in bound["items"]],
            }
        )
        if bound["order_by"] is not None:
            keys = []
            for entry in bound["order_by"]:
                # DISTINCT ORDER BY entries index projected output columns.
                name = bound["items"][entry[1]].output_name
                keys.append(
                    {
                        "column": name,
                        "direction": "DESC" if entry[2] else "ASC",
                        "nulls": "FIRST" if entry[3] else "LAST",
                    }
                )
            operators.append({"operator": "Sort", "keys": keys})
        if select.limit is not None:
            operators.append({"operator": "Limit", "count": select.limit})
    else:
        if bound["order_by"] is not None:
            alias_by_expr = {
                id(item.expr): item.output_name
                for item in bound["items"]
                if item.kind == "expr"
            }
            keys = []
            for entry in bound["order_by"]:
                if entry[0] == "col":
                    name = schema.columns[entry[1]].name
                    descending, nulls_first = entry[2], entry[3]
                elif entry[0] == "expr":
                    name = alias_by_expr[id(entry[1])]
                    descending, nulls_first = entry[2], entry[3]
                else:  # aggregate-query ORDER BY indexes a selected output column
                    name = bound["items"][entry[0]].output_name
                    descending, nulls_first = entry[1], entry[2]
                keys.append(
                    {
                        "column": name,
                        "direction": "DESC" if descending else "ASC",
                        "nulls": "FIRST" if nulls_first else "LAST",
                    }
                )
            operators.append({"operator": "Sort", "keys": keys})

        if select.limit is not None:
            operators.append({"operator": "Limit", "count": select.limit})
        operators.append(project_operator)

    output = [
        {
            "name": item.output_name,
            "type": item.out_type,
            "nullable": item.nullable,
        }
        for item in bound["items"]
    ]

    return {
        "sources": [_source_description(key, metadata) for key, metadata, _ in sources],
        "operators": operators,
        "output": output,
    }
