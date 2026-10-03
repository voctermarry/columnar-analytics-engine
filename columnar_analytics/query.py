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

The explain plan lists the referenced sources and their Scan operators in
FROM/JOIN order with ``required_columns`` attributed to each source, then
one Join operator per step in the same order, recording the step type and
the two normalised qualified keys (the earlier table on the left, the
freshly introduced table on the right).
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
    FORMAT_VERSION_V2,
    ColumnSchema,
    Schema,
    Table,
    inspect_file,
    inspect_row_groups,
    read_file,
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

# Optional join strategies for multi-file (join-chain) queries.  ``None``
# keeps the implicit historical path (a right-key hash lookup).  ``HASH`` is that same path
# requested explicitly; ``SORT_MERGE`` sorts both sides on the join key and
# merges the runs.  Both produce byte-identical results; only the explicit
# spellings surface in the explain plan.
_JOIN_STRATEGIES = frozenset(("hash", "sort_merge"))
_JOIN_STRATEGY_LABELS = {"hash": "HASH", "sort_merge": "SORT_MERGE"}


def _validate_join_strategy(join_strategy: Any) -> str | None:
    """Validate the optional ``join_strategy`` argument.

    Returns the canonical lowercase strategy name, or ``None`` when the
    caller did not request one.  Any other value (including a non-string)
    raises :class:`ValueError`; callers run this before opening any file.
    """
    if join_strategy is None:
        return None
    if not isinstance(join_strategy, str) or join_strategy not in _JOIN_STRATEGIES:
        allowed = ", ".join(sorted(_JOIN_STRATEGIES))
        raise ValueError(
            f"invalid join_strategy: {join_strategy!r}; expected one of {allowed}"
        )
    return join_strategy


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
# Node tuples:
#   ("literal", python_value, type_name)
#   ("column", name, table|None, table_quoted)        -- as parsed
#   ("column", name, index, type_name, nullable)      -- after binding
#   ("arith", op, left_node, right_node)              -- as parsed (+ - * /)
#   ("arith", op, left_node, right_node, type_name)   -- after binding
#   ("unary", operand, negate)                        -- as parsed
#   ("unary", operand, negate, type_name)             -- after binding
#   ("case", ((cond, result), ...), else_node|None)   -- as parsed
#   ("case", ((cond, result), ...), else_node|None, type_name) -- after binding
#   ("cmp", op, left_node, right_node)
#   ("isnull", operand, negate)
#   ("not", operand)
#   ("and"|"or", left, right)
# Predicate nodes are boolean-typed (three-valued at evaluation time);
# "literal"/"column"/"arith"/"unary" nodes are value nodes.
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
    if where is not None and where[0] in (
        "literal",
        "column",
        "arith",
        "unary",
        "case",
    ):
        type_name = _expr_type(where)
        if type_name != "bool":
            raise QueryValidationError(
                f"WHERE clause must be boolean, got {type_name}"
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
                out_type = _expr_type(expr)
                # Arithmetic scalar expressions stay numeric-only; a CASE
                # expression may additionally yield bool or utf8 results
                # (its own WHEN/ELSE type consistency is checked at binding).
                if expr[0] == "case":
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
                        nullable=_expr_nullable(expr),
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
        _require_having_boolean(having, "HAVING clause")

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
) -> tuple:
    tag = node[0]
    if tag == "literal":
        return node
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
        return ("column", name, col_index, col.type, col.nullable)
    if tag == "hagg":
        slot = require_aggregate(node[1], node[2], node[5])
        agg_item = agg_order[slot]
        return ("hagg", slot, agg_item.out_type, agg_item.nullable)
    if tag == "not":
        operand = _bind_having(
            node[1], schema, group_index_set, agg_order, require_aggregate
        )
        _require_having_boolean(operand, "NOT")
        return ("not", operand)
    if tag in ("and", "or"):
        left = _bind_having(
            node[1], schema, group_index_set, agg_order, require_aggregate
        )
        right = _bind_having(
            node[2], schema, group_index_set, agg_order, require_aggregate
        )
        _require_having_boolean(left, tag.upper())
        _require_having_boolean(right, tag.upper())
        return (tag, left, right)
    if tag == "isnull":
        operand = _bind_having(
            node[1], schema, group_index_set, agg_order, require_aggregate
        )
        if operand[0] not in ("literal", "column", "hagg"):
            raise QuerySyntaxError(
                "IS NULL operand must be a group column, an aggregate or a literal"
            )
        return ("isnull", operand, node[2])
    if tag == "cmp":
        op = node[1]
        left = _bind_having(
            node[2], schema, group_index_set, agg_order, require_aggregate
        )
        right = _bind_having(
            node[3], schema, group_index_set, agg_order, require_aggregate
        )
        if left[0] not in ("literal", "column", "hagg") or right[0] not in (
            "literal",
            "column",
            "hagg",
        ):
            raise QuerySyntaxError(
                "comparison operands must be group columns, aggregates or literals"
            )
        left_t = _having_value_type(left)
        right_t = _having_value_type(right)
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
        return ("cmp", op, left, right)
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


def _having_value_type(node: tuple) -> str:
    """The static value type of a bound HAVING leaf."""
    tag = node[0]
    if tag == "literal":
        return node[2]
    if tag == "column":
        return node[3]
    if tag == "hagg":
        return node[2]
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


def _require_having_boolean(node: tuple, context: str) -> None:
    if node[0] in ("cmp", "isnull", "not", "and", "or"):
        return
    type_name = _having_value_type(node)
    if type_name != "bool":
        raise QueryValidationError(
            f"{context} requires a boolean operand, got {type_name}"
        )


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


def _bind_expr(node: tuple, schema: Schema) -> tuple:
    tag = node[0]
    if tag == "literal":
        return node
    if tag == "column":
        name = node[1]
        try:
            index = schema.index(name)
        except KeyError:
            raise QueryValidationError(f"unknown column: {name!r}") from None
        col = schema.columns[index]
        return ("column", name, index, col.type, col.nullable)
    if tag == "not":
        operand = _bind_expr(node[1], schema)
        _require_boolean(operand, "NOT")
        return ("not", operand)
    if tag in ("and", "or"):
        left = _bind_expr(node[1], schema)
        right = _bind_expr(node[2], schema)
        _require_boolean(left, tag.upper())
        _require_boolean(right, tag.upper())
        return (tag, left, right)
    if tag == "isnull":
        operand = _bind_expr(node[1], schema)
        if operand[0] not in ("literal", "column", "arith", "unary", "case"):
            raise QuerySyntaxError(
                "IS NULL operand must be a column reference or a literal"
            )
        return ("isnull", operand, node[2])
    if tag == "unary":
        operand = _bind_expr(node[1], schema)
        type_name = _expr_type(operand)
        if type_name not in ("int64", "float64"):
            raise QueryValidationError(
                f"unary +/- requires a numeric operand, got {type_name}"
            )
        return ("unary", operand, node[2], type_name)
    if tag == "arith":
        left = _bind_expr(node[2], schema)
        right = _bind_expr(node[3], schema)
        left_t = _expr_type(left)
        right_t = _expr_type(right)
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
        return ("arith", op, left, right, out_type)
    if tag == "case":
        bound_branches = []
        result_types: list[str] = []
        for cond_node, result_node in node[1]:
            cond = _bind_expr(cond_node, schema)
            _require_boolean(cond, "WHEN")
            result = _bind_expr(result_node, schema)
            bound_branches.append((cond, result))
            result_types.append(_expr_type(result))
        else_bound = None
        if node[2] is not None:
            else_bound = _bind_expr(node[2], schema)
            result_types.append(_expr_type(else_bound))
        out_type = _unify_case_types(result_types)
        return ("case", tuple(bound_branches), else_bound, out_type)
    if tag == "cmp":
        op = node[1]
        left = _bind_expr(node[2], schema)
        right = _bind_expr(node[3], schema)
        if left[0] not in (
            "literal", "column", "arith", "unary", "case"
        ) or right[0] not in ("literal", "column", "arith", "unary", "case"):
            raise QuerySyntaxError("comparison operands must be column references or literals")
        left_t = _expr_type(left)
        right_t = _expr_type(right)
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
        return ("cmp", op, left, right)
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


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


def _expr_type(node: tuple) -> str:
    """The static type of a bound value node."""
    tag = node[0]
    if tag == "literal":
        return node[2]
    if tag == "column":
        return node[3]
    if tag == "unary":
        return node[3]
    if tag == "arith":
        return node[4]
    if tag == "case":
        return node[3]
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


def _expr_nullable(node: tuple) -> bool:
    """Whether a bound value node can yield NULL (derived from its columns)."""
    tag = node[0]
    if tag == "literal":
        return False
    if tag == "column":
        return node[4]
    if tag == "unary":
        return _expr_nullable(node[1])
    if tag == "arith":
        return _expr_nullable(node[2]) or _expr_nullable(node[3])
    if tag == "case":
        # The value is NULL only when a reachable result yields NULL, or
        # every WHEN fails/UNKNOWN and no ELSE was given (implicit NULL).
        # A NULL condition merely routes the row elsewhere and never makes
        # a non-NULL result nullable.
        if node[2] is None:
            return True
        return any(_expr_nullable(result) for _cond, result in node[1]) or _expr_nullable(
            node[2]
        )
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


def _require_boolean(node: tuple, context: str) -> None:
    if node[0] in ("cmp", "isnull", "not", "and", "or"):
        return
    type_name = _expr_type(node)
    if type_name != "bool":
        raise QueryValidationError(f"{context} requires a boolean operand, got {type_name}")


# ---------------------------------------------------------------------------
# Three-valued-logic evaluation
# ---------------------------------------------------------------------------


def _eval(node: tuple, row: tuple) -> bool | None:
    tag = node[0]
    if tag == "literal":
        return node[1]
    if tag == "column":
        return row[node[2]]
    if tag == "unary":
        value = _eval(node[1], row)
        if value is None:
            return None
        if not node[2]:  # unary plus keeps the value
            return value
        return _negate_value(value)
    if tag == "arith":
        left = _eval(node[2], row)
        right = _eval(node[3], row)
        if left is None or right is None:
            return None
        return _arith_value(node[1], left, right)
    if tag == "case":
        # Conditions are tried in written order; only TRUE matches.  FALSE
        # and UNKNOWN fall through, so unhit results are never evaluated:
        # their division-by-zero, int64 overflow or non-finite float64
        # cannot raise.  The chosen result alone is evaluated.
        for condition, result in node[1]:
            if _eval(condition, row) is True:
                value = _eval(result, row)
                break
        else:
            value = None if node[2] is None else _eval(node[2], row)
        if value is None:
            return None
        # int64/float64 results unify to float64; ints become floats.
        if node[3] == "float64" and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        return value
    if tag == "isnull":
        value = _eval(node[1], row)
        result = value is None
        return (not result) if node[2] else result
    if tag == "not":
        value = _eval(node[1], row)
        return None if value is None else (not value)
    if tag in ("and", "or"):
        # Short-circuit SQL semantics; UNKNOWN propagates only when needed.
        if tag == "and":
            left = _eval(node[1], row)
            if left is False:
                return False
            right = _eval(node[2], row)
            if right is False:
                return False
            if left is None or right is None:
                return None
            return True
        left = _eval(node[1], row)
        if left is True:
            return True
        right = _eval(node[2], row)
        if right is True:
            return True
        if left is None or right is None:
            return None
        return False
    if tag == "cmp":
        left = _eval(node[2], row)
        right = _eval(node[3], row)
        if left is None or right is None:
            return None
        op = node[1]
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
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover


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

    Version 2 files decode only the columns the plan references and skip
    row groups whose statistics prove the WHERE condition impossible.
    """
    tokens = _tokenize(sql)
    select = _Parser(tokens).parse()
    return _scan_and_run(path, select, expected_table=_SINGLE_TABLE_NAME)


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
        return _scan_and_run(
            paths[from_key], rewritten, expected_table=None
        )

    table_keys = (from_key, *(step.new_key for step in steps))
    # Qualifier checks (unqualified / unknown-table column references) do
    # not need the files and run before any read.
    rewritten = _rewrite_select(select, _multi_table_resolver(table_keys))

    # Metadata resolves every referenced source's schema and version without
    # touching any data bytes.
    source_entries = []
    from_metadata = inspect_file(paths[from_key])
    current_schema = _schema_from_metadata(from_metadata)
    source_entries.append((from_key, paths[from_key], from_metadata, current_schema))
    current_schema = Schema(
        tuple(
            ColumnSchema(f"{from_key}.{col.name}", col.type, col.nullable)
            for col in current_schema.columns
        )
    )
    for step in steps:
        metadata = inspect_file(paths[step.new_key])
        new_schema = _schema_from_metadata(metadata)
        source_entries.append((step.new_key, paths[step.new_key], metadata, new_schema))
        current_schema = _build_joined_schema(current_schema, new_schema, step)

    # Binding uses only the combined metadata schema; every source is then scanned
    # (v1 fully, v2 with a projection of its referenced columns).  Row-group
    # statistics are never pushed into a join: the WHERE runs over the joined rows.
    bound = _bind_select(rewritten, current_schema, expected_table=None)
    referenced = _collect_required_indices(bound)
    for step in steps:
        referenced.add(current_schema.index(f"{step.prior_key}.{step.prior_col}"))
        referenced.add(current_schema.index(f"{step.new_key}.{step.new_col}"))
    referenced_names = {current_schema.columns[i].name for i in referenced}

    def scan_source(key: str, path: Any, metadata: Mapping, source_schema: Schema) -> Table:
        if metadata["format_version"] != FORMAT_VERSION_V2:
            return read_file(path)
        prefix = f"{key}."
        local_required = [
            col.name
            for col in source_schema.columns
            if f"{prefix}{col.name}" in referenced_names
        ]
        return _scan_one_source(path, metadata, source_schema, local_required)

    from_schema = _schema_from_metadata(from_metadata)
    combined = _qualify_table(
        scan_source(from_key, paths[from_key], from_metadata, from_schema), from_key
    )
    for key, path, metadata, new_schema in source_entries[1:]:
        step = next(s for s in steps if s.new_key == key)
        new_table = scan_source(key, path, metadata, new_schema)
        combined = _execute_join_step(combined, new_table, step, strategy)
    return _run_bound(combined, rewritten, bound)


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

    The intermediate (prior) schema already names its columns
    ``table.column``; the freshly introduced table still carries its bare
    schema and its columns are prefixed with the new table name here.
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

    # The combined schema keeps the prior columns (already qualified) and
    # then appends the new table's columns named "new_table.column".  An
    # outer side whose unmatched rows are padded with NULL gains nullable
    # result columns: the intermediate is the join's left side, the new
    # table its right side; LEFT pads the right side, RIGHT pads the left
    # side, FULL pads both; INNER keeps the existing nullability.
    pad_left = step.kind in ("right", "full")
    pad_right = step.kind in ("left", "full")
    combined_columns = [
        ColumnSchema(
            col.name,
            col.type,
            True if pad_left else col.nullable,
        )
        for col in prior_schema.columns
    ] + [
        ColumnSchema(
            f"{step.new_key}.{col.name}",
            col.type,
            True if pad_right else col.nullable,
        )
        for col in new_schema.columns
    ]
    names = [col.name for col in combined_columns]
    if len(set(names)) != len(names):
        raise QueryValidationError("joined tables produce duplicate column names")
    return Schema(combined_columns)


def _execute_join_step(
    prior: Table,
    new: Table,
    step: _JoinStep,
    strategy: str | None = None,
) -> Table:
    schema = _build_joined_schema(prior.schema, new.schema, step)
    prior_idx = prior.schema.index(f"{step.prior_key}.{step.prior_col}")
    new_idx = new.schema.index(step.new_col)
    if strategy == "sort_merge":
        columns = _sort_merge_join_columns(prior, new, prior_idx, new_idx, step.kind)
    else:
        columns = _join_columns(prior, new, prior_idx, new_idx, step.kind)
    return Table._from_storage(schema, columns)


def _join_columns(
    left: Table, right: Table, left_idx: int, right_idx: int, kind: str
) -> list:
    """Hash equi-join preserving the required file row order."""
    left_cols = left._columns
    right_cols = right._columns
    # NULL keys never match, so they stay out of the right-side index.
    index: dict[Any, list[int]] = {}
    for j, key in enumerate(right_cols[right_idx]):
        if key is not None:
            index.setdefault(key, []).append(j)
    matches_by_left: dict[int, list[int]] = {}
    matches_by_right: dict[int, list[int]] = {}
    # Left rows are scanned in file order so the inverse lists keep left
    # file order for the RIGHT / FULL unmatched-right bookkeeping.
    for i, key in enumerate(left_cols[left_idx]):
        if key is None:
            continue
        matches = index.get(key)
        if matches:
            matches_by_left[i] = matches
            for j in matches:
                matches_by_right.setdefault(j, []).append(i)
    return _emit_join_columns(
        left, right, matches_by_left, matches_by_right, kind
    )


def _sort_merge_join_columns(
    left: Table, right: Table, left_idx: int, right_idx: int, kind: str
) -> list:
    """Sort-merge equi-join.

    Both sides are stably sorted by the join key (NULLs excluded, they can
    never match) and equal-key runs are merged.  The matches are emitted by
    the shared :func:`_emit_join_columns` path, so the result is identical
    to :func:`_join_columns`; the two paths differ only in how the matches
    are found.
    """
    left_cols = left._columns
    right_cols = right._columns
    left_keys = left_cols[left_idx]
    right_keys = right_cols[right_idx]
    left_order = sorted(
        (i for i in range(left.row_count) if left_keys[i] is not None),
        key=lambda i: left_keys[i],
    )
    right_order = sorted(
        (j for j in range(right.row_count) if right_keys[j] is not None),
        key=lambda j: right_keys[j],
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

    # The inverse map lists matching left rows in left-file order; walking
    # left rows in order (rather than the sorted order) preserves it.
    matches_by_right: dict[int, list[int]] = {}
    for i in range(left.row_count):
        for j in matches_by_left.get(i, ()):
            matches_by_right.setdefault(j, []).append(i)

    return _emit_join_columns(
        left, right, matches_by_left, matches_by_right, kind
    )


def _emit_join_columns(
    left: Table,
    right: Table,
    matches_by_left: dict,
    matches_by_right: dict,
    kind: str,
) -> list:
    """Expand match maps into joined columns for every supported join kind.

    INNER / LEFT / FULL are driven by left-file order (matched combinations
    first, then unmatched left rows), and FULL additionally appends the
    unmatched right rows in right-file order.  RIGHT is driven by right-file
    order, with each right row's combinations expanded in left-file order.
    Padding rows carry NULLs on the unmatched side.
    """
    left_cols = left._columns
    right_cols = right._columns
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
        for j in range(right.row_count):
            left_matches = matches_by_right.get(j)
            if left_matches:
                for i in left_matches:
                    emit_match(i, j)
            else:
                emit_unmatched_right(j)
        return out

    for i in range(left.row_count):
        right_matches = matches_by_left.get(i)
        if right_matches:
            for j in right_matches:
                emit_match(i, j)
        elif kind in ("left", "full"):
            emit_unmatched_left(i)
    if kind == "full":
        for j in range(right.row_count):
            if j not in matches_by_right:
                emit_unmatched_right(j)
    return out


def _run_query(table: Table, select: _Select, expected_table="input") -> Table:
    bound = _bind_select(select, table.schema, expected_table)
    return _run_bound(table, select, bound)


def _run_bound(table: Table, select: _Select, bound: Mapping) -> Table:
    source_columns = table._columns
    row_count = table.row_count
    where = bound["where"]

    if where is None:
        selected = list(range(row_count))
    else:
        selected = [
            i
            for i in range(row_count)
            if _eval(where, tuple(col[i] for col in source_columns)) is True
        ]

    if bound["mode"] == "plain":
        return _run_plain(table, select, bound, selected)
    return _run_aggregate(table, select, bound, selected)


# ---------------------------------------------------------------------------
# Version 2 scans: column projection and row-group statistics pushdown
# ---------------------------------------------------------------------------


def _scan_and_run(path: Any, select: _Select, expected_table) -> Table:
    """Scan one file for ``select``, honouring v2 projection/pruning.

    The plan is bound against the file metadata before any data byte is read.
    Version 1 files scan fully; version 2 files decode only the referenced
    columns and only the row groups whose statistics cannot prove the WHERE
    clause impossible.  The remaining (still row-by-row) filtering runs with
    the unchanged engine so results are identical in every case.
    """
    metadata = inspect_file(path)
    schema = _schema_from_metadata(metadata)
    if metadata["format_version"] != FORMAT_VERSION_V2:
        # Legacy v1 path keeps its original order exactly: the whole file is read
        # (and fully validated) before binding/execution, so a corrupt data
        # section surfaces as ColumnarFormatError ahead of any binding error.
        return _run_query(read_file(path), select, expected_table)
    bound = _bind_select(select, schema, expected_table)

    groups = inspect_row_groups(path)["row_groups"]
    kept, _predicates = _select_row_groups(bound["where"], schema, groups)
    table = _read_v2_scan(path, schema, bound, kept)
    return _run_bound(table, select, bound)


def _read_v2_scan(path: Any, schema: Schema, bound: Mapping, kept_groups: list[int]) -> Table:
    """Read only the plan-referenced columns of the selected v2 row groups.

    The returned table keeps the full schema: unreferenced columns are
    NULL placeholders that the already-bound plan never evaluates.
    """
    required = _collect_required_indices(bound)
    ordered_names = [
        col.name for index, col in enumerate(schema.columns) if index in required
    ]
    projected = read_file(path, columns=ordered_names, row_groups=kept_groups)
    return _restore_full_schema(schema, ordered_names, projected)


def _restore_full_schema(
    schema: Schema, read_names: Sequence[str], projected: Table
) -> Table:
    """Pad a projected v2 scan with NULL placeholders for unread columns."""
    row_count = projected.row_count
    decoded = {name: projected._columns[i] for i, name in enumerate(read_names)}
    full_columns = [decoded.get(col.name, (None,) * row_count) for col in schema.columns]
    return Table._from_storage(schema, full_columns)


def _scan_one_source(
    path: Any, metadata: Mapping, schema: Schema, required_names: Sequence[str]
) -> Table:
    """Scan one join source: full read for v1, projected read for v2."""
    if metadata["format_version"] != FORMAT_VERSION_V2:
        return read_file(path)
    projected = read_file(path, columns=list(required_names))
    return _restore_full_schema(schema, required_names, projected)


def _and_conjuncts(node: tuple | None) -> list[tuple]:
    """Flatten the top-level AND chain of a bound predicate tree."""
    if node is None:
        return []
    if node[0] == "and":
        return _and_conjuncts(node[1]) + _and_conjuncts(node[2])
    return [node]


def _pushdown_predicates(where: tuple | None) -> list[tuple]:
    """The AND-conjoined predicates safe to evaluate against row-group statistics.

    Only a comparison of one bare column with a typed literal constant, and
    ``IS [NOT] NULL`` on a bare column, are pushed.  OR, NOT, CASE,
    arithmetic, column-to-column comparisons and anything that cannot be
    decided from min/max/null_count stay a row-level filter.
    """
    pushed: list[tuple] = []
    for conjunct in _and_conjuncts(where):
        tag = conjunct[0]
        if tag == "cmp":
            left, right = conjunct[2], conjunct[3]
            if left[0] == "column" and right[0] == "literal":
                pushed.append(conjunct)
            elif right[0] == "column" and left[0] == "literal":
                pushed.append(conjunct)
        elif tag == "isnull" and conjunct[1][0] == "column":
            pushed.append(conjunct)
    return pushed


def _select_row_groups(
    where: tuple | None, schema: Schema, groups: list[dict]
) -> tuple[list[int], list[tuple]]:
    """Return (kept group indices, pushable predicates).

    A group is excluded only when one pushed predicate is provably never TRUE
    given the group's min/max/null_count statistics.
    """
    predicates = _pushdown_predicates(where)
    kept: list[int] = []
    for index, group in enumerate(groups):
        stats = {entry["name"]: entry for entry in group["columns"]}
        if all(
            _predicate_possible(pred, stats, group["row_count"]) for pred in predicates
        ):
            kept.append(index)
    return kept, predicates


def _predicate_possible(node: tuple, stats: Mapping, group_rows: int) -> bool:
    """Whether ``node`` could evaluate to TRUE for any row in the group."""
    tag = node[0]
    if tag == "isnull":
        column = node[1]
        entry = stats[column[1]]
        if node[2]:  # IS NOT NULL
            return entry["null_count"] < group_rows
        return entry["null_count"] > 0
    # A pushed comparison always has exactly one bare column operand.
    op = node[1]
    left, right = node[2], node[3]
    if left[0] == "column":
        return _comparison_possible(op, stats[left[1]], right, True, group_rows)
    return _comparison_possible(op, stats[right[1]], left, False, group_rows)


def _comparison_possible(
    op: str, entry: Mapping, literal_node: tuple, column_left: bool, group_rows: int
) -> bool:
    """Stats test for ``column op literal`` (or the mirrored orientation)."""
    value = _eval(literal_node, ())
    if value is None:
        return False
    if entry["null_count"] == group_rows:
        # Every value is NULL: no comparison can be TRUE.
        return False
    minimum, maximum = entry["min"], entry["max"]
    if op in ("=", "!="):
        in_range = minimum <= value <= maximum
        if op == "=":
            return in_range
        # != is impossible only when every non-NULL value equals ``value``.
        return not (minimum == maximum == value)
    if not column_left:
        # Literal on the left mirrors the operator: lit <op> column.
        op = _mirror_op(op)
    if op == "<":
        return minimum < value
    if op == "<=":
        return minimum <= value
    if op == ">":
        return maximum > value
    return maximum >= value  # ">="


def _mirror_op(op: str) -> str:
    return {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "=", "!=": "!="}[op]


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
            out_columns.append(
                tuple(
                    _eval(item.expr, _row_values(source_columns, i))
                    for i in selected
                )
            )
    return out_columns


def _project_item_rows(items, source_columns, selected) -> list[tuple]:
    return [
        tuple(
            source_columns[item.col_index][i]
            if item.kind == "column"
            else _eval(item.expr, _row_values(source_columns, i))
            for item in items
        )
        for i in selected
    ]


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
    node: tuple,
    source_columns: tuple[tuple, ...],
    rows: tuple[int, ...],
    agg_values: tuple,
) -> bool | None:
    """Three-valued evaluation of a bound HAVING condition for one group."""
    tag = node[0]
    if tag == "literal":
        return node[1]
    if tag == "column":
        # Bound HAVING columns are always grouping columns; groups are never
        # empty (and the global aggregate case carries no column nodes).
        return source_columns[node[2]][rows[0]]
    if tag == "hagg":
        return agg_values[node[1]]
    if tag == "isnull":
        value = _eval_having(node[1], source_columns, rows, agg_values)
        result = value is None
        return (not result) if node[2] else result
    if tag == "not":
        value = _eval_having(node[1], source_columns, rows, agg_values)
        return None if value is None else (not value)
    if tag in ("and", "or"):
        # Short-circuit SQL semantics; UNKNOWN propagates only when needed.
        left = _eval_having(node[1], source_columns, rows, agg_values)
        right = _eval_having(node[2], source_columns, rows, agg_values)
        if tag == "and":
            if left is False or right is False:
                return False
            if left is None or right is None:
                return None
            return True
        if left is True or right is True:
            return True
        if left is None or right is None:
            return None
        return False
    if tag == "cmp":
        left = _eval_having(node[2], source_columns, rows, agg_values)
        right = _eval_having(node[3], source_columns, rows, agg_values)
        if left is None or right is None:
            return None
        op = node[1]
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
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover


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
    # here, so their errors surface even for rows LIMIT would cut.
    key_sources: list[tuple] = []
    for entry in order_by:
        if entry[0] == "col":
            key_sources.append((source_columns[entry[1]], entry[2], entry[3]))
        else:  # "expr"
            values = {
                i: _eval(entry[1], _row_values(source_columns, i)) for i in selected
            }
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
    pushdowns = {
        _SINGLE_TABLE_NAME: _pushdown_info(path, metadata, schema, bound)
    }
    return _build_explain(
        sources, schema, select, bound, steps=(), pushdowns=pushdowns
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
        pushdowns = {
            from_key: _pushdown_info(paths[from_key], from_metadata, from_schema, bound)
        }
        return _build_explain(
            sources, from_schema, rewritten, bound, steps=(), pushdowns=pushdowns
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
    # Join scans never push WHERE statistics: the predicate runs after the
    # whole chain, so each source selects all of its row groups.
    pushdowns = {}
    for key, metadata, _source_schema in source_entries:
        if metadata["format_version"] == FORMAT_VERSION_V2:
            total = len(inspect_row_groups(paths[key])["row_groups"])
            pushdowns[key] = (total, total, None)
    return _build_explain(
        tuple(source_entries),
        current_schema,
        rewritten,
        bound,
        steps=steps,
        strategy=strategy,
        pushdowns=pushdowns,
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


def _collect_expr_columns(node: tuple, indices: set) -> None:
    tag = node[0]
    if tag == "literal":
        return
    if tag == "column":
        indices.add(node[2])
        return
    if tag in ("not", "isnull", "unary"):
        _collect_expr_columns(node[1], indices)
        return
    if tag in ("and", "or"):
        _collect_expr_columns(node[1], indices)
        _collect_expr_columns(node[2], indices)
        return
    if tag == "arith":
        _collect_expr_columns(node[2], indices)
        _collect_expr_columns(node[3], indices)
        return
    if tag == "case":
        for condition, result in node[1]:
            _collect_expr_columns(condition, indices)
            _collect_expr_columns(result, indices)
        if node[2] is not None:
            _collect_expr_columns(node[2], indices)
        return
    if tag == "cmp":
        _collect_expr_columns(node[2], indices)
        _collect_expr_columns(node[3], indices)
        return
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover


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
        _collect_having_columns(bound["having"], indices)
    for entry in bound["order_by"] or ():
        if entry[0] == "col":
            indices.add(entry[1])
        elif entry[0] == "expr":
            _collect_expr_columns(entry[1], indices)
        # Aggregate ORDER BY entries index a selected output column, which
        # is already covered by the projection scan above.
    return indices


def _collect_having_columns(node: tuple, indices: set) -> None:
    """Column indices referenced by a bound HAVING tree (agg args excluded)."""
    tag = node[0]
    if tag in ("literal", "hagg"):
        return
    if tag == "column":
        indices.add(node[2])
        return
    if tag in ("not", "isnull"):
        _collect_having_columns(node[1], indices)
        return
    if tag in ("and", "or"):
        _collect_having_columns(node[1], indices)
        _collect_having_columns(node[2], indices)
        return
    if tag == "cmp":
        _collect_having_columns(node[2], indices)
        _collect_having_columns(node[3], indices)
        return
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover


def _expr_json(node: tuple) -> dict:
    """Render a bound expression as a recursive JSON-serialisable tree.

    Internal nodes carry ``kind`` / ``operator`` / ``operands``; leaves are
    typed literals or bound column names.
    """
    tag = node[0]
    if tag == "literal":
        return {"kind": "literal", "type": node[2], "value": node[1]}
    if tag == "column":
        return {"kind": "column", "name": node[1]}
    if tag == "unary":
        operator = "-" if node[2] else "+"
        return {
            "kind": "unary",
            "operator": operator,
            "operands": [_expr_json(node[1])],
        }
    if tag == "arith":
        return {
            "kind": "arithmetic",
            "operator": node[1],
            "operands": [_expr_json(node[2]), _expr_json(node[3])],
        }
    if tag == "case":
        # Written order is preserved; an omitted ELSE is rendered as null,
        # which also marks the implicit result nullable in the output schema.
        return {
            "kind": "case",
            "cases": [
                {"when": _expr_json(cond), "then": _expr_json(result)}
                for cond, result in node[1]
            ],
            "else": None if node[2] is None else _expr_json(node[2]),
        }
    if tag == "cmp":
        return {
            "kind": "comparison",
            "operator": node[1],
            "operands": [_expr_json(node[2]), _expr_json(node[3])],
        }
    if tag == "isnull":
        return {
            "kind": "is_null",
            "operator": "IS NOT NULL" if node[2] else "IS NULL",
            "operands": [_expr_json(node[1])],
        }
    if tag == "not":
        return {
            "kind": "not",
            "operator": "NOT",
            "operands": [_expr_json(node[1])],
        }
    if tag in ("and", "or"):
        return {
            "kind": "logic",
            "operator": tag.upper(),
            "operands": [_expr_json(node[1]), _expr_json(node[2])],
        }
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover


def _having_expr_json(node: tuple, schema: Schema, aggregates: tuple) -> dict:
    """Render a bound HAVING condition as a recursive JSON-serialisable tree.

    The boolean structure mirrors :func:`_expr_json`; value leaves are typed
    literals, bound grouping columns or aggregate calls (the aggregate leaf
    additionally carries its static ``type`` and ``nullable`` flag).
    """
    tag = node[0]
    if tag == "literal":
        return {"kind": "literal", "type": node[2], "value": node[1]}
    if tag == "column":
        return {"kind": "column", "name": node[1]}
    if tag == "hagg":
        agg = aggregates[node[1]]
        leaf = {
            "kind": "aggregate",
            "function": agg.func,
            "argument": None if agg.arg_index < 0 else schema.columns[agg.arg_index].name,
            "type": agg.out_type,
            "nullable": agg.nullable,
        }
        if agg.distinct:
            leaf["distinct"] = True
        return leaf
    if tag == "cmp":
        return {
            "kind": "comparison",
            "operator": node[1],
            "operands": [
                _having_expr_json(node[2], schema, aggregates),
                _having_expr_json(node[3], schema, aggregates),
            ],
        }
    if tag == "isnull":
        return {
            "kind": "is_null",
            "operator": "IS NOT NULL" if node[2] else "IS NULL",
            "operands": [_having_expr_json(node[1], schema, aggregates)],
        }
    if tag == "not":
        return {
            "kind": "not",
            "operator": "NOT",
            "operands": [_having_expr_json(node[1], schema, aggregates)],
        }
    if tag in ("and", "or"):
        return {
            "kind": "logic",
            "operator": tag.upper(),
            "operands": [
                _having_expr_json(node[1], schema, aggregates),
                _having_expr_json(node[2], schema, aggregates),
            ],
        }
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover


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


def _combine_conjuncts(conjuncts: Sequence[tuple]) -> tuple | None:
    """Join flattened AND conjuncts back into one bound predicate tree."""
    if not conjuncts:
        return None
    node = conjuncts[0]
    for following in conjuncts[1:]:
        node = ("and", node, following)
    return node


def _pushdown_info(
    path: Any, metadata: Mapping, schema: Schema, bound: Mapping
) -> tuple[int, int, dict | None] | None:
    """Scan-level row-group pushdown information for one v2 source.

    Returns ``(total groups, selected groups, pushed condition JSON)`` or
    ``None`` for version 1 sources (whose Scan plan is unchanged).
    """
    if metadata["format_version"] != FORMAT_VERSION_V2:
        return None
    groups = inspect_row_groups(path)["row_groups"]
    total = len(groups)
    kept, predicates = _select_row_groups(bound["where"], schema, groups)
    pushed_tree = _combine_conjuncts(predicates)
    condition = _expr_json(pushed_tree) if pushed_tree is not None else None
    return total, len(kept), condition


def _build_explain(
    sources: tuple,
    schema: Schema,
    select: _Select,
    bound: Mapping,
    steps: tuple[_JoinStep, ...] = (),
    strategy: str | None = None,
    pushdowns: Mapping | None = None,
) -> dict:
    pushdowns = pushdowns or {}
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
        scan = {"operator": "Scan", "source": key, "required_columns": required}
        info = pushdowns.get(key)
        if info is not None:
            total, selected, condition = info
            scan["row_groups_total"] = total
            scan["row_groups_selected"] = selected
            scan["pushed_condition"] = condition
        operators.append(scan)

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
            join_operator["strategy"] = _JOIN_STRATEGY_LABELS[strategy]
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
                    "condition": _having_expr_json(
                        bound["having"], schema, bound["aggregates"]
                    ),
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
