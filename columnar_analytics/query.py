"""Single-file SQL query layer.

Public API:

* :func:`query_file` -- run a ``SELECT ... FROM input [WHERE ...]`` query
  against one columnar file and return a :class:`~columnar_analytics.format.Table`
* :class:`QuerySyntaxError` -- lexical / grammatical errors
* :class:`QueryValidationError` -- unknown columns, wrong table name,
  type-incompatible comparisons, invalid aggregate use

The accepted grammar (keywords case-insensitive)::

    query       := SELECT select_item (',' select_item)*
                   FROM ident [WHERE expr] [GROUP BY ident (',' ident)*]
                   [ORDER BY order_item (',' order_item)*]
                   [LIMIT uint]
    select_item := '*' | ident | agg_name '(' ('*' | ident) ')'
                   | scalar_expr AS alias
    order_item  := (ident | agg_name '(' ('*' | ident) ')')
                   [ASC | DESC] [NULLS FIRST | NULLS LAST]
    scalar_expr := add_expr
    add_expr    := mul_expr (('+' | '-') mul_expr)*
    mul_expr    := unary_expr (('*' | '/') unary_expr)*
    unary_expr  := ('+' | '-') unary_expr | scalar_atom
    scalar_atom := '(' scalar_expr ')' | ident | numeric_literal
    expr       := or_expr
    or_expr     := and_expr (OR and_expr)*
    and_expr    := cmp_expr (AND cmp_expr)*
    cmp_expr    := not_factor (cmp_op not_factor)?
    not_factor  := NOT not_factor | postfix
    postfix     := scalar_pred (IS [NOT] NULL)?
    scalar_pred := add_expr            -- a '(' atom re-enters the boolean grammar
    literal     := TRUE | FALSE | [+-]? int64 | [+-]? finite float64
                   | single-quoted utf8

A SELECT scalar expression is numeric only (int64 / float64 columns and
numeric literals) and must give its result a unique name with ``AS alias``;
the alias follows the existing identifier rules and output keeps written
order, while bare columns and aggregate calls keep their existing labels.
Arithmetic precedence is parentheses, unary sign, ``*``/``/``, ``+``/``-``.
Two int64 operands combined with ``+``, ``-`` or ``*`` yield int64; a float64
operand or any ``/`` yields float64; a unary sign keeps its operand's type.
NULL on either side yields NULL; an output column is nullable exactly when
the expression references a nullable column (pure constants are non-null).
int64 overflow, division by zero and non-finite float64 results raise
:class:`QueryValidationError` at evaluation time.  WHERE comparisons accept
scalar expressions on either side with the existing type-compatibility and
three-valued-logic rules; a bare numeric expression is not a boolean
predicate.  Scalar expressions are not extended into aggregates: GROUP BY
and aggregate arguments stay plain column references or COUNT(*), and a
scalar SELECT expression in an aggregate/GROUP BY query is rejected.

The supported aggregates are ``COUNT(*)``, ``COUNT(ident)`` and
``SUM`` / ``AVG`` / ``MIN`` / ``MAX`` applied to one column.  Without
GROUP BY the projection may contain aggregate expressions only; with
GROUP BY it may additionally contain the grouped columns.  A bare ``*``
must not be mixed with aggregates or grouping.  Aggregates are not
allowed inside WHERE, may not be nested, and aliases are not supported.

Operator precedence (highest first) is NOT, comparison, AND, OR; IS [NOT]
NULL is a postfix of its value.  Comparison sides are value expressions --
column references, literals or parenthesised numeric scalar expressions.
WHERE follows SQL three-valued logic: a normal comparison against NULL
yields UNKNOWN, UNKNOWN propagates through the logical operators, and only
TRUE rows are returned.

For a non-aggregate query execution proceeds WHERE, sort-key computation,
stable sort, LIMIT, then the SELECT result expressions, so rows filtered or
capped away never trigger a SELECT-side division by zero or overflow.
ORDER BY may name an input column or a SELECT expression alias (an alias
sharing an input column's name wins); ASC is the default, NULLs sort last
regardless of direction unless NULLS FIRST / NULLS LAST is given, and rows
equal on every sort key keep their original relative order.  After WHERE,
rows are grouped in GROUP BY column order (a NULL key forms its own group);
without an explicit ORDER BY groups come out in the order of their first
selected row.  In an aggregate query ORDER BY may only name selected
grouped columns or selected aggregate expressions, and LIMIT keeps the
first N groups.  Without GROUP BY and aggregates an aggregate query over
zero selected rows still yields one output row (COUNT 0, the other
aggregates NULL); with GROUP BY it yields zero rows.

Two-file queries (:func:`query_files`) add one equi-join to the grammar::

    query := SELECT ... FROM left_table [join_kind JOIN right_table
             ON left_table.column = right_table.column] ...
    join_kind := INNER | LEFT

``sources`` maps table names to file paths; only the tables referenced by
the statement are read.  Aliases, compound ON conditions, other join types
and a second join are rejected as syntax errors before any file is opened.
In a join query every column reference outside ``COUNT(*)`` must be
qualified as ``table.column`` (either part may be a double-quoted
identifier).  ``SELECT *`` emits the left schema followed by the right
schema with ``table.column`` names; explicit projections keep their
qualified names and aggregates keep the uppercase ``FUNC(table.column)``
labels.  Join keys may share a type or mix int64 with float64; NULL keys
never match.  INNER JOIN emits every matching combination, LEFT JOIN also
emits unmatched left rows with the right-side values set to NULL (the
right result columns are nullable).  Rows expand in left-file order, and
within one left row in right-file order, before WHERE / GROUP BY /
ORDER BY / LIMIT apply with their usual semantics.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any

from .format import ColumnSchema, Schema, Table, read_file

__all__ = [
    "QuerySyntaxError",
    "QueryValidationError",
    "query_file",
    "query_files",
]

_KEYWORDS = frozenset(
    (
        "select",
        "from",
        "where",
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
        "inner",
        "left",
        "join",
        "on",
    )
)

# Aggregate function names are ordinary (case-insensitive) identifiers that
# gain call syntax only in the projection and ORDER BY.
_AGG_NAMES = frozenset(("count", "sum", "avg", "min", "max"))
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


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
            # used by two-table queries ("table.column").
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
#   ("unary", sign, operand)                          -- as parsed
#   ("unary", sign, operand, type_name, nullable)     -- after binding
#   ("bin", op, left, right)                          -- as parsed
#   ("bin", op, left, right, type_name, nullable)     -- after binding
#   ("cmp", op, left_node, right_node)
#   ("isnull", operand, negate)
#   ("not", operand)
#   ("and"|"or", left, right)
# Arithmetic nodes are numeric (int64/float64); predicate nodes are
# boolean-typed with three-valued evaluation.
#
# Projection / ORDER BY reference items:
#   ("column_ref", name)                 -- a plain column name
#   ("agg", func_upper, arg_name|None)   -- an aggregate call; arg None = '*'
# Projection may additionally carry an "expr" _RefItem (a bound scalar tree
# with its required AS alias).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RefItem:
    """One projection element or ORDER BY key as parsed."""

    kind: str  # "star" | "column" | "agg" | "expr"
    name: str | None = None  # column name for kind == "column"
    func: str | None = None  # uppercase function name for kind == "agg"
    arg: str | None = None  # aggregate column argument; "" stands for '*'
    descending: bool = False
    nulls_first: bool | None = None  # None -> default (NULLs last)
    # Table qualifiers (two-table queries only; None when unqualified):
    table: str | None = None
    table_quoted: bool = False
    arg_table: str | None = None
    arg_table_quoted: bool = False
    # kind == "expr": the parsed scalar expression tree and its AS alias.
    expr: tuple | None = None
    alias: str | None = None


@dataclass(frozen=True)
class _Join:
    """The single optional equi-join of a two-table query."""

    kind: str  # "inner" | "left"
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
    order_by: tuple[_RefItem, ...] | None
    limit: int | None
    join: _Join | None = None

    @property
    def star(self) -> bool:
        return len(self.items) == 1 and self.items[0].kind == "star"

    @property
    def has_aggregate(self) -> bool:
        return any(item.kind == "agg" for item in self.items)

    @property
    def is_aggregate_query(self) -> bool:
        return self.has_aggregate or self.group_by is not None


class _Parser:
    _CMP_OPS = frozenset(("=", "!=", "<", "<=", ">", ">="))

    def __init__(self, tokens: list[_Token], allow_join: bool = False):
        self.tokens = tokens
        self.pos = 0
        # Two-table statements (qualified names + one JOIN clause) are only
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
        items = tuple(self._parse_projection())
        self._expect_keyword("from")
        table_tok = self._expect_table_name()
        join = None
        if self._allow_join:
            if self._accept_keyword("inner"):
                self._expect_keyword("join")
                join = self._parse_join("inner")
            elif self._accept_keyword("left"):
                self._expect_keyword("join")
                join = self._parse_join("left")
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
            order_by=order_by,
            limit=limit,
            join=join,
        )

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
        # Aliases are not supported on bare columns or aggregates; a trailing
        # identifier (including the spelling "as") where FROM is expected names
        # one.  Plain queries keep their historical QuerySyntaxError
        # classification; aggregate queries (or statements with GROUP BY) treat
        # the unsupported alias as a validation error.  Computed scalar
        # expressions consume their own required AS alias inside
        # _parse_select_item and therefore never reach this check.
        if all(item.kind != "expr" for item in items):
            tok = self._peek()
            if tok.kind in ("ident", "qident"):
                grouped_ahead = any(
                    t.kind == "keyword" and t.value == "group"
                    for t in self.tokens[self.pos :]
                )
                if any(item.kind == "agg" for item in items) or grouped_ahead:
                    raise QueryValidationError(
                        f"aliases are not supported; unexpected {tok.text!r} after projection item"
                    )
                raise QuerySyntaxError(f"expected FROM, got {tok.text!r}")
        else:
            tok = self._peek()
            if tok.kind in ("ident", "qident"):
                raise QuerySyntaxError(f"expected FROM, got {tok.text!r}")
        return items

    def _parse_select_item(self) -> _RefItem:
        tok = self._peek()
        if tok.kind == "star":
            self._next()
            return _RefItem("star")
        # A leading '(', sign or numeric literal starts a computed scalar
        # expression; such items must name their result with AS.
        if (
            tok.kind == "number"
            or (tok.kind == "op" and tok.value in ("(", "+", "-"))
        ):
            node = self._parse_scalar()
            return _RefItem("expr", expr=node, alias=self._parse_alias())
        item = self._parse_column_or_agg()
        nxt = self._peek()
        is_arith = nxt.kind == "star" or (
            nxt.kind == "op" and nxt.value in ("+", "-", "*", "/")
        )
        if not is_arith:
            return item
        if item.kind == "agg":
            raise QueryValidationError(
                f"aggregate {item.func}(...) cannot be combined with a scalar expression"
            )
        node = self._parse_scalar_tail(
            ("column", item.name, item.table, item.table_quoted)
        )
        return _RefItem("expr", expr=node, alias=self._parse_alias())

    def _parse_alias(self) -> str:
        # AS is an ordinary identifier token (it is not a reserved keyword);
        # the spelling is accepted case-insensitively.
        tok = self._peek()
        if tok.kind == "ident" and tok.value.lower() == "as":
            self._next()
        name_tok = self._peek()
        if name_tok.kind not in ("ident", "qident"):
            raise QuerySyntaxError(
                f"computed expression requires AS followed by an alias, got {name_tok.text!r}"
            )
        self._next()
        return name_tok.value

    # Scalar numeric expressions ---------------------------------------------
    # Precedence (lowest first): additive, multiplicative, unary, atom.
    # '_parse_scalar' parses a whole expression; '_parse_scalar_tail' seeds it
    # with an already-consumed leading column atom.  ``paren`` (WHERE only)
    # re-enters the boolean grammar for a parenthesised group, so predicates
    # and arithmetic can nest freely inside WHERE.

    def _parse_scalar(self, paren=None) -> tuple:
        return self._parse_scalar_add_from(
            self._parse_scalar_mult_from(self._parse_scalar_unary(paren), paren),
            paren,
        )

    def _parse_scalar_tail(self, first_atom: tuple) -> tuple:
        return self._parse_scalar_add_from(
            self._parse_scalar_mult_from(first_atom, None), None
        )

    def _parse_scalar_add_from(self, left: tuple, paren) -> tuple:
        while True:
            tok = self._peek()
            if tok.kind == "op" and tok.value in ("+", "-"):
                self._next()
                self._expect_operand(tok.value)
                right = self._parse_scalar_mult_from(self._parse_scalar_unary(paren), paren)
                left = ("bin", tok.value, left, right)
            else:
                return left

    def _parse_scalar_mult_from(self, left: tuple, paren) -> tuple:
        while True:
            tok = self._peek()
            if tok.kind == "star" or (
                tok.kind == "op" and tok.value in ("*", "/")
            ):
                op = "*" if tok.kind == "star" else tok.value
                self._next()
                self._expect_operand(op)
                right = self._parse_scalar_unary(paren)
                left = ("bin", op, left, right)
            else:
                return left

    def _expect_operand(self, op: str) -> None:
        # The token following a binary operator or unary sign must start an
        # operand.  A bare "as" here is the alias marker with the operand
        # missing, which is a grammatical error recognised before file access.
        tok = self._peek()
        if tok.kind == "ident" and tok.value.lower() == "as":
            raise QuerySyntaxError(f"missing operand after {op!r}")
        if tok.kind in ("op", "keyword", "star", "eof") and not (
            tok.kind == "op" and tok.value in ("(", "+", "-")
        ):
            raise QuerySyntaxError(f"missing operand after {op!r}")

    def _parse_scalar_unary(self, paren) -> tuple:
        tok = self._peek()
        if tok.kind == "op" and tok.value in ("+", "-"):
            sign = tok.value
            self._next()
            nxt = self._peek()
            if nxt.kind == "number" and not isinstance(nxt.value, bool):
                # Fold the sign into a literal so literal range errors stay
                # grammatical (and surface before any file is read).
                return self._parse_signed_literal(nxt, sign)
            self._expect_operand(sign)
            operand = self._parse_scalar_unary(paren)
            return ("unary", sign, operand)
        return self._parse_scalar_atom(paren)

    def _parse_signed_literal(self, tok: _Token, sign: str) -> tuple:
        self._next()
        value = tok.value
        if isinstance(value, int):
            value = value if sign == "+" else -value
            if not (_INT64_MIN <= value <= _INT64_MAX):
                raise QuerySyntaxError("integer literal is outside the int64 range")
            return ("literal", value, "int64")
        value = math.copysign(value, -1.0 if sign == "-" else 1.0)
        if not math.isfinite(value):
            raise QuerySyntaxError("float literal must be finite")
        return ("literal", value, "float64")

    def _parse_scalar_atom(self, paren) -> tuple:
        if self._accept_op("("):
            if paren is not None:
                node = paren()
            else:
                node = self._parse_scalar()
            self._expect_op(")")
            return node
        return self._parse_scalar_value()

    def _parse_scalar_value(self) -> tuple:
        tok = self._peek()
        if tok.kind == "number":
            self._next()
            if isinstance(tok.value, bool):
                return ("literal", tok.value, "bool")
            if isinstance(tok.value, int):
                if tok.value > _INT64_MAX:
                    raise QuerySyntaxError("integer literal is outside the int64 range")
                return ("literal", tok.value, "int64")
            return ("literal", tok.value, "float64")
        if tok.kind == "string":
            self._next()
            return ("literal", tok.value, "utf8")
        if tok.kind in ("ident", "qident"):
            return self._parse_scalar_column()
        if tok.kind == "keyword":
            raise QuerySyntaxError(
                f"unexpected keyword {tok.text.upper()!r} in expression"
            )
        raise QuerySyntaxError(f"unexpected token {tok.text!r} in expression")

    def _parse_scalar_column(self) -> tuple:
        tok = self._peek()
        if self._allow_join and self._qualifier_ahead():
            item = self._parse_qualified_column()
            nxt = self._peek()
            if nxt.kind == "op" and nxt.value == "(":
                raise QuerySyntaxError(
                    "function calls are not allowed in scalar expressions"
                )
            return ("column", item.name, item.table, item.table_quoted)
        self._next()
        nxt = self._peek()
        if nxt.kind == "op" and nxt.value == "(":
            if tok.kind == "ident" and tok.value.lower() in _AGG_NAMES:
                raise QueryValidationError(
                    f"aggregate {tok.value.upper()}(...) is not allowed in scalar expressions"
                )
            raise QuerySyntaxError(
                f"unsupported function {tok.text!r} in scalar expression"
            )
        return ("column", tok.value, None, False)

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
        arg, arg_table, arg_table_quoted = self._parse_aggregate_args(func)
        self._expect_op(")")
        return _RefItem(
            "agg",
            func=func.upper(),
            arg=arg,
            arg_table=arg_table,
            arg_table_quoted=arg_table_quoted,
        )

    def _parse_aggregate_args(self, func: str) -> tuple:
        # Exactly one argument: '*' for COUNT, otherwise one (possibly
        # table-qualified) identifier.  Returns (name, table, table_quoted);
        # name "" stands for '*'.
        tok = self._peek()
        if tok.kind == "star":
            if func != "count":
                raise QuerySyntaxError(f"{func.upper()} does not accept '*'")
            self._next()
            return ("", None, False)
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
            return (item.name, item.table, item.table_quoted)
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
        return (tok.value, None, False)

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
        if tok.kind not in ("ident", "qident"):
            raise QuerySyntaxError(f"expected identifier, got {tok.text!r}")
        if self._allow_join and self._qualifier_ahead():
            item = self._parse_qualified_column()
            return (item.table, item.table_quoted, item.name)
        self._next()
        return (None, False, tok.value)

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
        # Parentheses re-enter the boolean grammar; arithmetic is the level
        # immediately below the comparison / logical operators.
        node = self._parse_scalar(self._parse_or)
        if self._accept_keyword("is"):
            negate = self._accept_keyword("not")
            self._expect_keyword("null")
            node = ("isnull", node, negate)
        return node

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
    out_type: str = ""
    nullable: bool = True
    # kind == "expr": the bound numeric scalar expression.
    expr: tuple | None = None


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
    if where is not None and not _is_boolean_node(where) and _scalar_type(where) != "bool":
        type_name = _scalar_type(where)
        raise QueryValidationError(
            f"WHERE clause must be boolean, got {type_name}"
        )

    if not select.is_aggregate_query:
        return _bind_plain(select, schema, where)
    return _bind_aggregate(select, schema, where)


def _is_boolean_node(node: tuple) -> bool:
    return node[0] in ("cmp", "isnull", "not", "and", "or")


def _bind_plain(select: _Select, schema: Schema, where) -> dict:
    # Non-aggregate path: '*' or a comma-separated list of plain columns and
    # computed numeric expressions (the latter always carry an AS alias).
    if select.star:
        items = tuple(
            _BoundItem(
                "column",
                output_name=col.name,
                col_index=i,
                out_type=col.type,
                nullable=col.nullable,
            )
            for i, col in enumerate(schema.columns)
        )
    else:
        bound_items: list[_BoundItem] = []
        output_seen: set[str] = set()
        for item in select.items:
            if item.kind == "star":
                # A star mixed into a projection remains grammatical.
                raise QuerySyntaxError("'*' cannot be mixed with other projection items")
            if item.kind == "expr":
                node, out_type, nullable = _bind_scalar(item.expr, schema)
                if out_type not in ("int64", "float64"):
                    raise QueryValidationError(
                        f"computed SELECT expressions must be int64 or float64, got {out_type}"
                    )
                label = item.alias
                if label in output_seen:
                    raise QueryValidationError(f"duplicate result column: {label!r}")
                output_seen.add(label)
                bound_items.append(
                    _BoundItem(
                        "expr",
                        output_name=label,
                        expr=node,
                        out_type=out_type,
                        nullable=nullable,
                    )
                )
                continue
            name = item.name
            if name in output_seen:
                raise QueryValidationError(f"duplicate column in projection: {name!r}")
            output_seen.add(name)
            try:
                col_index = schema.index(name)
            except KeyError:
                raise QueryValidationError(f"unknown column: {name!r}") from None
            col = schema.columns[col_index]
            bound_items.append(
                _BoundItem(
                    "column",
                    output_name=name,
                    col_index=col_index,
                    out_type=col.type,
                    nullable=col.nullable,
                )
            )
        items = tuple(bound_items)

    aliases = {
        item.output_name: (item.expr, item.out_type, item.nullable)
        for item in items
        if item.kind == "expr"
    }
    order_by = _bind_plain_order_by(select.order_by, schema, aliases)
    return {
        "mode": "plain",
        "items": items,
        "where": where,
        "order_by": order_by,
    }


def _bind_plain_order_by(select_order_by, schema: Schema, aliases: dict):
    if select_order_by is None:
        return None
    # Each bound key is ("c", col_index, descending, nulls_first) for a plain
    # input column or ("e", bound_expr, descending, nulls_first) for a SELECT
    # alias.  A name spelling an alias resolves to that alias even when an
    # input column shares the name.
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
        if name in aliases:
            bound.append(("e", aliases[name][0], item.descending, nulls_first))
            continue
        try:
            col_index = schema.index(name)
        except KeyError:
            raise QueryValidationError(
                f"unknown ORDER BY column: {name!r}"
            ) from None
        bound.append(("c", col_index, item.descending, nulls_first))
    return tuple(bound)


def _bind_aggregate(select: _Select, schema: Schema, where) -> dict:
    if select.star:
        raise QueryValidationError("'*' cannot be combined with aggregates or GROUP BY")
    if any(item.kind == "expr" for item in select.items):
        raise QueryValidationError(
            "scalar expressions are not allowed alongside aggregates or GROUP BY"
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
            bound_items.append(_bind_agg_item(item, schema, output_seen))

    if select.group_by is None and has_plain:
        # No grouping: every projected column must be an aggregate.
        raise QueryValidationError(
            "without GROUP BY, the projection may contain aggregates only"
        )

    order_by = _bind_aggregate_order_by(
        select.order_by, bound_items, schema
    )
    return {
        "mode": "aggregate",
        "group_indices": tuple(group_indices),
        "items": tuple(bound_items),
        "where": where,
        "order_by": order_by,
    }


def _bind_agg_item(item: _RefItem, schema: Schema, output_seen: set[str]) -> _BoundItem:
    func = item.func
    if item.arg == "":
        if func != "COUNT":
            raise QuerySyntaxError(f"{func} does not accept '*'")
        label = "COUNT(*)"
        arg_index = -1
        arg_col = None
    else:
        arg_name = item.arg
        try:
            arg_index = schema.index(arg_name)
        except KeyError:
            raise QueryValidationError(f"unknown column: {arg_name!r}") from None
        arg_col = schema.columns[arg_index]
        # The result label uses the column name as spelled in the schema.
        label = f"{func}({arg_col.name})"
    if label in output_seen:
        raise QueryValidationError(f"duplicate result column: {label!r}")
    output_seen.add(label)

    if func == "COUNT":
        out_type, nullable = "int64", False
    elif func == "SUM":
        if arg_col.type not in ("int64", "float64"):
            raise QueryValidationError(
                f"SUM requires an int64 or float64 argument, got {arg_col.type}"
            )
        out_type, nullable = arg_col.type, True
    elif func == "AVG":
        if arg_col.type not in ("int64", "float64"):
            raise QueryValidationError(
                f"AVG requires an int64 or float64 argument, got {arg_col.type}"
            )
        out_type, nullable = "float64", True
    else:  # MIN / MAX accept all four existing types.
        out_type, nullable = arg_col.type, True
    return _BoundItem(
        "agg",
        output_name=label,
        func=func,
        arg_index=arg_index,
        arg_type=arg_col.type if arg_col is not None else "",
        out_type=out_type,
        nullable=nullable,
    )


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
            selected[("agg", f"{bound.func}|{arg_key}")] = i

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
                label = f"{item.func}({real_name})"
            key = ("agg", f"{item.func}|{arg_key}")
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
    if tag in ("literal", "column", "bin", "unary"):
        return _bind_scalar(node, schema)[0]
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
        if operand[0] not in ("literal", "column"):
            raise QuerySyntaxError(
                "IS NULL operand must be a column reference or a literal"
            )
        return ("isnull", operand, node[2])
    if tag == "cmp":
        op = node[1]
        if node[2][0] not in ("literal", "column", "bin", "unary") or node[3][0] not in (
            "literal",
            "column",
            "bin",
            "unary",
        ):
            raise QuerySyntaxError("comparison operands must be value expressions")
        left, left_t, _ = _bind_scalar(node[2], schema)
        right, right_t, _ = _bind_scalar(node[3], schema)
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


def _bind_scalar(node: tuple, schema: Schema) -> tuple:
    """Bind a numeric scalar tree.

    Returns ``(bound_node, type_name, nullable)``.  Arithmetic operators only
    accept int64 / float64; a lone bool or utf8 leaf is allowed so that plain
    comparison operands flow through this binder unchanged.
    """
    tag = node[0]
    if tag in ("cmp", "isnull", "not", "and", "or"):
        # A parenthesised boolean predicate used in a numeric position is a
        # value of type bool; _require_numeric rejects it with a type error.
        bound = _bind_expr(node, schema)
        return bound, "bool", False
    if tag == "literal":
        return node, node[2], False
    if tag == "column":
        name = node[1]
        try:
            index = schema.index(name)
        except KeyError:
            raise QueryValidationError(f"unknown column: {name!r}") from None
        col = schema.columns[index]
        return ("column", name, index, col.type, col.nullable), col.type, col.nullable
    if tag == "unary":
        operand, op_type, nullable = _bind_scalar(node[2], schema)
        _require_numeric(op_type, "a unary sign")
        return ("unary", node[1], operand, op_type, nullable), op_type, nullable
    if tag == "bin":
        op = node[1]
        left, left_t, left_null = _bind_scalar(node[2], schema)
        right, right_t, right_null = _bind_scalar(node[3], schema)
        _require_numeric(left_t, op)
        _require_numeric(right_t, op)
        if op == "/" or "float64" in (left_t, right_t):
            out_type = "float64"
        else:
            out_type = "int64"
        nullable = left_null or right_null
        return ("bin", op, left, right, out_type, nullable), out_type, nullable
    raise QuerySyntaxError(f"unsupported value expression: {tag}")  # pragma: no cover


def _require_numeric(type_name: str, context: str) -> None:
    if type_name not in ("int64", "float64"):
        raise QueryValidationError(
            f"{context} requires int64 or float64 operands, got {type_name}"
        )


def _scalar_type(node: tuple) -> str:
    tag = node[0]
    if tag == "literal":
        return node[2]
    if tag == "column":
        return node[3]
    if tag in ("bin", "unary"):
        return node[-2]
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover


def _require_boolean(node: tuple, context: str) -> None:
    if node[0] in ("cmp", "isnull", "not", "and", "or"):
        return
    type_name = _scalar_type(node)
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
        value = _eval(node[2], row)
        if value is None:
            return None
        if node[3] == "int64":
            result = value if node[1] == "+" else -value
            if not (_INT64_MIN <= result <= _INT64_MAX):
                raise QueryValidationError("integer arithmetic overflowed the int64 range")
            return result
        result = value if node[1] == "+" else -value
        if not math.isfinite(result):
            raise QueryValidationError("arithmetic produced a non-finite float64 value")
        return float(result)
    if tag == "bin":
        left = _eval(node[2], row)
        if left is None:
            # NULL propagates without evaluating the other side, so rows that
            # will be dropped never trigger its division / overflow.
            return None
        right = _eval(node[3], row)
        if right is None:
            return None
        op = node[1]
        out_type = node[4]
        if op == "/":
            if right == 0:
                raise QueryValidationError("division by zero")
            result = left / right
        elif out_type == "int64":
            if op == "+":
                result = left + right
            elif op == "-":
                result = left - right
            else:
                result = left * right
            if not (_INT64_MIN <= result <= _INT64_MAX):
                raise QueryValidationError("integer arithmetic overflowed the int64 range")
            return result
        else:
            if op == "+":
                result = left + right
            elif op == "-":
                result = left - right
            else:
                result = left * right
        if not math.isfinite(result):
            raise QueryValidationError("arithmetic produced a non-finite float64 value")
        return float(result)
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


# ---------------------------------------------------------------------------
# Aggregate computation
# ---------------------------------------------------------------------------


def _aggregate_value(func: str, arg_index: int, arg_type: str, rows, source_columns):
    if func == "COUNT":
        if arg_index == -1:
            return len(rows)
        col = source_columns[arg_index]
        return sum(1 for i in rows if col[i] is not None)

    col = source_columns[arg_index]
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


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def query_file(path: Any, sql: str) -> Table:
    """Run ``sql`` against the single columnar file ``path``.

    Returns a :class:`~columnar_analytics.format.Table` with columns in
    projection order.  Rows are filtered by WHERE, then grouped (for
    aggregate queries) or sorted by ORDER BY, then capped by LIMIT, then
    projected; without ORDER BY the file's original row order (or the
    first-selected-row group order) is kept.  The statement is parsed
    before the file is touched, so purely grammatical errors surface as
    :class:`QuerySyntaxError` regardless of whether ``path`` exists.
    Unknown columns, duplicate result columns, the wrong table name,
    type-incompatible predicates, ungrouped columns, illegal aggregate
    arguments or int64 SUM overflow raise :class:`QueryValidationError`;
    malformed files raise :class:`~columnar_analytics.format.ColumnarFormatError`;
    other I/O failures propagate as :class:`OSError`.
    """
    tokens = _tokenize(sql)
    select = _Parser(tokens).parse()
    table = read_file(path)
    return _run_query(table, select)


def query_table(table: Table, sql: str) -> Table:
    """Apply ``sql`` (``FROM input``) to an in-memory table."""
    tokens = _tokenize(sql)
    select = _Parser(tokens).parse()
    return _run_query(table, select)


def query_files(sources: Any, sql: str) -> Table:
    """Run ``sql`` against the tables named by ``sources``.

    ``sources`` maps table names to columnar file paths; only the tables
    referenced by the statement are read.  The statement may join two of
    them once (``INNER JOIN`` / ``LEFT JOIN ... ON t1.col = t2.col``); see
    the module docstring for the exact grammar and semantics.

    A non-mapping or empty ``sources``, non-string keys or non-path values
    raise :class:`ValueError` before any file is touched.  Lexical and
    grammatical errors (including join-keyword, ON and multiple-join
    problems) raise :class:`QuerySyntaxError` before any file is read.
    Unknown tables or columns, unqualified column references in a join
    query, duplicate tables, misattributed or type-incompatible join keys
    and duplicate result columns raise :class:`QueryValidationError`;
    malformed files raise :class:`~columnar_analytics.format.ColumnarFormatError`;
    other I/O failures propagate as :class:`OSError`.
    """
    paths = _validate_sources(sources)
    tokens = _tokenize(sql)
    select = _Parser(tokens, allow_join=True).parse()
    left_key = _resolve_table_ref(select.table, select.table_quoted, tuple(paths))
    if select.join is None:
        rewritten = _rewrite_select(select, _single_table_resolver(left_key))
        table = read_file(paths[left_key])
        return _run_query(table, rewritten, expected_table=None)

    join = select.join
    right_key = _resolve_table_ref(join.table, join.table_quoted, tuple(paths))
    if right_key == left_key:
        raise QueryValidationError(f"duplicate table {right_key!r} in join")
    on_left = _resolve_table_ref(join.left_key[0], join.left_key[1], (left_key, right_key))
    on_right = _resolve_table_ref(join.right_key[0], join.right_key[1], (left_key, right_key))
    if on_left != left_key or on_right != right_key:
        raise QueryValidationError(
            "ON keys must reference the left and right tables respectively"
        )
    # Qualifier checks (unqualified / unknown-table column references) do
    # not need the files and run before any read.
    rewritten = _rewrite_select(select, _join_resolver(left_key, right_key))
    left_table = read_file(paths[left_key])
    right_table = read_file(paths[right_key])
    combined = _build_joined_table(left_key, left_table, right_key, right_table, join)
    return _run_query(combined, rewritten, expected_table=None)


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


def _join_resolver(left_key: str, right_key: str):
    def resolve(table, table_quoted, column):
        if table is None:
            raise QueryValidationError(
                f"column reference {column!r} must be qualified with a table name"
            )
        key = _resolve_table_ref(table, table_quoted, (left_key, right_key))
        return f"{key}.{column}"

    return resolve


def _rewrite_select(select: _Select, resolve) -> _Select:
    """Resolve every column reference to its canonical output name."""
    items = tuple(_rewrite_ref_item(item, resolve) for item in select.items)
    # Computed SELECT expressions publish unqualified aliases; ORDER BY may
    # reference those names directly, so they must bypass table qualification.
    aliases = {item.alias for item in select.items if item.kind == "expr"}
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
    return _Select(
        items=items,
        table=select.table,
        table_quoted=select.table_quoted,
        where=where,
        group_by=group_by,
        order_by=order_by,
        limit=select.limit,
    )


def _rewrite_ref_item(item: _RefItem, resolve) -> _RefItem:
    if item.kind == "star":
        return item
    if item.kind == "expr":
        return _RefItem(
            "expr",
            expr=_rewrite_expr(item.expr, resolve),
            alias=item.alias,
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
        descending=item.descending,
        nulls_first=item.nulls_first,
    )


def _rewrite_expr(node: tuple, resolve) -> tuple:
    tag = node[0]
    if tag == "literal":
        return node
    if tag == "column":
        return ("column", resolve(node[2], node[3], node[1]), None, False)
    if tag == "not":
        return ("not", _rewrite_expr(node[1], resolve))
    if tag in ("and", "or"):
        return (tag, _rewrite_expr(node[1], resolve), _rewrite_expr(node[2], resolve))
    if tag == "isnull":
        return ("isnull", _rewrite_expr(node[1], resolve), node[2])
    if tag == "unary":
        return ("unary", node[1], _rewrite_expr(node[2], resolve))
    if tag == "bin":
        return (
            "bin",
            node[1],
            _rewrite_expr(node[2], resolve),
            _rewrite_expr(node[3], resolve),
        )
    if tag == "cmp":
        return ("cmp", node[1], _rewrite_expr(node[2], resolve), _rewrite_expr(node[3], resolve))
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


def _build_joined_table(
    left_key: str, left: Table, right_key: str, right: Table, join: _Join
) -> Table:
    left_col = join.left_key[2]
    right_col = join.right_key[2]
    try:
        left_idx = left.schema.index(left_col)
    except KeyError:
        raise QueryValidationError(
            f"unknown column: {f'{left_key}.{left_col}'!r}"
        ) from None
    try:
        right_idx = right.schema.index(right_col)
    except KeyError:
        raise QueryValidationError(
            f"unknown column: {f'{right_key}.{right_col}'!r}"
        ) from None
    left_type = left.schema.columns[left_idx].type
    right_type = right.schema.columns[right_idx].type
    if left_type != right_type and not {left_type, right_type} <= {"int64", "float64"}:
        raise QueryValidationError(
            f"join key types are incompatible: {left_type} and {right_type}"
        )

    # The combined schema is left columns then right columns, named
    # "table.column"; a LEFT JOIN makes every right-side column nullable.
    combined_columns = [
        ColumnSchema(f"{left_key}.{col.name}", col.type, col.nullable)
        for col in left.schema.columns
    ] + [
        ColumnSchema(
            f"{right_key}.{col.name}",
            col.type,
            True if join.kind == "left" else col.nullable,
        )
        for col in right.schema.columns
    ]
    names = [col.name for col in combined_columns]
    if len(set(names)) != len(names):
        raise QueryValidationError("joined tables produce duplicate column names")
    schema = Schema(combined_columns)
    columns = _join_columns(left, right, left_idx, right_idx, join.kind)
    return Table._from_storage(schema, columns)


def _join_columns(
    left: Table, right: Table, left_idx: int, right_idx: int, kind: str
) -> list:
    """Nested-loop equi-join preserving left-then-right file row order."""
    left_cols = left._columns
    right_cols = right._columns
    left_width = len(left_cols)
    right_width = len(right_cols)
    # NULL keys never match, so they stay out of the right-side index.
    index: dict[Any, list[int]] = {}
    for j, key in enumerate(right_cols[right_idx]):
        if key is not None:
            index.setdefault(key, []).append(j)
    out = [[] for _ in range(left_width + right_width)]
    for i in range(left.row_count):
        key = left_cols[left_idx][i]
        matches = index.get(key) if key is not None else None
        if matches:
            for j in matches:
                for c in range(left_width):
                    out[c].append(left_cols[c][i])
                for c in range(right_width):
                    out[left_width + c].append(right_cols[c][j])
        elif kind == "left":
            for c in range(left_width):
                out[c].append(left_cols[c][i])
            for c in range(right_width):
                out[left_width + c].append(None)
    return out


def _run_query(table: Table, select: _Select, expected_table="input") -> Table:
    bound = _bind_select(select, table.schema, expected_table)
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


def _run_plain(table: Table, select: _Select, bound, selected: list[int]) -> Table:
    items = bound["items"]
    order_by = bound["order_by"]
    source_columns = table._columns

    def row_values(i: int) -> tuple:
        return tuple(col[i] for col in source_columns)

    # Execution order for a non-aggregate query: WHERE (already applied by the
    # caller), compute sort keys, stable sort, LIMIT, then the result
    # expressions -- so filtered or capped rows never trigger a SELECT-side
    # division by zero or overflow.
    if order_by is not None:
        key_cache: dict[int, tuple] = {}

        def sort_key(i: int) -> tuple:
            cached = key_cache.get(i)
            if cached is None:
                row = row_values(i)
                cached = tuple(
                    source_columns[spec[1]][i]
                    if spec[0] == "c"
                    else _eval(spec[1], row)
                    for spec in order_by
                )
                key_cache[i] = cached
            return cached

        comparator = _make_plain_comparator(sort_key, order_by)
        selected = sorted(selected, key=cmp_to_key(comparator))

    if select.limit is not None:
        selected = selected[: select.limit]

    out_columns = []
    for item in items:
        if item.kind == "column":
            out_columns.append(
                tuple(source_columns[item.col_index][i] for i in selected)
            )
        else:
            out_columns.append(tuple(_eval(item.expr, row_values(i)) for i in selected))
    out_schema = Schema(
        [
            ColumnSchema(item.output_name, item.out_type, nullable=item.nullable)
            for item in items
        ]
    )
    return Table._from_storage(out_schema, out_columns)


def _run_aggregate(table: Table, select: _Select, bound, selected: list[int]) -> Table:
    group_indices = bound["group_indices"]
    bound_items = bound["items"]
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

    # Materialise one output tuple per group, following the SELECT order.
    materialised: list[tuple] = []
    for rows in groups:
        values = []
        for item in bound_items:
            if item.kind == "column":
                values.append(source_columns[item.col_index][rows[0]])
            else:
                values.append(
                    _aggregate_value(
                        item.func, item.arg_index, item.arg_type, rows, source_columns
                    )
                )
        materialised.append(tuple(values))

    if order_by is not None:
        comparator = _make_tuple_comparator(materialised, order_by)
        order = sorted(range(len(materialised)), key=cmp_to_key(comparator))
    else:
        order = list(range(len(materialised)))

    if select.limit is not None:
        order = order[: select.limit]

    width = len(bound_items)
    out_columns = [tuple(materialised[r][c] for r in order) for c in range(width)]
    return Table._from_storage(out_schema, out_columns)


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


def _make_plain_comparator(sort_key, order_by):
    def compare(a: int, b: int) -> int:
        va_keys = sort_key(a)
        vb_keys = sort_key(b)
        for spec, va, vb in zip(order_by, va_keys, vb_keys):
            c = _compare_scalar(va, vb, spec[2], spec[3])
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
