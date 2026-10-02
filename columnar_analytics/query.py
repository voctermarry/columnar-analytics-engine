"""Single-file SQL query layer.

Public API:

* :func:`query_file` -- run a ``SELECT ... FROM input [WHERE ...]`` query
  against one columnar file and return a :class:`~columnar_analytics.format.Table`
* :func:`query_files` -- run a statement (optionally with one equi-join)
  against a table-name to path mapping
* :func:`explain_file` / :func:`explain_files` -- parse, bind and plan the
  same statements without executing them; only file metadata is read
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
    operand     := ident | literal
    literal     := TRUE | FALSE | [+-]? int64 | [+-]? finite float64
                   | single-quoted utf8

The supported aggregates are ``COUNT(*)``, ``COUNT(ident)`` and
``SUM`` / ``AVG`` / ``MIN`` / ``MAX`` applied to one column.  Without
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
never trigger division-by-zero or overflow in the projection.  ORDER BY
may name a SELECT alias (an alias shadows an input column of the same
name) and sorts on the expression's result type with the usual NULL
placement and stability rules.  Aggregate queries do not accept scalar
expressions: GROUP BY keys and aggregate arguments stay plain column
references, and mixing a scalar expression into an aggregate query
raises :class:`QueryValidationError`.

Operator precedence (highest first) is NOT, comparison, AND, OR; IS [NOT]
NULL is a postfix of its scalar operand.  Comparison operands may be
column references, literals or scalar arithmetic expressions.  WHERE
follows SQL three-valued logic: a normal
comparison against NULL yields UNKNOWN, UNKNOWN propagates through the
logical operators, and only TRUE rows are returned.

After WHERE, rows are grouped in GROUP BY column order (a NULL key forms
its own group); without an explicit ORDER BY groups come out in the order
of their first selected row.  ORDER BY may only name selected grouped
columns or selected aggregate expressions; ASC is the default, NULLs sort
last regardless of direction unless NULLS FIRST / NULLS LAST is given, and
groups equal on every sort key keep that first-row order.  LIMIT then
keeps the first N groups.  Without GROUP BY and aggregates an aggregate
query over zero selected rows still yields one output row (COUNT 0, the
other aggregates NULL); with GROUP BY it yields zero rows.

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

from .format import ColumnSchema, Schema, Table, inspect_file, read_file

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
#   ("arith", op, left_node, right_node)              -- as parsed (+ - * /)
#   ("arith", op, left_node, right_node, type_name)   -- after binding
#   ("unary", operand, negate)                        -- as parsed
#   ("unary", operand, negate, type_name)             -- after binding
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
    expr: tuple | None = None  # parsed scalar expression for kind == "expr"
    descending: bool = False
    nulls_first: bool | None = None  # None -> default (NULLs last)
    # Table qualifiers (two-table queries only; None when unqualified):
    table: str | None = None
    table_quoted: bool = False
    arg_table: str | None = None
    arg_table_quoted: bool = False


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
            self._reject_expression_in_agg_arg(func)
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
        self._reject_expression_in_agg_arg(func)
        return (tok.value, None, False)

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
            ):
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
    if where is not None and where[0] in ("literal", "column", "arith", "unary"):
        type_name = _expr_type(where)
        if type_name != "bool":
            raise QueryValidationError(
                f"WHERE clause must be boolean, got {type_name}"
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
                if out_type not in ("int64", "float64"):
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
    order_by = _bind_plain_order_by(select.order_by, schema, aliases)
    return {
        "mode": "plain",
        "items": tuple(bound_items),
        "where": where,
        "order_by": order_by,
    }


def _bind_plain_order_by(select_order_by, schema: Schema, aliases: dict):
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
        if operand[0] not in ("literal", "column", "arith", "unary"):
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
    if tag == "cmp":
        op = node[1]
        left = _bind_expr(node[2], schema)
        right = _bind_expr(node[3], schema)
        if left[0] not in ("literal", "column", "arith", "unary") or right[0] not in (
            "literal", "column", "arith", "unary"
        ):
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


def _referenced_source_paths(sources: Any, sql: str) -> tuple:
    """Resolve the file paths of the tables the statement actually reads.

    Parsing-only counterpart of :func:`query_files`: no file is ever opened.
    Returns the paths in ``FROM`` / ``JOIN`` order (one entry for a
    single-table statement, two for a join).  Source validation raises
    :class:`ValueError`; lexical / grammatical errors raise
    :class:`QuerySyntaxError`; unknown or ambiguous table names raise
    :class:`QueryValidationError`.
    """
    paths = _validate_sources(sources)
    tokens = _tokenize(sql)
    select = _Parser(tokens, allow_join=True).parse()
    left_key = _resolve_table_ref(select.table, select.table_quoted, tuple(paths))
    referenced = [paths[left_key]]
    if select.join is not None:
        right_key = _resolve_table_ref(
            select.join.table, select.join.table_quoted, tuple(paths)
        )
        if right_key == left_key:
            raise QueryValidationError(f"duplicate table {right_key!r} in join")
        referenced.append(paths[right_key])
    return tuple(referenced)


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
    if tag == "not":
        return ("not", _rewrite_expr(node[1], resolve))
    if tag in ("and", "or"):
        return (tag, _rewrite_expr(node[1], resolve), _rewrite_expr(node[2], resolve))
    if tag == "isnull":
        return ("isnull", _rewrite_expr(node[1], resolve), node[2])
    if tag == "cmp":
        return ("cmp", node[1], _rewrite_expr(node[2], resolve), _rewrite_expr(node[3], resolve))
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover - defensive


def _build_joined_schema(
    left_key: str,
    left_schema: Schema,
    right_key: str,
    right_schema: Schema,
    join: _Join,
) -> Schema:
    """Validate the ON keys and derive the combined post-join schema."""
    left_col = join.left_key[2]
    right_col = join.right_key[2]
    try:
        left_idx = left_schema.index(left_col)
    except KeyError:
        raise QueryValidationError(
            f"unknown column: {f'{left_key}.{left_col}'!r}"
        ) from None
    try:
        right_idx = right_schema.index(right_col)
    except KeyError:
        raise QueryValidationError(
            f"unknown column: {f'{right_key}.{right_col}'!r}"
        ) from None
    left_type = left_schema.columns[left_idx].type
    right_type = right_schema.columns[right_idx].type
    if left_type != right_type and not {left_type, right_type} <= {"int64", "float64"}:
        raise QueryValidationError(
            f"join key types are incompatible: {left_type} and {right_type}"
        )

    # The combined schema is left columns then right columns, named
    # "table.column"; a LEFT JOIN makes every right-side column nullable.
    combined_columns = [
        ColumnSchema(f"{left_key}.{col.name}", col.type, col.nullable)
        for col in left_schema.columns
    ] + [
        ColumnSchema(
            f"{right_key}.{col.name}",
            col.type,
            True if join.kind == "left" else col.nullable,
        )
        for col in right_schema.columns
    ]
    names = [col.name for col in combined_columns]
    if len(set(names)) != len(names):
        raise QueryValidationError("joined tables produce duplicate column names")
    return Schema(combined_columns)


def _build_joined_table(
    left_key: str, left: Table, right_key: str, right: Table, join: _Join
) -> Table:
    schema = _build_joined_schema(
        left_key, left.schema, right_key, right.schema, join
    )
    left_idx = left.schema.index(join.left_key[2])
    right_idx = right.schema.index(join.right_key[2])
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
    # Execution order: WHERE (done by the caller) -> sort keys -> stable sort
    # -> LIMIT -> result expressions, so rows filtered out or cut by LIMIT
    # never evaluate the SELECT expressions.
    if order_by is not None:
        comparator = _make_row_comparator(source_columns, order_by, selected)
        selected = sorted(selected, key=cmp_to_key(comparator))

    if select.limit is not None:
        selected = selected[: select.limit]

    out_columns = []
    out_schema_columns = []
    for item in items:
        if item.kind == "column":
            out_columns.append(
                tuple(source_columns[item.col_index][i] for i in selected)
            )
            out_schema_columns.append(table.schema.columns[item.col_index])
        else:  # "expr"
            out_columns.append(
                tuple(
                    _eval(item.expr, _row_values(source_columns, i))
                    for i in selected
                )
            )
            out_schema_columns.append(
                ColumnSchema(item.output_name, item.out_type, nullable=item.nullable)
            )
    out_schema = Schema(out_schema_columns)
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
    return _build_explain(sources, schema, select, bound, join=None)


def explain_files(sources: Any, sql: str) -> dict:
    """Produce the query plan for ``sql`` against the mapped tables.

    Like :func:`query_files` for parsing, source resolution and binding,
    but only the metadata of the referenced files is read: unreferenced
    sources are never opened and no data section is ever touched.  The
    returned value is a JSON-serialisable ordered dict with the fixed
    top-level keys ``sources``, ``operators`` and ``output``.

    A non-mapping or empty ``sources``, non-string keys or non-path
    values raise :class:`ValueError` before any file is touched;
    :class:`QuerySyntaxError` is raised before files are read as well.
    Unknown tables or columns and other binding problems raise
    :class:`QueryValidationError`; malformed metadata or a declared-size
    mismatch raise :class:`~columnar_analytics.format.ColumnarFormatError`;
    other I/O failures propagate as :class:`OSError`.
    """
    paths = _validate_sources(sources)
    tokens = _tokenize(sql)
    select = _Parser(tokens, allow_join=True).parse()
    left_key = _resolve_table_ref(select.table, select.table_quoted, tuple(paths))
    if select.join is None:
        rewritten = _rewrite_select(select, _single_table_resolver(left_key))
        left_metadata = inspect_file(paths[left_key])
        left_schema = _schema_from_metadata(left_metadata)
        sources = ((left_key, left_metadata, left_schema),)
        bound = _bind_select(rewritten, left_schema, expected_table=None)
        return _build_explain(sources, left_schema, rewritten, bound, join=None)

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
    # not need the files and run before any metadata is read.
    rewritten = _rewrite_select(select, _join_resolver(left_key, right_key))
    left_metadata = inspect_file(paths[left_key])
    right_metadata = inspect_file(paths[right_key])
    left_schema = _schema_from_metadata(left_metadata)
    right_schema = _schema_from_metadata(right_metadata)
    combined_schema = _build_joined_schema(
        left_key, left_schema, right_key, right_schema, join
    )
    sources = (
        (left_key, left_metadata, left_schema),
        (right_key, right_metadata, right_schema),
    )
    bound = _bind_select(rewritten, combined_schema, expected_table=None)
    return _build_explain(
        sources,
        combined_schema,
        rewritten,
        bound,
        join=(left_key, right_key, join),
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
    for entry in bound["order_by"] or ():
        if entry[0] == "col":
            indices.add(entry[1])
        elif entry[0] == "expr":
            _collect_expr_columns(entry[1], indices)
        # Aggregate ORDER BY entries index a selected output column, which
        # is already covered by the projection scan above.
    return indices


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


def _build_explain(
    sources: tuple,
    schema: Schema,
    select: _Select,
    bound: Mapping,
    join: tuple | None,
) -> dict:
    referenced = _collect_required_indices(bound)
    if join is not None:
        # The ON keys feed the join even when neither is projected.
        left_key, right_key, join_node = join
        for key_name, column_name in (
            (left_key, join_node.left_key[2]),
            (right_key, join_node.right_key[2]),
        ):
            referenced.add(schema.index(f"{key_name}.{column_name}"))

    operators: list = []
    referenced_names = {schema.columns[i].name for i in referenced}
    for key, _metadata, source_schema in sources:
        if join is None:
            required = [col.name for col in schema.columns if col.name in referenced_names]
        else:
            prefix = f"{key}."
            required = [
                col.name
                for col in source_schema.columns
                if f"{prefix}{col.name}" in referenced_names
            ]
        operators.append(
            {"operator": "Scan", "source": key, "required_columns": required}
        )

    if join is not None:
        operators.append(
            {
                "operator": "Join",
                "type": join_node.kind.upper(),
                "left": {"table": left_key, "column": join_node.left_key[2]},
                "right": {"table": right_key, "column": join_node.right_key[2]},
            }
        )

    if bound["where"] is not None:
        operators.append({"operator": "Filter", "condition": _expr_json(bound["where"])})

    if bound["mode"] == "aggregate":
        operators.append(
            {
                "operator": "Aggregate",
                "group_keys": [
                    schema.columns[index].name for index in bound["group_indices"]
                ],
                "aggregates": [
                    {
                        "function": item.func,
                        "argument": (
                            None
                            if item.arg_index < 0
                            else schema.columns[item.arg_index].name
                        ),
                        "output": item.output_name,
                    }
                    for item in bound["items"]
                    if item.kind == "agg"
                ],
            }
        )

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
        else:  # "expr"
            expression = _expr_json(item.expr)
        projections.append({"expression": expression, "output": item.output_name})
    operators.append({"operator": "Project", "expressions": projections})

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
