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
    order_item  := (ident | agg_name '(' ('*' | ident) ')')
                   [ASC | DESC] [NULLS FIRST | NULLS LAST]
    expr       := or_expr
    or_expr     := and_expr (OR and_expr)*
    and_expr    := cmp_expr (AND cmp_expr)*
    cmp_expr    := not_factor (cmp_op not_factor)?
    not_factor  := NOT not_factor | postfix
    postfix     := atom (IS [NOT] NULL)?
    atom        := '(' expr ')' | operand
    operand     := ident | literal
    literal     := TRUE | FALSE | [+-]? int64 | [+-]? finite float64
                   | single-quoted utf8

The supported aggregates are ``COUNT(*)``, ``COUNT(ident)`` and
``SUM`` / ``AVG`` / ``MIN`` / ``MAX`` applied to one column.  Without
GROUP BY the projection may contain aggregate expressions only; with
GROUP BY it may additionally contain the grouped columns.  A bare ``*``
must not be mixed with aggregates or grouping.  Aggregates are not
allowed inside WHERE, may not be nested, and aliases are not supported.

Operator precedence (highest first) is NOT, comparison, AND, OR; IS [NOT]
NULL is a postfix of its atom.  Comparison operands must be column
references or literals.  WHERE follows SQL three-valued logic: a normal
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
    )
)

# JOIN-related words are *contextual*: they are NOT in ``_KEYWORDS``, so they
# lex as ordinary identifiers and stay usable as column names (and, in
# multi-table queries, table names).  The parser recognises them by their
# case-folded text only in the FROM/JOIN positions where they are required;
# ``_parse_optional_join`` enumerates the words directly.

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
        if ch in ",()+-":
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
        if ch == ".":
            # ".5" stays a float literal; a lone dot is the qualifier in a
            # qualified name "table.column" and is left to the parser, which
            # rejects a stray dot as a syntax error.
            match = _NUMBER_RE.match(sql, i)
            if match and match.group(0) != ".":
                text = match.group(0)
                tokens.append(_Token("number", _parse_number(text), text))
                i = match.end()
                continue
            tokens.append(_Token("op", ".", "."))
            i += 1
            continue
        if ch.isdigit():
            match = _NUMBER_RE.match(sql, i)
            if match:
                text = match.group(0)
                tokens.append(_Token("number", _parse_number(text), text))
                i = match.end()
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
#   ("column", name, index, type_name, nullable)
#   ("cmp", op, left_node, right_node)
#   ("isnull", operand, negate)
#   ("not", operand)
#   ("and"|"or", left, right)
# Predicate nodes are boolean-typed (three-valued at evaluation time).
#
# Projection / ORDER BY reference items:
#   ("column_ref", name)                 -- a plain column name
#   ("agg", func_upper, arg_name|None)   -- an aggregate call; arg None = '*'
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _QName:
    """A parsed column reference, optionally table-qualified (``table.column``)."""

    table: str | None
    name: str
    table_quoted: bool = False


@dataclass(frozen=True)
class _Join:
    """The single optional join clause of a multi-table query."""

    join_type: str  # "inner" | "left"
    right_table: str
    right_quoted: bool
    left_key: _QName
    right_key: _QName


@dataclass(frozen=True)
class _RefItem:
    """One projection element or ORDER BY key as parsed."""

    kind: str  # "star" | "column" | "agg"
    name: str | None = None  # column name for kind == "column"
    func: str | None = None  # uppercase function name for kind == "agg"
    arg: str | None = None  # aggregate column argument; "" stands for '*'
    descending: bool = False
    nulls_first: bool | None = None  # None -> default (NULLs last)
    # Qualification, present only in multi-table (join) queries; ``None`` for
    # the single-file grammar.  ``arg_table`` qualifies an aggregate argument.
    table: str | None = None
    table_quoted: bool = False
    arg_table: str | None = None
    arg_table_quoted: bool = False


@dataclass
class _Select:
    items: tuple[_RefItem, ...]  # projection; may be a single ("star",) item
    table: str
    table_quoted: bool
    where: tuple | None
    group_by: tuple[_QName, ...] | None
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

    def __init__(self, tokens: list[_Token]):
        self.tokens = tokens
        self.pos = 0
        # Holds the most recently parsed aggregate argument as a _QName while
        # _parse_column_or_agg builds its _RefItem; None stands for '*'.
        self._agg_arg: _QName | None = None

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
        join = self._parse_optional_join()
        where = None
        if self._accept_keyword("where"):
            where = self._parse_or()
        group_by = None
        if self._accept_keyword("group"):
            self._expect_keyword("by")
            names = [self._parse_qname()]
            while self._accept_op(","):
                names.append(self._parse_qname())
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

    def _is_join_word(self, tok: _Token, word: str) -> bool:
        # Contextual recognition: a quoted "join"/"on"/... is an identifier,
        # never a keyword.
        return tok.kind == "ident" and tok.value.lower() == word

    def _expect_join_word(self, word: str) -> None:
        tok = self._peek()
        if not self._is_join_word(tok, word):
            raise QuerySyntaxError(f"expected {word.upper()}, got {tok.text!r}")
        self._next()

    def _parse_optional_join(self) -> _Join | None:
        """Parse ``[INNER | LEFT] JOIN right ON left.col = right.col``.

        Absent unless the token right after the left table is a contextual
        JOIN word.  Everything grammatical about the join is decided here so
        that a missing JOIN keyword, an illegal ON clause or a second join all
        surface as :class:`QuerySyntaxError` before any file is read.
        """
        tok = self._peek()
        if tok.kind == "eof":
            return None
        following = self.tokens[self.pos + 1]
        is_join_word = (
            tok.kind == "ident" and tok.value.lower() in ("inner", "left", "join")
        )
        # "INNER"/"LEFT" only starts a join when immediately followed by a bare
        # JOIN word; otherwise it is an ordinary table name and single-file
        # queries keep their historical (unknown-table) validation behaviour.
        has_join_keyword = (
            tok.kind == "ident" and tok.value.lower() == "join"
        ) or (
            is_join_word
            and following.kind == "ident"
            and following.value.lower() == "join"
        )
        if not is_join_word or not has_join_keyword:
            return None
        word = tok.value.lower()
        if word == "inner":
            self._next()
            self._expect_join_word("join")
            join_type = "inner"
        elif word == "left":
            self._next()
            self._expect_join_word("join")
            join_type = "left"
        else:  # bare JOIN implies INNER
            self._next()
            join_type = "inner"
        right_tok = self._expect_table_name()
        self._expect_join_word("on")
        left_key = self._parse_qname()
        self._expect_op("=")
        right_key = self._parse_qname()
        # A second join leaves another JOIN word in the token stream; the
        # clause parsers do not consume it and the trailing-input check in
        # parse() reports it.  Detect it directly for a precise message.
        extra = self._peek()
        if extra.kind == "ident" and extra.value.lower() in ("inner", "left", "join"):
            raise QuerySyntaxError("at most one JOIN is supported")
        return _Join(
            join_type=join_type,
            right_table=right_tok.value,
            right_quoted=right_tok.kind == "qident",
            left_key=left_key,
            right_key=right_key,
        )

    def _parse_qname(self) -> _QName:
        """Parse ``identifier`` or ``identifier.identifier``."""
        tok = self._peek()
        if tok.kind not in ("ident", "qident"):
            raise QuerySyntaxError(f"expected identifier, got {tok.text!r}")
        self._next()
        return self._qname_after_first(tok)

    def _qname_after_first(self, tok: _Token) -> _QName:
        if self._accept_op("."):
            col = self._peek()
            if col.kind not in ("ident", "qident"):
                raise QuerySyntaxError(
                    f"expected column name after '.', got {col.text!r}"
                )
            self._next()
            return _QName(tok.value, col.value, tok.kind == "qident")
        return _QName(None, tok.value)

    def _parse_projection(self) -> list[_RefItem]:
        items = [self._parse_select_item()]
        while self._accept_op(","):
            items.append(self._parse_select_item())
        # Aliases are not supported; a trailing identifier (including the
        # spelling "as") where FROM is expected names one.  Plain queries keep
        # their historical QuerySyntaxError classification; aggregate queries
        # treat the unsupported alias as a validation error.
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
        return items

    def _parse_select_item(self) -> _RefItem:
        tok = self._peek()
        if tok.kind == "star":
            self._next()
            return _RefItem("star")
        return self._parse_column_or_agg()

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

    def _parse_column_or_agg(self) -> _RefItem:
        tok = self._peek()
        if tok.kind not in ("ident", "qident"):
            raise QuerySyntaxError(f"expected identifier, got {tok.text!r}")
        nxt = self.tokens[self.pos + 1]
        # ``name.column`` (or ``name.`` followed by something) is a qualified
        # column reference, even when the first word is spelled like an
        # aggregate function.
        if nxt.kind == "op" and nxt.value == ".":
            qname = self._parse_qname()
            return _RefItem(
                "column",
                name=qname.name,
                table=qname.table,
                table_quoted=qname.table_quoted,
            )
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
        self._parse_aggregate_args(func)
        self._expect_op(")")
        arg = self._agg_arg
        return _RefItem(
            "agg",
            func=func.upper(),
            arg=arg.name if arg is not None else "",
            arg_table=arg.table if arg is not None else None,
            arg_table_quoted=arg.table_quoted if arg is not None else False,
        )

    def _parse_aggregate_args(self, func: str) -> None:
        # Exactly one argument: '*' for COUNT, otherwise one (qualified) name.
        tok = self._peek()
        if tok.kind == "star":
            if func != "count":
                raise QuerySyntaxError(f"{func.upper()} does not accept '*'")
            self._next()
            self._agg_arg = None
            return
        if tok.kind not in ("ident", "qident"):
            raise QuerySyntaxError(
                f"{func.upper()} requires one column argument, got {tok.text!r}"
            )
        qname = self._parse_qname()
        # A nested call is only detectable once the (qualified) argument name
        # has been consumed.
        nxt = self._peek()
        if nxt.kind == "op" and nxt.value == "(":
            if qname.table is None and qname.name.lower() in _AGG_NAMES:
                raise QueryValidationError(
                    f"aggregate functions must not be nested: "
                    f"{func.upper()}({qname.name.upper()}(...))"
                )
            raise QuerySyntaxError(
                f"{func.upper()} argument must be a column name, not a function call"
            )
        self._agg_arg = qname

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
        node = self._parse_atom()
        if self._accept_keyword("is"):
            negate = self._accept_keyword("not")
            self._expect_keyword("null")
            node = ("isnull", node, negate)
        return node

    def _parse_atom(self) -> tuple:
        if self._accept_op("("):
            node = self._parse_or()
            self._expect_op(")")
            return node
        return self._parse_operand()

    def _parse_operand(self) -> tuple:
        sign = 1
        tok = self._peek()
        if tok.kind == "op" and tok.value in ("+", "-"):
            if tok.value == "-":
                sign = -1
            self._next()
            tok = self._peek()
            if tok.kind != "number" or isinstance(tok.value, bool):
                raise QuerySyntaxError("sign must be followed by a numeric literal")
        if tok.kind == "number":
            self._next()
            value = tok.value
            if isinstance(value, bool):
                if sign == -1:
                    raise QuerySyntaxError("boolean literal cannot be negated")
                return ("literal", value, "bool")
            if isinstance(value, int):
                value *= sign
                if not (-(2**63) <= value <= 2**63 - 1):
                    raise QuerySyntaxError("integer literal is outside the int64 range")
            else:
                value = math.copysign(value, sign) if sign == -1 else value
                if not math.isfinite(value):
                    raise QuerySyntaxError("float literal must be finite")
            type_name = "int64" if isinstance(value, int) else "float64"
            return ("literal", value, type_name)
        if tok.kind == "string":
            if sign == -1:
                raise QuerySyntaxError("string literal cannot be negated")
            self._next()
            return ("literal", tok.value, "utf8")
        if tok.kind in ("ident", "qident"):
            if sign == -1:
                raise QuerySyntaxError("column reference cannot be negated")
            qname = self._parse_qname()
            nxt = self._peek()
            if (
                qname.table is None
                and qname.name.lower() in _AGG_NAMES
                and nxt.kind == "op"
                and nxt.value == "("
            ):
                raise QueryValidationError(
                    f"aggregate {qname.name.upper()}(...) is not allowed in WHERE"
                )
            return ("column", qname.table, qname.name, None, None, None)
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

    kind: str  # "column" | "agg"
    output_name: str
    # kind == "column":
    col_index: int = -1
    # kind == "agg":
    func: str = ""
    arg_index: int = -1  # -1 means COUNT(*)
    arg_type: str = ""
    out_type: str = ""
    nullable: bool = True


def _bind_select(select: _Select, schema: Schema, table_name: str = "input") -> dict:
    table_matches = (
        select.table == table_name
        if select.table_quoted
        else select.table.lower() == table_name.lower()
    )
    if not table_matches:
        raise QueryValidationError(
            f"unknown table {select.table!r}; only {table_name!r} is available here"
        )
    return _bind_parsed(select, schema)


def _select_has_qualification(select: _Select) -> bool:
    """True if any column reference carries a ``table.`` prefix."""
    if select.join is not None:
        return True  # join selects take the multi-table path, never this one
    for item in select.items:
        if item.table is not None or item.arg_table is not None:
            return True
    for item in select.order_by or ():
        if item.table is not None or item.arg_table is not None:
            return True
    for qname in select.group_by or ():
        if qname.table is not None:
            return True

    def walk(node) -> bool:
        if not isinstance(node, tuple):
            return False
        if node[0] == "column":
            return node[1] is not None
        return any(walk(child) for child in node[1:] if isinstance(child, tuple))

    return select.where is not None and walk(select.where)


def _bind_parsed(select: _Select, schema: Schema) -> dict:
    def resolve(node):
        # Single-file grammar: references are unqualified and resolve by name.
        if node[1] is not None:
            raise QueryValidationError(
                f"qualified column {node[1]}.{node[2]} is only valid in a multi-table query"
            )
        try:
            index = schema.index(node[2])
        except KeyError:
            raise QueryValidationError(f"unknown column: {node[2]!r}") from None
        col = schema.columns[index]
        return index, col

    where = _bind_expr(select.where, resolve) if select.where is not None else None
    if where is not None and where[0] in ("literal", "column"):
        type_name = _operand_type(where)
        if type_name != "bool":
            raise QueryValidationError(
                f"WHERE clause must be boolean, got {type_name}"
            )

    if not select.is_aggregate_query:
        return _bind_plain(select, schema, where)
    return _bind_aggregate(select, schema, where)


def _bind_plain(select: _Select, schema: Schema, where) -> dict:
    # Non-aggregate path: '*' or a comma-separated list of plain columns.
    if select.star:
        indices = tuple(range(len(schema.columns)))
    else:
        seen: set[str] = set()
        indices_list: list[int] = []
        for item in select.items:
            if item.kind == "star":
                # A star mixed into a plain projection remains grammatical.
                raise QuerySyntaxError("'*' cannot be mixed with other projection items")
            name = item.name
            if name in seen:
                raise QueryValidationError(f"duplicate column in projection: {name!r}")
            seen.add(name)
            try:
                indices_list.append(schema.index(name))
            except KeyError:
                raise QueryValidationError(f"unknown column: {name!r}") from None
        indices = tuple(indices_list)
    order_by = _bind_plain_order_by(select.order_by, schema)
    return {
        "mode": "plain",
        "indices": indices,
        "where": where,
        "order_by": order_by,
    }


def _bind_plain_order_by(select_order_by, schema: Schema):
    if select_order_by is None:
        return None
    bound: list[tuple[int, bool, bool]] = []
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
        try:
            col_index = schema.index(name)
        except KeyError:
            raise QueryValidationError(
                f"unknown ORDER BY column: {name!r}"
            ) from None
        nulls_first = item.nulls_first if item.nulls_first is not None else False
        bound.append((col_index, item.descending, nulls_first))
    return tuple(bound)


def _bind_aggregate(select: _Select, schema: Schema, where) -> dict:
    if select.star:
        raise QueryValidationError("'*' cannot be combined with aggregates or GROUP BY")

    # Resolve GROUP BY columns first: order matters for the grouping key, and
    # duplicates / unknown columns are rejected here.
    group_indices: list[int] = []
    group_seen: set[str] = set()
    for qname in select.group_by or ():
        name = qname.name
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


def _bind_expr(node: tuple, resolve) -> tuple:
    tag = node[0]
    if tag == "literal":
        return node
    if tag == "column":
        index, col = resolve(node)
        return ("column", node[1], col.name, index, col.type, col.nullable)
    if tag == "not":
        operand = _bind_expr(node[1], resolve)
        _require_boolean(operand, "NOT")
        return ("not", operand)
    if tag in ("and", "or"):
        left = _bind_expr(node[1], resolve)
        right = _bind_expr(node[2], resolve)
        _require_boolean(left, tag.upper())
        _require_boolean(right, tag.upper())
        return (tag, left, right)
    if tag == "isnull":
        operand = _bind_expr(node[1], resolve)
        if operand[0] not in ("literal", "column"):
            raise QuerySyntaxError(
                "IS NULL operand must be a column reference or a literal"
            )
        return ("isnull", operand, node[2])
    if tag == "cmp":
        op = node[1]
        left = _bind_expr(node[2], resolve)
        right = _bind_expr(node[3], resolve)
        if left[0] not in ("literal", "column") or right[0] not in ("literal", "column"):
            raise QuerySyntaxError("comparison operands must be column references or literals")
        left_t = _operand_type(left)
        right_t = _operand_type(right)
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


def _operand_type(node: tuple) -> str:
    return node[2] if node[0] == "literal" else node[4]


def _require_boolean(node: tuple, context: str) -> None:
    if node[0] in ("cmp", "isnull", "not", "and", "or"):
        return
    type_name = node[2] if node[0] == "literal" else node[4]
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
        return row[node[3]]
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
    _reject_single_file_join(select)
    table = read_file(path)
    return _run_query(table, select)


def _reject_single_file_join(select: _Select) -> None:
    """Reproduce the single-file grammar for statements a join could relax.

    Before joins existed a stray dot was a lexical error and a JOIN clause was
    an unexpected-trailing-input syntax error, both raised before the file was
    read.  Keep that classification (and pre-read timing) for ``query_file``.
    """
    if select.join is not None:
        raise QuerySyntaxError("JOIN is only supported by query_files")
    if _select_has_qualification(select):
        raise QuerySyntaxError("unexpected '.'; qualified columns require a JOIN")


def query_files(sources: Mapping[str, Any], sql: str) -> Table:
    """Run a two-file equi-join ``sql`` query against the files in ``sources``.

    ``sources`` is a non-empty mapping of table name to file path (an
    :class:`os.PathLike`).  The statement has the shape::

        SELECT ... FROM left_t
        ( INNER | LEFT ) JOIN right_t ON left_t.a = right_t.b
        [WHERE ...] [GROUP BY ...] [ORDER BY ...] [LIMIT n]

    Apart from ``COUNT(*)`` every column reference must be qualified as
    ``table.column``.  ``SELECT *`` outputs the left schema followed by the
    right schema, with columns named ``table.column``.  Join rows are
    expanded in left-file then right-file order before WHERE/GROUP BY/
    aggregate/ORDER BY/LIMIT are applied with their single-file semantics.

    The ``sources`` argument is validated (raising :class:`ValueError`) and
    the statement is parsed (raising :class:`QuerySyntaxError`) before any
    file is opened.  Unknown tables/columns, unqualified references,
    duplicate tables, wrong-sourced or type-incompatible join keys and
    duplicate result columns raise :class:`QueryValidationError`.
    """
    # Argument validation never touches the filesystem.
    if not isinstance(sources, Mapping) or len(sources) == 0:
        raise ValueError("sources must be a non-empty mapping of table name to path")
    for key, value in sources.items():
        if not isinstance(key, str) or not key:
            raise ValueError("sources keys must be non-empty table-name strings")
        if not isinstance(value, os.PathLike):
            raise ValueError("sources values must be path objects")

    tokens = _tokenize(sql)
    select = _Parser(tokens).parse()
    if select.join is None:
        raise QueryValidationError(
            "query_files requires a query with one INNER or LEFT JOIN"
        )

    join = select.join
    left_name = _resolve_source_table(select.table, select.table_quoted, sources)
    right_name = _resolve_source_table(
        join.right_table, join.right_quoted, sources
    )
    if left_name == right_name:
        raise QueryValidationError(
            f"table {left_name!r} may be joined only once (aliases are not supported)"
        )

    left_table = read_file(sources[left_name])
    right_table = read_file(sources[right_name])

    wide, rewritten = _prepare_join(
        select, left_name, left_table, join.join_type, right_name, right_table
    )
    return _run_query(wide, rewritten)


def query_table(table: Table, sql: str) -> Table:
    """Apply ``sql`` (``FROM input``) to an in-memory table."""
    tokens = _tokenize(sql)
    select = _Parser(tokens).parse()
    _reject_single_file_join(select)
    return _run_query(table, select)


def _resolve_source_table(name: str, quoted: bool, sources: Mapping[str, str]) -> str:
    """Map a SQL table spelling to the canonical key of ``sources``.

    Quoted names match exactly; bare names match case-insensitively, as with
    the fixed ``input`` table of single-file queries.
    """
    if quoted:
        if name in sources:
            return name
        raise QueryValidationError(f"unknown table: {name!r}")
    matches = [key for key in sources if key.lower() == name.lower()]
    if not matches:
        raise QueryValidationError(f"unknown table: {name!r}")
    if len(matches) > 1:
        raise QueryValidationError(
            f"table name {name!r} is ambiguous among {sorted(matches)!r}"
        )
    return matches[0]


def _prepare_join(
    select: _Select,
    left_name: str,
    left_table: Table,
    join_type: str,
    right_name: str,
    right_table: Table,
) -> tuple[Table, _Select]:
    """Validate the join against both schemas and materialise the wide table.

    Returns the wide (joined) table and a rewritten single-table
    :class:`_Select` whose unqualified names are the wide column names
    (``table.column``), so the existing binder and runner apply unchanged.
    """
    join = select.join
    left_cols = left_table.schema.columns
    right_cols = right_table.schema.columns

    def require_qualified(qname: _QName, context: str) -> None:
        if qname.table is None:
            raise QueryValidationError(
                f"{context} column {qname.name!r} must be qualified as table.column"
            )

    def resolve_side(qname: _QName, context: str) -> tuple[str, int, ColumnSchema]:
        require_qualified(qname, context)
        if qname.table_quoted:
            table_match = qname.table if qname.table in (left_name, right_name) else None
        else:
            table_matches = [
                n for n in (left_name, right_name) if n.lower() == qname.table.lower()
            ]
            table_match = table_matches[0] if len(table_matches) == 1 else None
            if len(table_matches) > 1:
                raise QueryValidationError(
                    f"table name {qname.table!r} is ambiguous"
                )
        if table_match is None:
            raise QueryValidationError(f"unknown table: {qname.table!r}")
        side_table = left_table if table_match == left_name else right_table
        try:
            col_index = side_table.schema.index(qname.name)
        except KeyError:
            raise QueryValidationError(
                f"unknown column: {table_match}.{qname.name}"
            ) from None
        return table_match, col_index, side_table.schema.columns[col_index]

    # --- join keys: left key must come from the left table, right key right -
    l_side, l_idx, l_col = resolve_side(join.left_key, "join")
    if l_side != left_name:
        raise QueryValidationError(
            f"left join key {join.left_key.table}.{join.left_key.name} "
            f"must reference the left table {left_name!r}"
        )
    r_side, r_idx, r_col = resolve_side(join.right_key, "join")
    if r_side != right_name:
        raise QueryValidationError(
            f"right join key {join.right_key.table}.{join.right_key.name} "
            f"must reference the right table {right_name!r}"
        )
    if not _keys_compatible(l_col.type, r_col.type):
        raise QueryValidationError(
            f"join keys have incompatible types {l_col.type} and {r_col.type}"
        )

    # --- materialise the wide rows in left-then-right original order --------
    left_storage = left_table._columns
    right_storage = right_table._columns
    n_right = len(right_cols)
    l_key_col = left_storage[l_idx]
    r_key_col = right_storage[r_idx]

    # Index right rows by key for the common non-NULL case; NULL never matches
    # and rows are emitted grouped by left row, so iterate per left row.
    wide_rows: list[tuple] = []
    for li in range(left_table.row_count):
        lv = l_key_col[li]
        left_vals = tuple(col[li] for col in left_storage)
        if lv is None:
            matches: list[int] = []
        else:
            matches = [
                ri
                for ri in range(right_table.row_count)
                if (rv := r_key_col[ri]) is not None and _key_equal(lv, rv)
            ]
        if matches:
            for ri in matches:
                wide_rows.append(
                    left_vals + tuple(col[ri] for col in right_storage)
                )
        elif join_type == "left":
            wide_rows.append(left_vals + (None,) * n_right)

    wide_schema = Schema(
        tuple(
            ColumnSchema(f"{left_name}.{c.name}", c.type, c.nullable)
            for c in left_cols
        )
        + tuple(
            ColumnSchema(
                f"{right_name}.{c.name}",
                c.type,
                # Right-side columns of a LEFT JOIN can be NULL-padded.
                True if join_type == "left" else c.nullable,
            )
            for c in right_cols
        )
    )
    width = len(wide_schema.columns)
    wide_storage = [
        tuple(row[c] for row in wide_rows) for c in range(width)
    ]
    wide = Table._from_storage(wide_schema, wide_storage)

    # --- rewrite the remaining clauses to wide, unqualified names -----------
    def wide_name(qname: _QName, context: str) -> str:
        side, col_index, col = resolve_side(qname, context)
        return f"{side}.{col.name}"

    rewritten_items = tuple(
        _rewrite_ref_item(item, wide_name) for item in select.items
    )
    rewritten_group = (
        tuple(
            _QName(None, wide_name(qname, "GROUP BY"))
            for qname in select.group_by
        )
        if select.group_by is not None
        else None
    )
    rewritten_order = (
        tuple(
            _rewrite_ref_item(item, wide_name) for item in select.order_by
        )
        if select.order_by is not None
        else None
    )
    rewritten_where = (
        _rewrite_where(select.where, wide_name)
        if select.where is not None
        else None
    )
    rewritten = _Select(
        items=rewritten_items,
        table="input",
        table_quoted=False,
        where=rewritten_where,
        group_by=rewritten_group,
        order_by=rewritten_order,
        limit=select.limit,
        join=None,
    )
    return wide, rewritten


def _keys_compatible(left_type: str, right_type: str) -> bool:
    # Equal types always join; across types only int64/float64 may mix.
    if left_type == right_type:
        return True
    return {left_type, right_type} == {"int64", "float64"}


def _key_equal(left, right) -> bool:
    # NULL is filtered by the caller.  int64/float64 compare numerically; the
    # stored values are finite and the types are validated as comparable.
    return left == right


def _rewrite_ref_item(item: _RefItem, wide_name) -> _RefItem:
    if item.kind == "star":
        return item
    if item.kind == "column":
        qname = _QName(item.table, item.name, item.table_quoted)
        return _RefItem(
            "column",
            name=wide_name(qname, "projection"),
            descending=item.descending,
            nulls_first=item.nulls_first,
        )
    # aggregate
    if item.arg == "":
        arg_name = ""
        arg_table = None
    else:
        qname = _QName(item.arg_table, item.arg, item.arg_table_quoted)
        arg_name = wide_name(qname, "aggregate argument")
        arg_table = None
    return _RefItem(
        "agg",
        name=None,
        func=item.func,
        arg=arg_name,
        descending=item.descending,
        nulls_first=item.nulls_first,
    )


def _rewrite_where(node: tuple, wide_name) -> tuple:
    tag = node[0]
    if tag == "literal":
        return node
    if tag == "column":
        qname = _QName(node[1], node[2])
        return ("column", None, wide_name(qname, "WHERE"), None, None, None)
    if tag in ("not",):
        return ("not", _rewrite_where(node[1], wide_name))
    if tag in ("and", "or"):
        return (
            tag,
            _rewrite_where(node[1], wide_name),
            _rewrite_where(node[2], wide_name),
        )
    if tag == "isnull":
        return ("isnull", _rewrite_where(node[1], wide_name), node[2])
    if tag == "cmp":
        return (
            "cmp",
            node[1],
            _rewrite_where(node[2], wide_name),
            _rewrite_where(node[3], wide_name),
        )
    raise QuerySyntaxError(f"unsupported expression: {tag}")  # pragma: no cover


def _run_query(table: Table, select: _Select) -> Table:
    if select.join is not None:
        raise QueryValidationError(
            "JOIN queries must be run with query_files"
        )
    if _select_has_qualification(select):
        # query_file already rejects this before reading; guard the in-memory
        # entry point with the same historical classification.
        raise QuerySyntaxError("unexpected '.'; qualified columns require a JOIN")
    bound = _bind_select(select, table.schema)
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
    indices = bound["indices"]
    order_by = bound["order_by"]
    source_columns = table._columns
    if order_by is not None:
        comparator = _make_row_comparator(source_columns, order_by)
        selected = sorted(selected, key=cmp_to_key(comparator))

    if select.limit is not None:
        selected = selected[: select.limit]

    out_columns = [
        tuple(source_columns[index][i] for i in selected) for index in indices
    ]
    out_schema = Schema([table.schema.columns[i] for i in indices])
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


def _make_row_comparator(
    source_columns: tuple[tuple, ...],
    order_by: tuple[tuple[int, bool, bool], ...],
):
    def compare(a: int, b: int) -> int:
        for col_index, descending, nulls_first in order_by:
            va = source_columns[col_index][a]
            vb = source_columns[col_index][b]
            c = _compare_scalar(va, vb, descending, nulls_first)
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
