"""Pure SQL parsing stage: tokeniser, parsed statement tree, grammar.

This is the first stage of the query pipeline and is strictly syntactic:
it never touches a file, a schema or any later stage's data, so a
:class:`QuerySyntaxError` (and the few grammar-level
:class:`QueryValidationError` rejections) is guaranteed to surface before
any source is opened.  Its output is the parsed statement tree consumed
by the name-rewrite stage (:mod:`columnar_analytics.rewrite`) and the
binder (:mod:`columnar_analytics.binder`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .errors import QuerySyntaxError, QueryValidationError

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


def _parse_statement(sql: str, allow_join: bool = False) -> _Select:
    """Tokenise and parse one statement (pure syntax; no file is touched)."""
    return _Parser(_tokenize(sql), allow_join=allow_join).parse()
