"""Single-file SQL query layer.

Public API:

* :func:`query_file` -- run a ``SELECT ... FROM input [WHERE ...]`` query
  against one columnar file and return a :class:`~columnar_analytics.format.Table`
* :class:`QuerySyntaxError` -- lexical / grammatical errors
* :class:`QueryValidationError` -- unknown columns, wrong table name,
  type-incompatible comparisons

The accepted grammar (keywords case-insensitive)::

    query       := SELECT ('*' | select_item (',' select_item)*)
                   FROM ident [WHERE expr]
                   [GROUP BY column_ref (',' column_ref)*]
                   [ORDER BY sort_item (',' sort_item)*]
                   [LIMIT uint]
    select_item := column_ref | aggregate
    aggregate   := func_name '(' ('*' | column_ref) ')'
    sort_item   := target [ASC | DESC] [NULLS FIRST | NULLS LAST]
    target      := column_ref | aggregate
    expr       := or_expr
    or_expr     := and_expr (OR and_expr)*
    and_expr    := cmp_expr (AND cmp_expr)*
    cmp_expr    := not_factor (cmp_op not_factor)?
    not_factor  := NOT not_factor | postfix
    postfix     := atom (IS [NOT] NULL)?
    atom        := '(' expr ')' | operand
    operand     := column_ref | literal | aggregate
    literal     := TRUE | FALSE | [+-]? int64 | [+-]? finite float64
                   | single-quoted utf8

Aggregates are COUNT/SUM/AVG/MIN/MAX; only COUNT accepts ``*`` and only as
its sole argument.  Without GROUP BY the projection may contain aggregates
only (and then aggregates every row surviving WHERE); with GROUP BY it may
additionally contain grouped columns.  Aliases and nested aggregates do not
exist.  ORDER BY may only name a selected grouped column or repeat a
selected aggregate expression.

Operator precedence (highest first) is NOT, comparison, AND, OR; IS [NOT]
NULL is a postfix of its atom.  Comparison operands must be column
references or literals.  WHERE follows SQL three-valued logic: a normal
comparison against NULL yields UNKNOWN, UNKNOWN propagates through the
logical operators, and only TRUE rows are returned.  Aggregates are not
predicates and are rejected inside WHERE at validation time.

After WHERE, rows are grouped by the GROUP BY column values in clause
order; a NULL key is one distinct group, and without ORDER BY groups are
emitted in the order their first surviving row appears in the file.
Aggregates are then computed per group (or once over every surviving row
when there is no GROUP BY), rows may be sorted by one or more selected
result columns, and LIMIT keeps the first N rows.  Without ORDER BY the
group / file order is preserved.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any

from .format import ColumnSchema, Schema, Table, read_file

__all__ = [
    "QuerySyntaxError",
    "QueryValidationError",
    "query_file",
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
        "as",
        "count",
        "sum",
        "avg",
        "min",
        "max",
    )
)

# Aggregate function keywords mapped to their canonical (upper-case) name.
_AGGREGATE_NAMES = frozenset(("count", "sum", "avg", "min", "max"))


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
        if ch.isdigit() or ch == ".":
            match = _NUMBER_RE.match(sql, i)
            if match:
                text = match.group(0)
                if text == ".":
                    raise QuerySyntaxError(f"unexpected character '.' at position {i}")
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
#   ("agg", func_name, ("star",) | ("column", name, ...))
#   ("cmp", op, left_node, right_node)
#   ("isnull", operand, negate)
#   ("not", operand)
#   ("and"|"or", left, right)
# Predicate nodes are boolean-typed (three-valued at evaluation time).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SelectItem:
    """One projection entry: a bare column name or a parsed aggregate."""

    column: str | None  # set for plain column references
    func: str | None  # canonical upper-case aggregate name
    arg: tuple  # ("star",) | ("column", name) | ("agg", ...); nested is invalid
    alias: bool = False  # an AS/bare alias followed the item; rejected for aggregates


@dataclass(frozen=True)
class _OrderItem:
    column: str | None  # set when the target is a plain column reference
    func: str | None  # set when the target is an aggregate
    arg: tuple
    descending: bool
    nulls_first: bool | None  # None -> the default (NULLs last, either direction)


@dataclass
class _Select:
    columns: tuple[_SelectItem, ...] | None  # None means '*'
    table: str
    table_quoted: bool
    where: tuple | None
    group_by: tuple[str, ...] | None
    order_by: tuple[_OrderItem, ...] | None
    limit: int | None


class _Parser:
    _CMP_OPS = frozenset(("=", "!=", "<", "<=", ">", ">="))

    def __init__(self, tokens: list[_Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> _Select:
        select = self._parse_select()
        if self._peek().kind != "eof":
            tok = self._peek()
            raise QuerySyntaxError(f"unexpected trailing input {tok.text!r}")
        return select

    def _parse_select(self) -> _Select:
        self._expect_keyword("select")
        if self._accept("star"):
            columns: tuple[_SelectItem, ...] | None = None
        else:
            columns = tuple(self._parse_projection())
        self._expect_keyword("from")
        table_tok = self._expect_table_name()
        where = None
        if self._accept_keyword("where"):
            where = self._parse_or()
        group_by = None
        if self._accept_keyword("group"):
            group_by = self._parse_group_by()
        order_by = None
        if self._accept_keyword("order"):
            self._expect_keyword("by")
            items = [self._parse_order_item()]
            while self._accept_op(","):
                items.append(self._parse_order_item())
            order_by = tuple(items)
        limit = None
        if self._accept_keyword("limit"):
            limit = self._parse_limit()
        return _Select(
            columns=columns,
            table=table_tok.value,
            table_quoted=table_tok.kind == "qident",
            where=where,
            group_by=group_by,
            order_by=order_by,
            limit=limit,
        )

    def _parse_group_by(self) -> tuple[str, ...]:
        self._expect_keyword("by")
        names = [self._expect_identifier_name()]
        while self._accept_op(","):
            names.append(self._expect_identifier_name())
        return tuple(names)

    def _parse_order_item(self) -> _OrderItem:
        column, func, arg = self._parse_order_target()
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
        return _OrderItem(column, func, arg, descending, nulls_first)

    def _parse_order_target(self) -> tuple[str | None, str | None, tuple | None]:
        tok = self._peek()
        if tok.kind == "keyword" and tok.value in _AGGREGATE_NAMES:
            func, arg = self._parse_aggregate_call()
            return None, func, arg
        return self._expect_identifier_name(), None, None

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

    def _parse_projection(self) -> list[_SelectItem]:
        items = [self._parse_select_item()]
        while self._accept_op(","):
            items.append(self._parse_select_item())
        return items

    def _parse_select_item(self) -> _SelectItem:
        tok = self._peek()
        if tok.kind == "keyword" and tok.value in _AGGREGATE_NAMES:
            func, arg = self._parse_aggregate_call()
            alias = self._parse_optional_alias()
            return _SelectItem(None, func, arg, alias)
        name = self._expect_identifier_name()
        return _SelectItem(name, None, None)

    def _parse_optional_alias(self) -> bool:
        """Consume an AS/bare alias after an aggregate, reporting it as such.

        Aliases are unsupported: an alias on a plain column is left untouched
        and surfaces as a plain syntax error, while an alias on an aggregate
        is flagged so binding can raise :class:`QueryValidationError`.
        """
        had_as = self._accept_keyword("as")
        tok = self._peek()
        if tok.kind in ("ident", "qident"):
            self._next()
            return True
        if had_as:
            raise QuerySyntaxError("expected alias identifier after AS")
        return False

    def _parse_aggregate_call(self) -> tuple[str, tuple]:
        """Parse ``FUNC ( arg )`` with the function keyword at head.

        ``arg`` is ``*``, a column reference or (grammatically) another
        aggregate call; the nested form is accepted here so that binding can
        reject it as a :class:`QueryValidationError` rather than a syntax
        error.
        """
        func_tok = self._next()
        func = func_tok.value.upper()
        self._expect_op("(")
        if self._accept("star"):
            arg: tuple = ("star",)
        elif self._peek().kind == "keyword" and self._peek().value in _AGGREGATE_NAMES:
            inner_func, inner_arg = self._parse_aggregate_call()
            arg = ("agg", inner_func, inner_arg)
        else:
            arg = ("column", self._expect_identifier_name())
        self._expect_op(")")
        return func, arg

    def _expect_table_name(self) -> _Token:
        tok = self._peek()
        if tok.kind in ("ident", "qident"):
            return self._next()
        raise QuerySyntaxError(f"expected table name, got {tok.text!r}")

    def _expect_identifier_name(self) -> str:
        tok = self._peek()
        if tok.kind in ("ident", "qident"):
            self._next()
            return tok.value
        raise QuerySyntaxError(f"expected identifier, got {tok.text!r}")

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
        if tok.kind == "keyword" and tok.value in _AGGREGATE_NAMES:
            func, arg = self._parse_aggregate_call()
            return ("agg", func, arg)
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
            self._next()
            return ("column", tok.value, None, None, None)
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
class _AggSpec:
    """A bound aggregate: function, its argument and its result shape."""

    func: str  # COUNT | SUM | AVG | MIN | MAX
    star_arg: bool
    arg_index: int  # source column index (ignored for COUNT(*))
    result_name: str  # e.g. "COUNT(*)" or "SUM(n)"
    result_type: str
    result_nullable: bool


@dataclass(frozen=True)
class _PlanItem:
    """One projection result: a grouped column or an aggregate."""

    kind: str  # "group" | "agg"
    source_index: int  # group column; for aggregates the argument column
    agg: _AggSpec | None


@dataclass
class _Plan:
    aggregate: bool
    where: tuple | None
    limit: int | None
    # Plain (non-aggregate) queries:
    indices: tuple[int, ...] | None
    source_order: tuple[tuple[int, bool, bool], ...] | None
    # Aggregate queries:
    grouped: bool
    group_indices: tuple[int, ...] | None
    items: tuple[_PlanItem, ...] | None
    result_schema: Schema | None
    result_order: tuple[tuple[int, bool, bool], ...] | None


def _bind_select(select: _Select, schema: Schema) -> _Plan:
    table_matches = (
        select.table == "input"
        if select.table_quoted
        else select.table.lower() == "input"
    )
    if not table_matches:
        raise QueryValidationError(f"unknown table {select.table!r}; only 'input' is supported")

    where = _bind_expr(select.where, schema) if select.where is not None else None
    if where is not None and where[0] in ("literal", "column"):
        type_name = _operand_type(where)
        if type_name != "bool":
            raise QueryValidationError(
                f"WHERE clause must be boolean, got {type_name}"
            )

    group_indices: tuple[int, ...] | None = None
    if select.group_by is not None:
        group_indices = _bind_group_by(select.group_by, schema)

    if select.columns is None:
        if group_indices is not None:
            raise QueryValidationError("'*' cannot be combined with GROUP BY")
        return _bind_plain(select, schema, where, indices=tuple(range(len(schema.columns))))

    has_aggregate = any(item.func is not None for item in select.columns)
    if group_indices is None and not has_aggregate:
        return _bind_plain(select, schema, where, indices=None)

    return _bind_aggregate(select, schema, where, group_indices)


def _bind_plain(
    select: _Select, schema: Schema, where: tuple | None, indices: tuple[int, ...] | None
) -> _Plan:
    if indices is None:
        seen: set[str] = set()
        resolved: list[int] = []
        for item in select.columns:
            name = item.column
            if name in seen:
                raise QueryValidationError(f"duplicate column in projection: {name!r}")
            seen.add(name)
            try:
                resolved.append(schema.index(name))
            except KeyError:
                raise QueryValidationError(f"unknown column: {name!r}") from None
        indices = tuple(resolved)

    source_order = None
    if select.order_by is not None:
        bound: list[tuple[int, bool, bool]] = []
        order_seen: set[str] = set()
        for item in select.order_by:
            if item.func is not None:
                raise QueryValidationError(
                    "ORDER BY aggregate is only valid for an aggregate query"
                )
            if item.column in order_seen:
                raise QueryValidationError(
                    f"duplicate column in ORDER BY: {item.column!r}"
                )
            order_seen.add(item.column)
            try:
                col_index = schema.index(item.column)
            except KeyError:
                raise QueryValidationError(
                    f"unknown ORDER BY column: {item.column!r}"
                ) from None
            nulls_first = item.nulls_first if item.nulls_first is not None else False
            bound.append((col_index, item.descending, nulls_first))
        source_order = tuple(bound)

    return _Plan(
        aggregate=False,
        where=where,
        limit=select.limit,
        indices=indices,
        source_order=source_order,
        grouped=False,
        group_indices=None,
        items=None,
        result_schema=None,
        result_order=None,
    )


def _bind_group_by(names: tuple[str, ...], schema: Schema) -> tuple[int, ...]:
    indices: list[int] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise QueryValidationError(f"duplicate column in GROUP BY: {name!r}")
        seen.add(name)
        try:
            indices.append(schema.index(name))
        except KeyError:
            raise QueryValidationError(f"unknown GROUP BY column: {name!r}") from None
    return tuple(indices)


def _bind_aggregate(
    select: _Select,
    schema: Schema,
    where: tuple | None,
    group_indices: tuple[int, ...] | None,
) -> _Plan:
    grouped = group_indices is not None
    group_set = set(group_indices) if grouped else set()

    items: list[_PlanItem] = []
    result_columns: list[ColumnSchema] = []
    result_names: set[str] = set()
    # Aggregates keyed by (func, star_arg, arg_index) for ORDER BY matching.
    agg_keys: dict[tuple[str, bool, int], int] = {}
    # Output positions of bare grouped columns, keyed by source index.
    group_positions: dict[int, int] = {}

    for item in select.columns:
        if item.alias:
            raise QueryValidationError("aliases are not supported")
        if item.func is None:
            name = item.column
            try:
                source_index = schema.index(name)
            except KeyError:
                raise QueryValidationError(f"unknown column: {name!r}") from None
            if not grouped:
                raise QueryValidationError(
                    f"column {name!r} must appear in GROUP BY or be wrapped in an aggregate"
                )
            if source_index not in group_set:
                raise QueryValidationError(
                    f"column {name!r} must appear in GROUP BY or be wrapped in an aggregate"
                )
            spec = None
            result_name = schema.columns[source_index].name
            col_schema = schema.columns[source_index]
            result_columns.append(
                ColumnSchema(col_schema.name, col_schema.type, col_schema.nullable)
            )
            group_positions[source_index] = len(items)
        else:
            spec = _bind_aggregate_item(item.func, item.arg, schema)
            source_index = -1 if spec.star_arg else spec.arg_index
            result_name = spec.result_name
            result_columns.append(
                ColumnSchema(spec.result_name, spec.result_type, spec.result_nullable)
            )
            agg_keys[(spec.func, spec.star_arg, spec.arg_index)] = len(items)
        if result_name in result_names:
            raise QueryValidationError(
                f"duplicate column in result: {result_name!r}"
            )
        result_names.add(result_name)
        items.append(_PlanItem("agg" if spec is not None else "group", source_index, spec))

    result_schema = Schema(result_columns)

    result_order = None
    if select.order_by is not None:
        bound_order: list[tuple[int, bool, bool]] = []
        order_seen: set[int] = set()
        for order_item in select.order_by:
            if order_item.func is None:
                try:
                    ref_index = schema.index(order_item.column)
                except KeyError:
                    raise QueryValidationError(
                        f"unknown ORDER BY column: {order_item.column!r}"
                    ) from None
                position = group_positions.get(ref_index)
                if position is None:
                    raise QueryValidationError(
                        f"ORDER BY column {order_item.column!r} is not among the selected results"
                    )
            else:
                spec = _bind_aggregate_item(order_item.func, order_item.arg, schema)
                position = agg_keys.get((spec.func, spec.star_arg, spec.arg_index))
                if position is None:
                    raise QueryValidationError(
                        f"ORDER BY aggregate {spec.result_name} is not among the selected results"
                    )
            if position in order_seen:
                raise QueryValidationError(
                    "duplicate column in ORDER BY"
                )
            order_seen.add(position)
            nulls_first = order_item.nulls_first if order_item.nulls_first is not None else False
            bound_order.append((position, order_item.descending, nulls_first))
        result_order = tuple(bound_order)

    return _Plan(
        aggregate=True,
        where=where,
        limit=select.limit,
        indices=None,
        source_order=None,
        grouped=grouped,
        group_indices=group_indices,
        items=tuple(items),
        result_schema=result_schema,
        result_order=result_order,
    )


def _bind_aggregate_item(func: str, arg: tuple, schema: Schema) -> _AggSpec:
    if arg[0] == "agg":
        raise QueryValidationError("aggregate calls cannot be nested")
    if arg[0] == "star":
        if func != "COUNT":
            raise QuerySyntaxError(f"only COUNT accepts '*', got {func}(*)")
        return _AggSpec(func, True, -1, "COUNT(*)", "int64", False)
    name = arg[1]
    try:
        index = schema.index(name)
    except KeyError:
        raise QueryValidationError(f"unknown column: {name!r}") from None
    col = schema.columns[index]
    result_name = f"{func}({col.name})"
    if func == "COUNT":
        return _AggSpec(func, False, index, result_name, "int64", False)
    if func in ("SUM", "AVG"):
        if col.type not in ("int64", "float64"):
            raise QueryValidationError(
                f"{func} requires an int64 or float64 argument, got {col.type} column {col.name!r}"
            )
        result_type = col.type if func == "SUM" else "float64"
        return _AggSpec(func, False, index, result_name, result_type, True)
    # MIN / MAX accept every type and keep it.
    return _AggSpec(func, False, index, result_name, col.type, True)


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
    if tag == "agg":
        raise QueryValidationError("aggregate functions are not allowed in WHERE")
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
        left = _bind_expr(node[2], schema)
        right = _bind_expr(node[3], schema)
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
    return node[2] if node[0] == "literal" else node[3]


def _require_boolean(node: tuple, context: str) -> None:
    if node[0] in ("cmp", "isnull", "not", "and", "or"):
        return
    type_name = node[2] if node[0] == "literal" else node[3]
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
# Public entry points
# ---------------------------------------------------------------------------


def query_file(path: Any, sql: str) -> Table:
    """Run ``sql`` against the single columnar file ``path``.

    Returns a :class:`~columnar_analytics.format.Table` with columns in
    projection order.  Rows are filtered by WHERE, then either grouped and
    aggregated or passed through; the surviving rows (or groups) are stably
    sorted when ORDER BY is given -- ties keep the file / first-row order --
    then capped by LIMIT, then projected.  The statement is parsed before
    the file is touched, so purely grammatical errors surface as
    :class:`QuerySyntaxError` regardless of whether ``path`` exists.
    Unknown columns, duplicate result or grouping columns, non-grouped
    plain columns, aggregates with illegal argument types, ORDER BY targets
    outside the projection, int64 SUM overflow and non-finite float
    aggregation raise :class:`QueryValidationError`; malformed files raise
    :class:`~columnar_analytics.format.ColumnarFormatError`; other I/O
    failures propagate as :class:`OSError`.
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


def _run_query(table: Table, select: _Select) -> Table:
    plan = _bind_select(select, table.schema)

    source_columns = table._columns
    row_count = table.row_count
    if plan.where is None:
        selected = list(range(row_count))
    else:
        selected = [
            i
            for i in range(row_count)
            if _eval(plan.where, tuple(col[i] for col in source_columns)) is True
        ]

    if plan.aggregate:
        return _run_aggregate(plan, source_columns, selected)

    if plan.source_order is not None:
        comparator = _make_comparator(source_columns, plan.source_order)
        selected = sorted(selected, key=cmp_to_key(comparator))

    if plan.limit is not None:
        selected = selected[: plan.limit]

    out_columns = [
        tuple(source_columns[index][i] for i in selected) for index in plan.indices
    ]
    out_schema = Schema([table.schema.columns[i] for i in plan.indices])
    return Table._from_storage(out_schema, out_columns)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _run_aggregate(
    plan: _Plan, source_columns: tuple[tuple, ...], selected: list[int]
) -> Table:
    if plan.grouped:
        # Keys are tuples of grouping values; a NULL key is just another
        # distinct key.  Dict insertion order preserves the first surviving
        # row of each group, which is the output order without ORDER BY.
        groups: dict[tuple, list[int]] = {}
        for row_index in selected:
            key = tuple(source_columns[gi][row_index] for gi in plan.group_indices)
            groups.setdefault(key, []).append(row_index)
        group_rows = list(groups.values())
    else:
        # A scalar aggregate always has exactly one group, even when empty.
        group_rows = [selected]

    rows = [
        tuple(_compute_item(item, rows, source_columns) for item in plan.items)
        for rows in group_rows
    ]

    if plan.result_order is not None:
        result_columns = [
            tuple(row[position] for row in rows)
            for position in range(len(plan.items))
        ]
        comparator = _make_comparator(result_columns, plan.result_order)
        order = sorted(range(len(rows)), key=cmp_to_key(comparator))
        rows = [rows[i] for i in order]

    if plan.limit is not None:
        rows = rows[: plan.limit]

    out_columns = [tuple(row[position] for row in rows) for position in range(len(plan.items))]
    return Table._from_storage(plan.result_schema, out_columns)


def _compute_item(
    item: _PlanItem, rows: list[int], source_columns: tuple[tuple, ...]
) -> Any:
    if item.kind == "group":
        # Every row in the group shares the grouping value; take the first.
        return source_columns[item.source_index][rows[0]]
    return _compute_aggregate(item.agg, rows, source_columns)


def _compute_aggregate(
    spec: _AggSpec, rows: list[int], source_columns: tuple[tuple, ...]
) -> Any:
    func = spec.func
    if spec.star_arg:
        return len(rows)
    values = source_columns[spec.arg_index]
    if func == "COUNT":
        return sum(1 for row_index in rows if values[row_index] is not None)
    present = [values[row_index] for row_index in rows if values[row_index] is not None]
    if not present:
        return None
    if func == "MIN":
        return min(present)
    if func == "MAX":
        return max(present)
    if func == "SUM" and spec.result_type == "int64":
        total = sum(present)
        if not (-(2**63) <= total <= 2**63 - 1):
            raise QueryValidationError("SUM of int64 values overflows int64")
        return total
    # SUM(float64) is a left-to-right float64 accumulation; AVG divides the
    # accumulated total (exact integer total for int64 arguments) by the
    # count.  Row order is fixed, so results are reproducible; a non-finite
    # total or quotient is rejected rather than emitted.
    total = sum(present)
    if func == "SUM":
        if not math.isfinite(total):
            raise QueryValidationError("SUM produced a non-finite float64 result")
        return total
    average = total / len(present)
    if not math.isfinite(average):
        raise QueryValidationError("AVG produced a non-finite float64 result")
    return average


def _make_comparator(
    columns: tuple[tuple, ...] | list[tuple],
    order_by: tuple[tuple[int, bool, bool], ...],
):
    def compare(a: int, b: int) -> int:
        for col_index, descending, nulls_first in order_by:
            va = columns[col_index][a]
            vb = columns[col_index][b]
            if va is None or vb is None:
                if va is None and vb is None:
                    continue
                # NULL placement follows NULLS FIRST / NULLS LAST alone; the
                # default (and the explicit LAST spelling) keeps NULLs at the
                # end for both ASC and DESC, so DESC must not flip this part.
                none_before = -1 if nulls_first else 1
                c = none_before if va is None else -none_before
            else:
                c = (va > vb) - (va < vb)
                if descending:
                    c = -c
            if c:
                return c
        return 0

    return compare
