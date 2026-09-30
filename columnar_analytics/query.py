"""Single-file SQL query layer.

Public API:

* :func:`query_file` -- run a ``SELECT ... FROM input [WHERE ...]`` query
  against one columnar file and return a :class:`~columnar_analytics.format.Table`
* :class:`QuerySyntaxError` -- lexical / grammatical errors
* :class:`QueryValidationError` -- unknown columns, wrong table name,
  type-incompatible comparisons

The accepted grammar (keywords case-insensitive)::

    query       := SELECT ('*' | ident (',' ident)*)
                   FROM ident [WHERE expr]
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

Operator precedence (highest first) is NOT, comparison, AND, OR; IS [NOT]
NULL is a postfix of its atom.  Comparison operands must be column
references or literals.  WHERE follows SQL three-valued logic: a normal
comparison against NULL yields UNKNOWN, UNKNOWN propagates through the
logical operators, and only TRUE rows are returned.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .format import Schema, Table, read_file

__all__ = [
    "QuerySyntaxError",
    "QueryValidationError",
    "query_file",
]

_KEYWORDS = frozenset(
    ("select", "from", "where", "not", "and", "or", "is", "null", "true", "false")
)


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
#   ("cmp", op, left_node, right_node)
#   ("isnull", operand, negate)
#   ("not", operand)
#   ("and"|"or", left, right)
# Predicate nodes are boolean-typed (three-valued at evaluation time).
# ---------------------------------------------------------------------------


@dataclass
class _Select:
    columns: tuple[str, ...] | None  # None means '*'
    table: str
    table_quoted: bool
    where: tuple | None


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
            columns: tuple[str, ...] | None = None
        else:
            columns = tuple(self._parse_projection())
        self._expect_keyword("from")
        table_tok = self._expect_table_name()
        where = None
        if self._accept_keyword("where"):
            where = self._parse_or()
        return _Select(
            columns=columns,
            table=table_tok.value,
            table_quoted=table_tok.kind == "qident",
            where=where,
        )

    def _parse_projection(self) -> list[str]:
        names = [self._expect_identifier_name()]
        while self._accept_op(","):
            names.append(self._expect_identifier_name())
        return names

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


def _bind_select(select: _Select, schema: Schema) -> tuple[tuple[int, ...], tuple | None]:
    table_matches = (
        select.table == "input"
        if select.table_quoted
        else select.table.lower() == "input"
    )
    if not table_matches:
        raise QueryValidationError(f"unknown table {select.table!r}; only 'input' is supported")
    if select.columns is None:
        indices = tuple(range(len(schema.columns)))
    else:
        seen: set[str] = set()
        indices_list: list[int] = []
        for name in select.columns:
            if name in seen:
                raise QueryValidationError(f"duplicate column in projection: {name!r}")
            seen.add(name)
            try:
                indices_list.append(schema.index(name))
            except KeyError:
                raise QueryValidationError(f"unknown column: {name!r}") from None
        indices = tuple(indices_list)
    where = _bind_expr(select.where, schema) if select.where is not None else None
    if where is not None and where[0] in ("literal", "column"):
        type_name = _operand_type(where)
        if type_name != "bool":
            raise QueryValidationError(
                f"WHERE clause must be boolean, got {type_name}"
            )
    return indices, where


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
    projection order and rows in the file's original order.  The statement is
    parsed before the file is touched, so purely grammatical errors surface as
    :class:`QuerySyntaxError` regardless of whether ``path`` exists.  Unknown
    columns, the wrong table name or type-incompatible predicates raise
    :class:`QueryValidationError`; malformed files raise
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
    indices, where = _bind_select(select, table.schema)

    source_columns = table._columns
    row_count = table.row_count
    if where is None:
        out_columns = [source_columns[index] for index in indices]
    else:
        selected = [
            i
            for i in range(row_count)
            if _eval(where, tuple(col[i] for col in source_columns)) is True
        ]
        out_columns = [
            tuple(source_columns[index][i] for i in selected) for index in indices
        ]

    out_schema = Schema([table.schema.columns[i] for i in indices])
    return Table._from_storage(out_schema, out_columns)
