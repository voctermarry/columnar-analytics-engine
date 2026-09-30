"""Single-file SQL querying for columnar files.

Public API:

* :func:`query_file` -- run a restricted ``SELECT ... FROM input [WHERE ...]``
* :class:`QuerySyntaxError` -- lexical / grammatical errors
* :class:`QueryValidationError` -- semantic errors (unknown columns, bad table,
  incompatible operand types)

The SQL subset:

* Projection is ``*`` or a comma-separated list of bare/quoted column names;
  aliases are not allowed and a column may appear only once.
* The only table name is ``input``.
* ``WHERE`` allows parentheses, ``NOT``/``AND``/``OR``, the comparisons
  ``=`` ``!=`` ``<`` ``<=`` ``>`` ``>=`` and ``IS [NOT] NULL``.
* Operands are column references and literals: booleans (``TRUE``/``FALSE``),
  int64 integers, finite float64 numbers and single-quoted UTF-8 strings.
* Operator precedence (loosest to tightest): ``OR``, ``AND``, comparison,
  ``NOT``.
* Keywords are case-insensitive; identifiers match schema names exactly
  (including Unicode) unless wrapped in double quotes.

``WHERE`` follows SQL three-valued logic: comparisons against NULL yield
UNKNOWN, NULLs propagate through ``AND``/``OR``/``NOT`` and only TRUE rows are
emitted. The result columns follow the projection order; rows keep the file's
original order.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from .format import Schema, Table, read_file

__all__ = [
    "QuerySyntaxError",
    "QueryValidationError",
    "query_file",
]


class QuerySyntaxError(Exception):
    """Raised for lexical errors, incomplete input or unsupported syntax."""


class QueryValidationError(Exception):
    """Raised for unknown columns, a wrong table name or type mismatches."""


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------


_KEYWORDS = frozenset(
    ("select", "from", "where", "not", "and", "or", "is", "null", "true", "false", "input")
)


@dataclass(frozen=True)
class _Token:
    kind: str  # 'ident' | 'keyword' | 'op' | 'int' | 'float' | 'string' | 'end'
    value: Any
    position: int


def _is_ident_start(ch: str) -> bool:
    return ch == "_" or ch.isalpha()


def _is_ident_part(ch: str) -> bool:
    return ch == "_" or ch.isalnum()


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
        start = i

        if ch == '"':
            i += 1
            chars: list[str] = []
            closed = False
            while i < n:
                c = sql[i]
                if c == '"':
                    if i + 1 < n and sql[i + 1] == '"':
                        chars.append('"')
                        i += 2
                        continue
                    closed = True
                    i += 1
                    break
                chars.append(c)
                i += 1
            if not closed:
                raise QuerySyntaxError(f"unterminated quoted identifier at position {start}")
            if not chars:
                raise QuerySyntaxError(f"empty quoted identifier at position {start}")
            tokens.append(_Token("ident", "".join(chars), start))
            continue

        if ch == "'":
            i += 1
            chars = []
            closed = False
            while i < n:
                c = sql[i]
                if c == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        chars.append("'")
                        i += 2
                        continue
                    closed = True
                    i += 1
                    break
                if c == "\\":
                    # Backslash has no special meaning in SQL string literals.
                    raise QuerySyntaxError(
                        f"invalid backslash escape in string literal at position {i}"
                    )
                chars.append(c)
                i += 1
            if not closed:
                raise QuerySyntaxError(f"unterminated string literal at position {start}")
            tokens.append(_Token("string", "".join(chars), start))
            continue

        if _is_ident_start(ch):
            i += 1
            while i < n and _is_ident_part(sql[i]):
                i += 1
            word = sql[start:i]
            lowered = word.lower()
            if lowered in _KEYWORDS:
                tokens.append(_Token("keyword", lowered, start))
            else:
                tokens.append(_Token("ident", word, start))
            continue

        if ch.isdigit() or (ch == "." and i + 1 < n and sql[i + 1].isdigit()):
            token, i = _scan_number(sql, i)
            tokens.append(token)
            continue

        if ch in "=<>!":
            if i + 1 < n and sql[i + 1] == "=":
                op = sql[i : i + 2]
                if op not in ("!=", "<=", ">="):
                    raise QuerySyntaxError(f"unexpected operator {op!r} at position {start}")
                tokens.append(_Token("op", op, start))
                i += 2
            elif ch == "!":
                raise QuerySyntaxError(f"unexpected character '!' at position {start}")
            else:
                tokens.append(_Token("op", ch, start))
                i += 1
            continue

        if ch in "(),*.+-":
            tokens.append(_Token("op", ch, start))
            i += 1
            continue

        raise QuerySyntaxError(f"unexpected character {ch!r} at position {start}")

    tokens.append(_Token("end", None, n))
    return tokens


def _scan_number(sql: str, start: int) -> tuple[_Token, int]:
    n = len(sql)
    i = start
    is_float = False
    while i < n and sql[i].isdigit():
        i += 1
    if i < n and sql[i] == ".":
        is_float = True
        i += 1
        while i < n and sql[i].isdigit():
            i += 1
    if i < n and sql[i] in "eE":
        is_float = True
        i += 1
        if i < n and sql[i] in "+-":
            i += 1
        if not (i < n and sql[i].isdigit()):
            raise QuerySyntaxError(f"malformed numeric literal at position {start}")
        while i < n and sql[i].isdigit():
            i += 1
    text = sql[start:i]
    if is_float:
        try:
            value = float(text)
        except ValueError:
            raise QuerySyntaxError(f"malformed numeric literal at position {start}") from None
        if not math.isfinite(value):
            raise QuerySyntaxError(f"float literal must be finite at position {start}")
        return _Token("float", value, start), i
    try:
        value = int(text)
    except ValueError:
        raise QuerySyntaxError(f"malformed integer literal at position {start}") from None
    # The sign is parsed as part of the operand, so a literal may reach 2**63.
    if value > 2**63:
        raise QuerySyntaxError(f"integer literal outside int64 range at position {start}")
    return _Token("int", value, start), i


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ColumnRef:
    name: str


@dataclass(frozen=True)
class _Literal:
    value: Any  # bool | int | float | str
    type: str


@dataclass(frozen=True)
class _IsNull:
    operand: _ColumnRef
    negated: bool


@dataclass(frozen=True)
class _Comparison:
    left: _ColumnRef | _Literal
    op: str
    right: _ColumnRef | _Literal


@dataclass(frozen=True)
class _Not:
    operand: Any


@dataclass(frozen=True)
class _Logical:
    op: str  # 'and' | 'or'
    left: Any
    right: Any


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[_Token]):
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> _Token:
        return self._tokens[self._pos]

    def _advance(self) -> _Token:
        token = self._tokens[self._pos]
        if token.kind != "end":
            self._pos += 1
        return token

    def _accept_kw(self, word: str) -> bool:
        token = self._peek()
        if token.kind == "keyword" and token.value == word:
            self._advance()
            return True
        return False

    def _expect_kw(self, word: str) -> _Token:
        token = self._peek()
        if token.kind != "keyword" or token.value != word:
            if token.kind == "end":
                raise QuerySyntaxError(f"expected {word.upper()} but reached end of input")
            raise QuerySyntaxError(
                f"expected {word.upper()} at position {token.position}"
            )
        return self._advance()

    def parse(self) -> tuple[tuple[str, ...] | None, Any]:
        self._expect_kw("select")
        projection = self._parse_projection()
        self._expect_kw("from")
        table = self._peek()
        if table.kind == "end":
            raise QuerySyntaxError("expected a table name after FROM but reached end of input")
        if table.kind != "keyword" or table.value != "input":
            raise QueryValidationError("only the table name 'input' is supported")
        self._advance()
        where = None
        if self._accept_kw("where"):
            where = self._parse_or()
        if self._peek().kind != "end":
            token = self._peek()
            raise QuerySyntaxError(
                f"unexpected trailing input at position {token.position}"
            )
        return projection, where

    def _parse_projection(self) -> tuple[str, ...] | None:
        token = self._peek()
        if token.kind == "op" and token.value == "*":
            self._advance()
            return None
        names: list[str] = []
        seen: set[str] = set()
        while True:
            name = self._parse_column_name()
            if name in seen:
                raise QueryValidationError(f"duplicate column in projection: {name!r}")
            seen.add(name)
            names.append(name)
            comma = self._peek()
            if comma.kind == "op" and comma.value == ",":
                self._advance()
                continue
            break
        return tuple(names)

    def _parse_column_name(self) -> str:
        token = self._peek()
        if token.kind == "ident":
            self._advance()
            return token.value
        if token.kind == "keyword" and token.value == "input":
            # Bare word after SELECT is a column name, not the table keyword.
            self._advance()
            return "input"
        if token.kind == "end":
            raise QuerySyntaxError("expected a column name but reached end of input")
        raise QuerySyntaxError(f"expected a column name at position {token.position}")

    # WHERE grammar ---------------------------------------------------------

    def _parse_or(self) -> Any:
        node = self._parse_and()
        while self._accept_kw("or"):
            node = _Logical("or", node, self._parse_and())
        return node

    def _parse_and(self) -> Any:
        node = self._parse_not()
        while self._accept_kw("and"):
            node = _Logical("and", node, self._parse_not())
        return node

    def _parse_not(self) -> Any:
        if self._accept_kw("not"):
            return _Not(self._parse_not())
        return self._parse_predicate()

    def _parse_predicate(self) -> Any:
        token = self._peek()
        if token.kind == "op" and token.value == "(":
            self._advance()
            node = self._parse_or()
            close = self._peek()
            if not (close.kind == "op" and close.value == ")"):
                if close.kind == "end":
                    raise QuerySyntaxError("missing closing parenthesis")
                raise QuerySyntaxError(f"expected ')' at position {close.position}")
            self._advance()
            return node

        left = self._parse_operand()
        if self._accept_kw("is"):
            negated = self._accept_kw("not")
            null_token = self._peek()
            if null_token.kind != "keyword" or null_token.value != "null":
                if null_token.kind == "end":
                    raise QuerySyntaxError("expected NULL after IS")
                raise QuerySyntaxError(f"expected NULL after IS at position {null_token.position}")
            self._advance()
            if not isinstance(left, _ColumnRef):
                raise QuerySyntaxError(
                    "IS NULL / IS NOT NULL requires a column reference"
                )
            return _IsNull(left, negated)

        op_token = self._peek()
        if op_token.kind != "op" or op_token.value not in ("=", "!=", "<", "<=", ">", ">="):
            if op_token.kind == "end":
                raise QuerySyntaxError("incomplete WHERE expression")
            raise QuerySyntaxError(
                f"expected a comparison operator at position {op_token.position}"
            )
        self._advance()
        right = self._parse_operand()
        return _Comparison(left, op_token.value, right)

    def _parse_operand(self) -> _ColumnRef | _Literal:
        token = self._peek()
        if token.kind == "ident":
            self._advance()
            return _ColumnRef(token.value)
        if token.kind == "keyword" and token.value == "input":
            self._advance()
            return _ColumnRef("input")
        if token.kind == "keyword" and token.value in ("true", "false"):
            self._advance()
            return _Literal(token.value == "true", "bool")
        if token.kind == "keyword" and token.value == "null":
            raise QuerySyntaxError(
                f"NULL is not a supported operand at position {token.position}; use IS NULL"
            )
        sign = 1
        if token.kind == "op" and token.value in ("+", "-"):
            if token.value == "-":
                sign = -1
            self._advance()
            token = self._peek()
        if token.kind == "int":
            self._advance()
            value = sign * token.value
            if not (-(2**63) <= value <= 2**63 - 1):
                raise QuerySyntaxError(
                    f"integer literal outside int64 range at position {token.position}"
                )
            return _Literal(value, "int64")
        if token.kind == "float":
            self._advance()
            value = sign * token.value
            return _Literal(value, "float64")
        if token.kind == "string":
            self._advance()
            if sign == -1:
                raise QuerySyntaxError(
                    f"unexpected sign before string literal at position {token.position}"
                )
            return _Literal(token.value, "utf8")
        if token.kind == "end":
            raise QuerySyntaxError("incomplete WHERE expression")
        raise QuerySyntaxError(
            f"expected a column reference or literal at position {token.position}"
        )


# ---------------------------------------------------------------------------
# Semantic analysis
# ---------------------------------------------------------------------------


def _resolve(node: Any, schema: Schema) -> Any:
    """Annotate every column reference with its index and declared type."""
    if isinstance(node, _ColumnRef):
        try:
            index = schema.index(node.name)
        except KeyError:
            raise QueryValidationError(f"unknown column: {node.name!r}") from None
        col = schema.columns[index]
        return _BoundColumn(index, col.type, col.nullable, col.name)
    if isinstance(node, _IsNull):
        return _IsNull(_resolve(node.operand, schema), node.negated)
    if isinstance(node, _Comparison):
        return _Comparison(_resolve(node.left, schema), node.op, _resolve(node.right, schema))
    if isinstance(node, _Not):
        return _Not(_resolve(node.operand, schema))
    if isinstance(node, _Logical):
        return _Logical(node.op, _resolve(node.left, schema), _resolve(node.right, schema))
    return node  # _Literal unchanged


@dataclass(frozen=True)
class _BoundColumn:
    index: int
    type: str
    nullable: bool
    name: str


def _operand_type(node: Any) -> str:
    if isinstance(node, _BoundColumn):
        return node.type
    assert isinstance(node, _Literal)
    return node.type


def _validate(node: Any) -> None:
    if isinstance(node, (_BoundColumn, _Literal)):
        return
    if isinstance(node, _IsNull):
        _validate(node.operand)
        return
    if isinstance(node, _Comparison):
        _validate(node.left)
        _validate(node.right)
        left_type = _operand_type(node.left)
        right_type = _operand_type(node.right)
        compatible = {
            ("bool", "bool"),
            ("int64", "int64"),
            ("int64", "float64"),
            ("float64", "int64"),
            ("float64", "float64"),
            ("utf8", "utf8"),
        }
        if (left_type, right_type) not in compatible:
            raise QueryValidationError(
                f"cannot compare {left_type} with {right_type}"
            )
        if (left_type == "bool" or right_type == "bool") and node.op not in ("=", "!="):
            raise QueryValidationError("boolean values only support = and !=")
        return
    if isinstance(node, _Not):
        _validate(node.operand)
        return
    if isinstance(node, _Logical):
        _validate(node.left)
        _validate(node.right)
        return


# ---------------------------------------------------------------------------
# Evaluation (three-valued logic: None stands for UNKNOWN)
# ---------------------------------------------------------------------------


def _eval_operand(node: Any, row: tuple) -> Any:
    if isinstance(node, _BoundColumn):
        return row[node.index]
    return node.value


def _eval_predicate(node: Any, row: tuple) -> bool | None:
    if isinstance(node, _IsNull):
        value = _eval_operand(node.operand, row)
        is_null = value is None
        return (not is_null) if node.negated else is_null

    if isinstance(node, _Comparison):
        left = _eval_operand(node.left, row)
        right = _eval_operand(node.right, row)
        if left is None or right is None:
            return None
        op = node.op
        if op == "=":
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        return left >= right

    if isinstance(node, _Not):
        value = _eval_predicate(node.operand, row)
        return None if value is None else (not value)

    if isinstance(node, _Logical):
        left = _eval_predicate(node.left, row)
        right = _eval_predicate(node.right, row)
        if node.op == "and":
            if left is False or right is False:
                return False
            if left is None or right is None:
                return None
            return left and right
        # or
        if left is True or right is True:
            return True
        if left is None or right is None:
            return None
        return left or right

    raise AssertionError(f"unexpected expression node: {node!r}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def query_file(path: str | os.PathLike, sql: str) -> Table:
    """Run ``sql`` against the table stored in ``path`` and return a :class:`Table`.

    Raises :class:`QuerySyntaxError` for lexical/grammatical problems,
    :class:`QueryValidationError` for unknown columns, the wrong table name or
    incompatible types, and lets :class:`~columnar_analytics.ColumnarFormatError`
    and :class:`OSError` propagate from the file layer.
    """
    tokens = _tokenize(sql)
    projection, where = _Parser(tokens).parse()

    table = read_file(path)
    schema = table.schema

    if projection is not None:
        for name in projection:
            if name not in schema.names:
                raise QueryValidationError(f"unknown column: {name!r}")
        indices = [schema.index(name) for name in projection]
        result_schema = Schema([schema.columns[i] for i in indices])
    else:
        indices = list(range(len(schema.columns)))
        result_schema = schema

    bound_where = _resolve(where, schema) if where is not None else None
    if bound_where is not None:
        _validate(bound_where)

    source_columns = table._columns
    row_count = table.row_count
    if bound_where is None:
        kept = range(row_count)
    else:
        kept = [
            i
            for i in range(row_count)
            if _eval_predicate(bound_where, tuple(col[i] for col in source_columns))
        ]

    if row_count and indices:
        out_columns = [
            [source_columns[col_index][i] for i in kept] for col_index in indices
        ]
    else:
        out_columns = [[] for _ in indices]
    return Table._from_storage(result_schema, out_columns)
