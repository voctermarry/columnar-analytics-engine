"""In-memory table model: ordered schema, typed columns and projection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

# Supported logical column types.
BOOL = "bool"
INT64 = "int64"
FLOAT64 = "float64"
UTF8 = "utf8"

TYPES = frozenset({BOOL, INT64, FLOAT64, UTF8})


class ColumnarValidationError(ValueError):
    """Raised when a schema or column data violates the public contract."""


@dataclass(frozen=True)
class Field:
    """A named, typed and optionally nullable column."""

    name: str
    type: str
    nullable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ColumnarValidationError("column name must be a string")
        if not self.name:
            raise ColumnarValidationError("column name must be non-empty")
        if self.type not in TYPES:
            raise ColumnarValidationError(f"unsupported column type: {self.type!r}")
        if not isinstance(self.nullable, bool):
            raise ColumnarValidationError("nullable must be a bool")


class Schema:
    """An ordered, unique collection of :class:`Field` objects."""

    def __init__(self, fields: Iterable[Field | tuple]):
        fields = list(fields)
        if not fields:
            raise ColumnarValidationError("schema must contain at least one column")
        normalised: list[Field] = []
        seen: set[str] = set()
        for index, field in enumerate(fields):
            if not isinstance(field, Field):
                raise ColumnarValidationError(
                    f"schema entry {index} must be a Field"
                )
            if field.name in seen:
                raise ColumnarValidationError(
                    f"duplicate column name: {field.name!r}"
                )
            seen.add(field.name)
            normalised.append(field)
        self._fields = normalised

    @property
    def fields(self) -> list[Field]:
        return list(self._fields)

    @property
    def names(self) -> list[str]:
        return [field.name for field in self._fields]

    def field(self, name: str) -> Field:
        for candidate in self._fields:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def index(self, name: str) -> int:
        for i, candidate in enumerate(self._fields):
            if candidate.name == name:
                return i
        raise KeyError(name)

    def __len__(self) -> int:
        return len(self._fields)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Schema):
            return NotImplemented
        return self._fields == other._fields

    def __repr__(self) -> str:
        return f"Schema({self._fields!r})"


def _validate_scalar(value: Any, field: Field) -> None:
    if value is None:
        if not field.nullable:
            raise ColumnarValidationError(
                f"column {field.name!r} is not nullable but contains None"
            )
        return
    if field.type == BOOL:
        if not isinstance(value, bool):
            raise ColumnarValidationError(
                f"column {field.name!r} expects bool, got {type(value).__name__}"
            )
    elif field.type == INT64:
        # bool is a subclass of int; reject it explicitly.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ColumnarValidationError(
                f"column {field.name!r} expects int64, got {type(value).__name__}"
            )
        if not (-(2**63) <= value <= 2**63 - 1):
            raise ColumnarValidationError(
                f"column {field.name!r} value out of int64 range: {value}"
            )
    elif field.type == FLOAT64:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ColumnarValidationError(
                f"column {field.name!r} expects float64, got {type(value).__name__}"
            )
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            raise ColumnarValidationError(
                f"column {field.name!r} contains NaN or infinite value"
            )
    elif field.type == UTF8:
        if not isinstance(value, str):
            raise ColumnarValidationError(
                f"column {field.name!r} expects utf8 string, got {type(value).__name__}"
            )


class Table:
    """An ordered set of equally long typed columns."""

    def __init__(self, schema: Schema, columns: dict[str, list]):
        if not isinstance(schema, Schema):
            raise ColumnarValidationError("schema must be a Schema instance")
        if not isinstance(columns, dict):
            raise ColumnarValidationError("columns must be a dict of name -> list")

        names = schema.names
        if set(columns.keys()) != set(names):
            missing = [name for name in names if name not in columns]
            extra = [name for name in columns if name not in set(names)]
            if missing:
                raise ColumnarValidationError(
                    f"missing columns: {', '.join(map(repr, missing))}"
                )
            raise ColumnarValidationError(
                f"unexpected columns: {', '.join(map(repr, extra))}"
            )

        row_count = len(columns[names[0]])
        for name in names:
            values = columns[name]
            if not isinstance(values, list):
                raise ColumnarValidationError(
                    f"column {name!r} must be a list, got {type(values).__name__}"
                )
            if len(values) != row_count:
                raise ColumnarValidationError(
                    f"column {name!r} has {len(values)} rows, expected {row_count}"
                )
            field = schema.field(name)
            for value in values:
                _validate_scalar(value, field)

        self._schema = schema
        # Keep columns in schema order internally.
        self._columns = {name: list(columns[name]) for name in names}
        self._row_count = row_count

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def row_count(self) -> int:
        return self._row_count

    def column(self, name: str) -> list:
        if name not in self._columns:
            raise KeyError(name)
        return list(self._columns[name])

    def columns(self) -> dict[str, list]:
        return {name: list(values) for name, values in self._columns.items()}

    def project(self, names: Iterable[str]) -> "Table":
        """Return a new table containing the given columns in the given order."""
        names = list(names)
        if len(names) != len(set(names)):
            raise ColumnarValidationError("projection contains duplicate columns")
        for name in names:
            if name not in self._columns:
                raise KeyError(name)
        projected_schema = Schema([self._schema.field(name) for name in names])
        projected_columns = {name: self.column(name) for name in names}
        return Table(projected_schema, projected_columns)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Table):
            return NotImplemented
        return (
            self._schema == other._schema
            and self._row_count == other._row_count
            and self._columns == other._columns
        )

    def __repr__(self) -> str:
        return f"Table(schema={self._schema!r}, row_count={self._row_count})"
