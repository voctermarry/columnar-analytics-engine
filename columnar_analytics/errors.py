"""Query-layer exception types.

These live in a leaf module so every stage of the query pipeline --
parsing, binding, planning, execution and the public orchestration --
raises the same public exception types without importing one another.
"""

from __future__ import annotations

__all__ = [
    "QuerySyntaxError",
    "QueryValidationError",
]


class QuerySyntaxError(Exception):
    """Raised when a query is lexically or grammatically invalid."""


class QueryValidationError(Exception):
    """Raised when a syntactically valid query is incompatible with the schema."""
