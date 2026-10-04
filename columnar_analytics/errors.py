"""Query-layer exception types.

These two classes are the error boundary shared by every query stage: the
parser (:mod:`columnar_analytics.parser`) raises :class:`QuerySyntaxError`
for lexical and grammatical problems before any file is touched, while
binding, planning and execution raise :class:`QueryValidationError` for a
syntactically valid statement that is incompatible with the schema.  The
module has no dependencies, so every stage imports it without creating
import cycles.
"""


class QuerySyntaxError(Exception):
    """Raised when a query is lexically or grammatically invalid."""


class QueryValidationError(Exception):
    """Raised when a syntactically valid query is incompatible with the schema."""
