"""columnar-analytics-engine — Columnar analytics engine with a SQL subset and vectorised execution"""

__version__ = "0.1.0"

from .format import (
    ColumnSchema,
    ColumnarFormatError,
    FORMAT_VERSION,
    Schema,
    Table,
    inspect_file,
    read_file,
    write_file,
)
from .query import (
    QuerySyntaxError,
    QueryValidationError,
    query_file,
    query_files,
)

__all__ = [
    "__version__",
    "FORMAT_VERSION",
    "ColumnarFormatError",
    "ColumnSchema",
    "Schema",
    "Table",
    "write_file",
    "read_file",
    "inspect_file",
    "QuerySyntaxError",
    "QueryValidationError",
    "query_file",
    "query_files",
]
