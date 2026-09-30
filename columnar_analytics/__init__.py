"""columnar-analytics-engine — Columnar analytics engine with a SQL subset and vectorised execution"""

from .file import (
    ColumnarFormatError,
    WriteOptions,
    inspect_file,
    read_table,
    write_table,
)
from .table import (
    BOOL,
    FLOAT64,
    INT64,
    UTF8,
    Field,
    Schema,
    Table,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Field",
    "Schema",
    "Table",
    "WriteOptions",
    "write_table",
    "read_table",
    "inspect_file",
    "ColumnarFormatError",
    "BOOL",
    "INT64",
    "FLOAT64",
    "UTF8",
]
