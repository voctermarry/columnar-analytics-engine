"""Export SQL query results as reviewable CSV / JSONL files.

Public API:

* :func:`export_query_file` -- query one columnar file and write the result
* :func:`export_query_files` -- query a table-name to path mapping and write
* :class:`ValueError` -- an unknown format or a destination that aliases a
  referenced source file

The exports reuse the existing single-/multi-file query layer unchanged:
the accepted SQL subset, binding rules, join order, NULL and type
semantics and deterministic ordering are all :mod:`columnar_analytics.query`
behaviour.  ``export_query_files`` additionally forwards an optional
``join_strategy`` (``"hash"`` / ``"sort_merge"``) to the multi-file query
layer; an explicit strategy applies to every join step and both
strategies produce byte-identical exports.  No query syntax is added
here.

Both formats are UTF-8 without a BOM and use LF line endings.  CSV always
writes a header line with the result column names in schema order and quotes
RFC-4180-style only when needed (commas, quotes, CR or LF); NULL becomes an
empty unquoted field while an empty string becomes a quoted empty field,
bools render as ``true`` / ``false`` and numbers reuse the compact JSON
result text.  JSONL writes one compact JSON object per result row with keys
in result column order, unescaped non-ASCII text and one trailing LF; a
zero-row result is a zero-byte file.

The destination is replaced atomically and only after the whole query has
succeeded and the full output has been encoded, so any failure leaves an
existing target untouched and never creates a new one.  Parsing problems
raise :class:`~columnar_analytics.query.QuerySyntaxError`, binding problems
:class:`~columnar_analytics.query.QueryValidationError`, malformed sources
:class:`~columnar_analytics.format.ColumnarFormatError`, and directory,
permission or other filesystem problems :class:`OSError`.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from .format import Table
from .query import (
    _referenced_source_paths,
    _validate_join_strategy,
    query_file,
    query_files,
)

__all__ = [
    "export_query_file",
    "export_query_files",
]

_EXPORT_FORMATS = ("csv", "jsonl")


def export_query_file(
    path: Any, sql: str, destination: Any, format: str = "csv"
) -> int:
    """Run ``sql`` against one columnar file and export the result.

    Returns the number of result rows written.  See the module docstring for
    the CSV / JSONL byte layout and the atomic-write guarantee.

    :class:`ValueError` is raised for an unknown ``format`` or a
    ``destination`` resolving to the same path as the queried source;
    query and file errors keep their usual exception types.
    """
    format = _validate_format(format)
    _ensure_distinct_path(destination, (path,))
    table = query_file(path, sql)
    return _write_export(table, destination, format)


def export_query_files(
    sources: Any,
    sql: str,
    destination: Any,
    format: str = "csv",
    join_strategy: Any = None,
) -> int:
    """Run ``sql`` against the mapped tables and export the result.

    Returns the number of result rows written.  Only the tables actually
    referenced by the statement participate in the destination-overlap
    check; ``sources`` validation otherwise follows
    :func:`columnar_analytics.query.query_files`.

    ``join_strategy`` forwards the optional ``"hash"`` / ``"sort_merge"``
    join algorithm to :func:`columnar_analytics.query.query_files`; both
    strategies export byte-identical files.  Invalid ``sources`` or
    ``join_strategy``, an unknown ``format`` or a ``destination``
    resolving to the same path as any referenced source raises
    :class:`ValueError`; strategy and source validation happen before any
    source file is opened.
    """
    format = _validate_format(format)
    strategy = _validate_join_strategy(join_strategy)
    referenced = _referenced_source_paths(sources, sql, strategy)
    _ensure_distinct_path(destination, referenced)
    table = query_files(sources, sql, strategy)
    return _write_export(table, destination, format)


def _validate_format(format: Any) -> str:
    if not isinstance(format, str) or format not in _EXPORT_FORMATS:
        known = ", ".join(_EXPORT_FORMATS)
        raise ValueError(f"unknown export format: {format!r}; expected one of {known}")
    return format


def _ensure_distinct_path(destination: Any, referenced: Any) -> None:
    if not isinstance(destination, (str, os.PathLike)):
        raise ValueError("export destination must be a file path")
    target = os.path.realpath(os.fspath(destination))
    for source in referenced:
        if not isinstance(source, (str, os.PathLike)):
            # A non-path source can never alias the destination path.
            continue
        if os.path.realpath(os.fspath(source)) == target:
            raise ValueError(
                "export destination must not be a referenced source file: "
                f"{os.fspath(source)!r}"
            )


def _write_export(table: Table, destination: Any, format: str) -> int:
    if format == "csv":
        blob = _render_csv(table).encode("utf-8")
    else:
        blob = _render_jsonl(table).encode("utf-8")
    _atomic_replace(destination, blob)
    return table.row_count


def _render_csv(table: Table) -> str:
    lines = [_csv_row(_csv_text(name) for name in table.schema.names)]
    columns = table._columns
    for r in range(table.row_count):
        lines.append(
            _csv_row(
                _csv_scalar(columns[c][r]) for c in range(len(table.schema.columns))
            )
        )
    # Every line, including the last one and a header-only file, ends in LF.
    return "".join(line + "\n" for line in lines)


def _csv_row(fields) -> str:
    return ",".join(fields)


def _csv_text(text: str) -> str:
    # RFC-4180 quoting for a piece of UTF-8 text (headers and string
    # values alike): wrap when it carries a comma, quote, CR or LF and
    # double every quote inside.
    if any(ch in text for ch in (",", '"', "\r", "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


def _csv_scalar(value: Any) -> str:
    if value is None:
        # NULL is an empty unquoted field.
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        # An empty string is quoted so it stays distinct from NULL.
        return '""' if value == "" else _csv_text(value)
    # Numbers reuse the compact JSON result text.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _render_jsonl(table: Table) -> str:
    columns = table._columns
    lines = []
    for r in range(table.row_count):
        row = {
            table.schema.columns[c].name: columns[c][r]
            for c in range(len(table.schema.columns))
        }
        lines.append(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        )
    return "".join(line + "\n" for line in lines)


def _atomic_replace(destination: Any, blob: bytes) -> None:
    path = os.fspath(destination)
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_name = tempfile.mkstemp(prefix=".columnar-export-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
