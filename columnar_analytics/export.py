"""Export query results to CSV or JSONL files.

Public API:

* :func:`export_query_file` -- run a single-file query and write the result
  to a target file
* :func:`export_query_files` -- run a statement against a table-name to path
  mapping (optionally with one equi-join) and write the result

Both entry points reuse the existing SQL subset, binding rules, join
semantics, NULL and type semantics and the stable result ordering of
:mod:`columnar_analytics.query`; no query syntax is added.  ``format`` is
``"csv"`` (the default) or ``"jsonl"``; anything else raises
:class:`ValueError`.

CSV is UTF-8 without BOM and uses LF line endings.  The first line always
holds the column names in result-schema order.  A field containing a comma,
a double quote, CR or LF is enclosed in double quotes as a whole and inner
double quotes are doubled.  NULL is written as an empty unquoted field, the
empty string as a quoted empty field, booleans as ``true`` / ``false`` and
numbers in the same text form the compact JSON result output uses.

JSONL writes each result row as one compact JSON object with keys in result
column order; values keep their existing JSON types and null, non-ASCII
characters are not escaped and every line ends with a single LF.  A zero-row
result yields a header-only CSV file or a zero-byte JSONL file.  Repeating an
export with the same inputs, SQL, format and version produces byte-identical
output; without ORDER BY the deterministic order of the underlying query is
kept.

The target file is replaced (atomically) only after the query has fully
succeeded and the whole content is encoded; any failure neither creates the
target nor alters an existing one.  A target that resolves to the same path
as any actually referenced source file raises :class:`ValueError`.  SQL,
binding and columnar-format problems keep raising
:class:`~columnar_analytics.query.QuerySyntaxError`,
:class:`~columnar_analytics.query.QueryValidationError` and
:class:`~columnar_analytics.format.ColumnarFormatError`; invalid ``sources``
raise :class:`ValueError`; directory, permission and other filesystem
problems remain :class:`OSError`.  On success the number of written result
rows is returned.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .format import Table, _atomic_write
from .query import (
    _Parser,
    _resolve_table_ref,
    _tokenize,
    _validate_sources,
    query_file,
    query_files,
)

__all__ = ["export_query_file", "export_query_files"]

_FORMATS = ("csv", "jsonl")


def export_query_file(path: Any, sql: str, target: Any, format: str = "csv") -> int:
    """Run ``sql`` against ``path`` and write the result to ``target``.

    Returns the number of result rows written.  See the module docstring for
    the exact output formats, the atomic-replace guarantee and the exception
    contract.
    """
    _check_format(format)
    _check_target(target, (path,))
    table = query_file(path, sql)
    _write_result(table, target, format)
    return table.row_count


def export_query_files(sources: Any, sql: str, target: Any, format: str = "csv") -> int:
    """Run ``sql`` against the tables named by ``sources`` and write the result.

    Only the tables referenced by the statement are read, and only those
    referenced paths take part in the target/source collision check.  Returns
    the number of result rows written.
    """
    _check_format(format)
    referenced = _referenced_paths(sources, sql)
    _check_target(target, referenced)
    table = query_files(sources, sql)
    _write_result(table, target, format)
    return table.row_count


def _check_format(format: Any) -> None:
    if format not in _FORMATS:
        raise ValueError(f"unknown export format: {format!r}")


def _referenced_paths(sources: Any, sql: str) -> list:
    """The paths of the tables the statement actually references.

    Mirrors the validation and resolution prefix of ``query_files`` exactly,
    so sources, syntax and table-resolution errors surface with the same
    types and order as the query itself would raise them.
    """
    paths = _validate_sources(sources)
    select = _Parser(_tokenize(sql), allow_join=True).parse()
    left_key = _resolve_table_ref(select.table, select.table_quoted, tuple(paths))
    referenced = [paths[left_key]]
    join = select.join
    if join is not None:
        right_key = _resolve_table_ref(join.table, join.table_quoted, tuple(paths))
        if right_key != left_key:
            referenced.append(paths[right_key])
    return referenced


def _check_target(target: Any, sources) -> None:
    target_real = os.path.realpath(target)
    for source in sources:
        if os.path.realpath(source) == target_real:
            raise ValueError(
                f"export target resolves to a referenced source file: {source!r}"
            )


def _write_result(table: Table, target: Any, format: str) -> None:
    if format == "csv":
        text = _render_csv(table)
    else:
        text = _render_jsonl(table)
    # The whole content is encoded before the target is touched, so an
    # encoding failure can never create or alter it.
    blob = text.encode("utf-8")
    _atomic_write(target, blob)


def _render_csv(table: Table) -> str:
    names = [col.name for col in table.schema.columns]
    columns = table._columns
    lines = [",".join(_csv_field(name) for name in names)]
    for r in range(table.row_count):
        lines.append(",".join(_csv_field(col[r]) for col in columns))
    return "\n".join(lines) + "\n"


def _csv_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        # The same text form the compact JSON result output produces.
        return json.dumps(value, allow_nan=False)
    if value == "":
        return '""'
    if any(ch in value for ch in ',"\r\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


def _render_jsonl(table: Table) -> str:
    names = [col.name for col in table.schema.columns]
    columns = table._columns
    lines = []
    for r in range(table.row_count):
        row = {name: col[r] for name, col in zip(names, columns)}
        lines.append(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        )
    return "".join(line + "\n" for line in lines)
