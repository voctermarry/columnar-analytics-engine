"""Command line entry point for columnar-analytics-engine."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from . import __version__
from .format import ColumnarFormatError, inspect_file
from .query import QuerySyntaxError, QueryValidationError, query_file, query_files

_QUERY_ERRORS = (QuerySyntaxError, QueryValidationError, ColumnarFormatError)


def _emit_table(table) -> None:
    payload = {
        "columns": [
            {"name": col.name, "type": col.type, "nullable": col.nullable}
            for col in table.schema.columns
        ],
        "rows": [
            [table._columns[c][r] for c in range(len(table.schema.columns))]
            for r in range(table.row_count)
        ],
    }
    out = sys.stdout
    try:
        out.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    json.dump(payload, out, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    out.write("\n")


def _parse_sources(raw: str) -> dict[str, pathlib.Path]:
    """Parse the ``query-files`` sources JSON into a name->:class:`Path` map.

    Malformed JSON or a payload that is not a non-empty object of
    name->path-string raise :class:`ValueError`; no path is ever touched.
    """
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid sources JSON: {exc}") from None
    if not isinstance(payload, dict) or not payload:
        raise ValueError("sources JSON must be a non-empty object mapping name to path")
    sources: dict[str, pathlib.Path] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError("sources keys must be non-empty table-name strings")
        if not isinstance(value, str) or not value:
            raise ValueError(f"sources path for {key!r} must be a non-empty string")
        sources[key] = pathlib.Path(value)
    return sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="columnar-analytics-engine", description="Columnar analytics engine with a SQL subset and vectorised execution")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="print the current version")
    inspect_parser = sub.add_parser("inspect", help="inspect a columnar file and print its metadata as JSON")
    inspect_parser.add_argument("path", help="path to the columnar file")
    query_parser = sub.add_parser("query", help="query a columnar file with SQL and print JSON")
    query_parser.add_argument("path", help="path to the columnar file")
    query_parser.add_argument("sql", help="SELECT statement to run against the file")
    query_files_parser = sub.add_parser(
        "query-files",
        help="query two columnar files with an equi-join SQL statement and print JSON",
    )
    query_files_parser.add_argument(
        "sources_json",
        help='JSON object mapping table name to file path, e.g. \'{"a":"a.caef","b":"b.caef"}\'',
    )
    query_files_parser.add_argument("sql", help="SELECT ... FROM a INNER|LEFT JOIN b ON a.k = b.k")
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "inspect":
        try:
            metadata = inspect_file(args.path)
        except ColumnarFormatError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        out = sys.stdout
        try:
            out.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
        json.dump(metadata, out, ensure_ascii=False, separators=(",", ":"))
        out.write("\n")
        return 0

    if args.command == "query":
        try:
            table = query_file(args.path, args.sql)
        except _QUERY_ERRORS as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _emit_table(table)
        return 0

    if args.command == "query-files":
        try:
            sources = _parse_sources(args.sources_json)
            table = query_files(sources, args.sql)
        except (ValueError, *_QUERY_ERRORS) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _emit_table(table)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
