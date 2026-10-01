"""Command line entry point for columnar-analytics-engine."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .format import ColumnarFormatError, inspect_file
from .query import (
    QuerySyntaxError,
    QueryValidationError,
    explain_file,
    explain_files,
    query_file,
    query_files,
)

_QUERY_ERRORS = (QuerySyntaxError, QueryValidationError, ColumnarFormatError)


def _write_json(payload) -> int:
    out = sys.stdout
    try:
        out.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    json.dump(payload, out, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    out.write("\n")
    return 0


def _print_query_result(table) -> int:
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
    return _write_json(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="columnar-analytics-engine", description="Columnar analytics engine with a SQL subset and vectorised execution")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="print the current version")
    inspect_parser = sub.add_parser("inspect", help="inspect a columnar file and print its metadata as JSON")
    inspect_parser.add_argument("path", help="path to the columnar file")
    query_parser = sub.add_parser("query", help="query a columnar file with SQL and print JSON")
    query_parser.add_argument("path", help="path to the columnar file")
    query_parser.add_argument("sql", help="SELECT statement to run against the file")
    query_files_parser = sub.add_parser("query-files", help="query two columnar files with a join and print JSON")
    query_files_parser.add_argument("sources", help="JSON object mapping table names to columnar file paths")
    query_files_parser.add_argument("sql", help="SELECT statement to run against the mapped tables")
    explain_parser = sub.add_parser("explain", help="explain a SQL statement against a columnar file and print the plan JSON")
    explain_parser.add_argument("path", help="path to the columnar file")
    explain_parser.add_argument("sql", help="SELECT statement to plan against the file")
    explain_files_parser = sub.add_parser("explain-files", help="explain a SQL statement against mapped tables and print the plan JSON")
    explain_files_parser.add_argument("sources", help="JSON object mapping table names to columnar file paths")
    explain_files_parser.add_argument("sql", help="SELECT statement to plan against the mapped tables")
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
        return _print_query_result(table)

    if args.command == "query-files":
        try:
            sources = json.loads(args.sources)
        except json.JSONDecodeError as exc:
            print(f"invalid sources JSON: {exc}", file=sys.stderr)
            return 2
        try:
            table = query_files(sources, args.sql)
        except _QUERY_ERRORS as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return _print_query_result(table)

    if args.command == "explain":
        try:
            plan = explain_file(args.path, args.sql)
        except _QUERY_ERRORS as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return _write_json(plan)

    if args.command == "explain-files":
        try:
            sources = json.loads(args.sources)
        except json.JSONDecodeError as exc:
            print(f"invalid sources JSON: {exc}", file=sys.stderr)
            return 2
        try:
            plan = explain_files(sources, args.sql)
        except _QUERY_ERRORS as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return _write_json(plan)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
