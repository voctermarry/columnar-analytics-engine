"""Command line entry point for columnar-analytics-engine."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .file import ColumnarFormatError, inspect_file


def _inspect(path: str) -> int:
    try:
        result = inspect_file(path)
    except ColumnarFormatError as exc:
        print(f"columnar-analytics-engine: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"columnar-analytics-engine: {exc}", file=sys.stderr)
        return 1

    payload = {
        "format_version": result["format_version"],
        "row_count": result["row_count"],
        "columns": [
            {
                "name": column["name"],
                "type": column["type"],
                "nullable": column["nullable"],
                "compression": column["compression"],
                "encoding": column["encoding"],
                "row_count": column["row_count"],
                "null_count": column["null_count"],
                "min": column["min"],
                "max": column["max"],
            }
            for column in result["columns"]
        ],
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="columnar-analytics-engine", description="Columnar analytics engine with a SQL subset and vectorised execution")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="print the current version")

    inspect_parser = sub.add_parser(
        "inspect", help="print file metadata as UTF-8 JSON"
    )
    inspect_parser.add_argument("path", help="path to a columnar file")

    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "inspect":
        return _inspect(args.path)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
