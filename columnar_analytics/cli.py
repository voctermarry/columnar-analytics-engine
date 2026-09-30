"""Command line entry point for columnar-analytics-engine."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="columnar-analytics-engine", description="Columnar analytics engine with a SQL subset and vectorised execution")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="print the current version")
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
