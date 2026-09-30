"""Tests for the command line interface: version, help and inspect."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from columnar_analytics import __version__, write_table, WriteOptions
from columnar_analytics.cli import main

from .test_file import make_table


def run_cli(*argv):
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


def test_version_text_and_code():
    code, out, err = run_cli("version")
    assert code == 0
    assert out == f"{__version__}\n"
    assert err == ""


def test_no_command_prints_help_to_stdout():
    code, out, err = run_cli()
    assert code == 0
    assert "usage:" in out
    assert "inspect" in out
    assert err == ""


def test_inspect_outputs_json(tmp_path):
    path = tmp_path / "data.col"
    write_table(path, make_table(), WriteOptions(compression="zlib", dict_encoding=["label"]))
    code, out, err = run_cli("inspect", str(path))
    assert code == 0
    assert err == ""

    payload = json.loads(out)
    assert list(payload.keys()) == ["format_version", "row_count", "columns"]
    assert payload["format_version"] == 1
    assert payload["row_count"] == 5
    assert [column["name"] for column in payload["columns"]] == [
        "flag",
        "n",
        "ratio",
        "label",
    ]
    n = payload["columns"][1]
    assert list(n.keys()) == [
        "name",
        "type",
        "nullable",
        "compression",
        "encoding",
        "row_count",
        "null_count",
        "min",
        "max",
    ]
    assert n["min"] == -(2**63)
    assert n["max"] == 2**63 - 1


def test_inspect_missing_file_returns_1(tmp_path):
    code, out, err = run_cli("inspect", str(tmp_path / "missing.col"))
    assert code == 1
    assert out == ""
    assert err.strip() != ""


def test_inspect_bad_magic_returns_2(tmp_path):
    path = tmp_path / "bad.col"
    path.write_bytes(b"not a columnar file at all" * 3)
    code, out, err = run_cli("inspect", str(path))
    assert code == 2
    assert out == ""
    assert "magic" in err


def test_inspect_truncated_returns_2(tmp_path):
    path = tmp_path / "data.col"
    write_table(path, make_table())
    path.write_bytes(path.read_bytes()[:30])
    code, out, err = run_cli("inspect", str(path))
    assert code == 2
    assert out == ""
    assert err.strip() != ""


def test_inspect_output_is_utf8(tmp_path):
    path = tmp_path / "data.col"
    write_table(path, make_table(), WriteOptions(dict_encoding=["label"]))
    code, out, err = run_cli("inspect", str(path))
    assert code == 0
    # Re-encode the captured text as UTF-8; Chinese values must survive.
    encoded = out.encode("utf-8")
    payload = json.loads(encoded.decode("utf-8"))
    label = payload["columns"][3]
    assert label["max"] == "苹果"
