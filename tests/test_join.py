"""Tests for the two-file equi-join layer and the ``query-files`` CLI command."""

from __future__ import annotations

import io
import contextlib
import json
import os
import pathlib
import subprocess
import sys

import pytest

from columnar_analytics import (
    ColumnSchema,
    ColumnarFormatError,
    QuerySyntaxError,
    QueryValidationError,
    Schema,
    Table,
    query_file,
    query_files,
    write_file,
)
from columnar_analytics.cli import main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


EMP_SCHEMA = Schema(
    [
        ColumnSchema("eid", "int64"),
        ColumnSchema("dept", "utf8", nullable=True),
        ColumnSchema("sal", "int64", nullable=True),
    ]
)
DEPT_SCHEMA = Schema(
    [
        ColumnSchema("dname", "utf8"),
        ColumnSchema("loc", "utf8"),
    ]
)

EMP = {
    # dept: eng, eng, hr, None(no match), sales(no dept row)
    "eid": [1, 2, 3, 4, 5],
    "dept": ["eng", "eng", "hr", None, "sales"],
    "sal": [10, 20, 30, None, 50],
}
DEPT = {
    "dname": ["eng", "hr", "eng"],
    "loc": ["NYC", "SFO", "LON"],
}


@pytest.fixture()
def sources(tmp_path):
    emp = tmp_path / "emp.caef"
    dept = tmp_path / "dept.caef"
    write_file(emp, Table(EMP_SCHEMA, EMP))
    write_file(dept, Table(DEPT_SCHEMA, DEPT))
    return {"emp": emp, "dept": dept}


def _rows(result):
    return [[c[i] for c in result._columns] for i in range(result.row_count)]


ON = "emp.dept = dept.dname"


# ---------------------------------------------------------------------------
# INNER JOIN
# ---------------------------------------------------------------------------


def test_inner_star_schema_order_and_nullability(sources):
    result = query_files(sources, f"select * from emp inner join dept on {ON}")
    assert result.column_names == (
        "emp.eid",
        "emp.dept",
        "emp.sal",
        "dept.dname",
        "dept.loc",
    )
    assert result.schema.columns == (
        ColumnSchema("emp.eid", "int64"),
        ColumnSchema("emp.dept", "utf8", nullable=True),
        ColumnSchema("emp.sal", "int64", nullable=True),
        ColumnSchema("dept.dname", "utf8"),
        ColumnSchema("dept.loc", "utf8"),
    )


def test_bare_join_means_inner(sources):
    a = query_files(sources, f"select * from emp join dept on {ON}")
    b = query_files(sources, f"select * from emp inner join dept on {ON}")
    assert _rows(a) == _rows(b)


def test_inner_emits_each_matching_pair_in_left_then_right_order(sources):
    # eng rows (eid 1,2) each match dept rows 0 (NYC) and 2 (LON) in that
    # right-file order; hr (eid 3) matches SFO.  Non-matches 4,5 are dropped.
    result = query_files(sources, f"select emp.eid, dept.loc from emp inner join dept on {ON}")
    assert _rows(result) == [
        [1, "NYC"],
        [1, "LON"],
        [2, "NYC"],
        [2, "LON"],
        [3, "SFO"],
    ]


def test_inner_null_keys_never_match(sources):
    result = query_files(
        sources, f"select emp.eid from emp inner join dept on {ON} where emp.eid = 4"
    )
    assert result.row_count == 0


def test_explicit_projection_keeps_qualified_names(sources):
    result = query_files(
        sources, f"select dept.loc, emp.eid from emp inner join dept on {ON}"
    )
    assert result.column_names == ("dept.loc", "emp.eid")
    assert _rows(result) == [
        ["NYC", 1],
        ["LON", 1],
        ["NYC", 2],
        ["LON", 2],
        ["SFO", 3],
    ]


def test_join_keywords_case_insensitive(sources):
    result = query_files(
        sources, f"select emp.eid from emP InNeR jOiN dePt On emp.dept = dept.dname"
    )
    assert result.column("emp.eid") == [1, 1, 2, 2, 3]


def test_quoted_identifiers_exact_match(tmp_path):
    a = tmp_path / "A.caef"
    b = tmp_path / "B.caef"
    write_file(a, Table(Schema([ColumnSchema("K", "int64")]), {"K": [1]}))
    write_file(b, Table(Schema([ColumnSchema("k", "int64")]), {"k": [1]}))
    result = query_files(
        {"A": a, "B": b}, 'select A."K", B."k" from A inner join B on A."K" = B."k"'
    )
    assert result.column_names == ("A.K", "B.k")
    assert _rows(result) == [[1, 1]]
    # The quoted table name is case-sensitive.
    with pytest.raises(QueryValidationError):
        query_files({"A": a, "B": b}, 'select * from "a" join B on "a"."K" = B.k')


def test_int64_float64_cross_type_keys(tmp_path):
    left = tmp_path / "l.caef"
    right = tmp_path / "r.caef"
    write_file(
        left,
        Table(
            Schema([ColumnSchema("k", "int64"), ColumnSchema("v", "utf8")]),
            {"k": [1, 2, 3], "v": ["a", "b", "c"]},
        ),
    )
    write_file(
        right,
        Table(
            Schema([ColumnSchema("k", "float64"), ColumnSchema("w", "int64")]),
            {"k": [1.0, 3.0], "w": [10, 30]},
        ),
    )
    result = query_files(
        {"l": left, "r": right},
        "select l.v, r.w from l inner join r on l.k = r.k order by l.v",
    )
    assert _rows(result) == [["a", 10], ["c", 30]]


# ---------------------------------------------------------------------------
# LEFT JOIN
# ---------------------------------------------------------------------------


def test_left_join_pads_unmatched_rows(sources):
    result = query_files(
        sources,
        f"select emp.eid, dept.loc, dept.dname from emp left join dept on {ON} order by emp.eid",
    )
    assert _rows(result) == [
        [1, "NYC", "eng"],
        [1, "LON", "eng"],
        [2, "NYC", "eng"],
        [2, "LON", "eng"],
        [3, "SFO", "hr"],
        [4, None, None],
        [5, None, None],
    ]


def test_left_join_unmatched_padding_order_without_order_by(sources):
    # Unmatched left rows keep their position in the left-file expansion:
    # eid 4 and 5 appear after all matched rows, in left order.
    result = query_files(
        sources, f"select emp.eid, dept.loc from emp left join dept on {ON}"
    )
    assert result.column("emp.eid") == [1, 1, 2, 2, 3, 4, 5]
    assert result.column("dept.loc") == [
        "NYC",
        "LON",
        "NYC",
        "LON",
        "SFO",
        None,
        None,
    ]


def test_left_join_marks_right_columns_nullable(sources):
    result = query_files(sources, f"select * from emp left join dept on {ON}")
    nullable = {c.name: c.nullable for c in result.schema.columns}
    assert nullable == {
        "emp.eid": False,
        "emp.dept": True,
        "emp.sal": True,
        "dept.dname": True,  # forced nullable by LEFT padding
        "dept.loc": True,
    }
    # An explicit right projection is likewise nullable; left keeps its flag.
    result = query_files(
        sources, f"select dept.loc, emp.eid from emp left join dept on {ON}"
    )
    cols = {c.name: c.nullable for c in result.schema.columns}
    assert cols == {"dept.loc": True, "emp.eid": False}


def test_left_join_no_matches_pads_every_left_row(tmp_path):
    left = tmp_path / "l.caef"
    right = tmp_path / "r.caef"
    write_file(left, Table(Schema([ColumnSchema("k", "int64")]), {"k": [1, 2]}))
    write_file(right, Table(Schema([ColumnSchema("k", "int64"), ColumnSchema("w", "utf8")]), {"k": [9], "w": ["z"]}))
    result = query_files(
        {"l": left, "r": right}, "select * from l left join r on l.k = r.k"
    )
    assert _rows(result) == [[1, None, None], [2, None, None]]
    inner = query_files(
        {"l": left, "r": right}, "select * from l inner join r on l.k = r.k"
    )
    assert inner.row_count == 0
    assert inner.column_names == ("l.k", "r.k", "r.w")


# ---------------------------------------------------------------------------
# WHERE / GROUP BY / aggregates / ORDER BY / LIMIT over joins
# ---------------------------------------------------------------------------


def test_where_runs_after_join(sources):
    result = query_files(
        sources,
        f"select emp.eid, dept.loc from emp left join dept on {ON} "
        "where dept.loc = 'LON'",
    )
    assert _rows(result) == [[1, "LON"], [2, "LON"]]


def test_where_can_discard_left_padded_rows(sources):
    result = query_files(
        sources,
        f"select emp.eid from emp left join dept on {ON} where dept.loc is not null",
    )
    assert result.column("emp.eid") == [1, 1, 2, 2, 3]


def test_group_by_and_aggregates_over_join(sources):
    result = query_files(
        sources,
        f"select dept.loc, count(*), sum(emp.sal), avg(emp.eid) "
        f"from emp inner join dept on {ON} group by dept.loc order by dept.loc",
    )
    assert result.column_names == ("dept.loc", "COUNT(*)", "SUM(emp.sal)", "AVG(emp.eid)")
    assert _rows(result) == [
        ["LON", 2, 30, 1.5],
        ["NYC", 2, 30, 1.5],
        ["SFO", 1, 30, 3.0],
    ]


def test_count_star_vs_count_column_under_left_join(sources):
    result = query_files(
        sources,
        "select count(*), count(dept.loc), count(emp.sal) "
        f"from emp left join dept on {ON}",
    )
    # 7 join rows; the two padded rows have no right loc.  emp.sal is a left
    # column, so padding never nulls it: only eid 4's genuine NULL is skipped.
    assert _rows(result) == [[7, 5, 6]]


def test_order_by_and_limit_after_join(sources):
    result = query_files(
        sources,
        f"select emp.eid, dept.loc from emp inner join dept on {ON} "
        "order by emp.eid desc, dept.loc limit 3",
    )
    # eid 3 (SFO) first; for eid 2 the secondary key dept.loc is ascending, so
    # LON precedes NYC.
    assert _rows(result) == [[3, "SFO"], [2, "LON"], [2, "NYC"]]


def test_order_by_need_not_be_projected(sources):
    result = query_files(
        sources, f"select emp.eid from emp inner join dept on {ON} order by dept.loc desc"
    )
    # SFO first (eid 3), then NYC/LON ties in right-file (NYC then LON).
    assert result.column("emp.eid") == [3, 1, 2, 1, 2]


def test_plain_join_query_without_order_by_is_stable(sources):
    runs = {
        tuple(
            query_files(
                sources,
                f"select emp.eid, dept.loc from emp inner join dept on {ON} limit 10",
            ).column("dept.loc")
        )
        for _ in range(3)
    }
    assert runs == {("NYC", "LON", "NYC", "LON", "SFO")}


def test_aggregate_result_names_include_qualified_argument(sources):
    result = query_files(
        sources, f"select min(emp.sal), max(emp.sal) from emp inner join dept on {ON}"
    )
    assert result.column_names == ("MIN(emp.sal)", "MAX(emp.sal)")
    assert _rows(result) == [[10, 30]]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_unknown_left_and_right_table(sources):
    with pytest.raises(QueryValidationError):
        query_files(sources, f"select * from nope inner join dept on nope.x = dept.dname")
    with pytest.raises(QueryValidationError):
        query_files(sources, f"select * from emp inner join nope on emp.dept = nope.x")


def test_unknown_qualified_column(sources):
    with pytest.raises(QueryValidationError):
        query_files(sources, f"select emp.nope from emp inner join dept on {ON}")
    with pytest.raises(QueryValidationError):
        query_files(
            sources, f"select emp.eid from emp inner join dept on {ON} where dept.nope = 1"
        )
    with pytest.raises(QueryValidationError):
        query_files(
            sources, f"select count(*) from emp inner join dept on {ON} group by emp.nope"
        )


def test_unqualified_references_rejected(sources):
    base = "from emp inner join dept on emp.dept = dept.dname"
    with pytest.raises(QueryValidationError):
        query_files(sources, f"select eid {base}")
    with pytest.raises(QueryValidationError):
        query_files(sources, f"select emp.eid {base} where eid > 1")
    with pytest.raises(QueryValidationError):
        query_files(sources, f"select count(*) {base} group by loc")
    with pytest.raises(QueryValidationError):
        query_files(sources, f"select count(sal) {base}")
    with pytest.raises(QueryValidationError):
        query_files(sources, f"select emp.eid {base} order by sal")


def test_count_star_is_the_only_unqualified_reference(sources):
    result = query_files(
        sources, f"select count(*) from emp inner join dept on {ON}"
    )
    assert result.column("COUNT(*)") == [5]


def test_same_table_on_both_sides_rejected(sources):
    with pytest.raises(QueryValidationError):
        query_files(sources, "select * from emp inner join emp on emp.eid = emp.sal")


def test_join_key_must_reference_the_correct_side(sources):
    with pytest.raises(QueryValidationError):
        query_files(sources, "select * from emp inner join dept on dept.dname = dept.loc")
    with pytest.raises(QueryValidationError):
        query_files(sources, "select * from emp inner join dept on emp.dept = emp.sal")


def test_incompatible_join_key_types(sources, tmp_path):
    # int64 vs utf8 cannot be compared as join keys.
    with pytest.raises(QueryValidationError):
        query_files(sources, "select * from emp inner join dept on emp.eid = dept.loc")

    # utf8 (left) vs bool (right) is likewise rejected, with both keys sourced
    # from their correct side.
    lft = tmp_path / "l.caef"
    rgt = tmp_path / "r.caef"
    write_file(lft, Table(Schema([ColumnSchema("k", "utf8")]), {"k": ["a"]}))
    write_file(rgt, Table(Schema([ColumnSchema("k", "bool"), ColumnSchema("w", "int64")]), {"k": [True], "w": [1]}))
    with pytest.raises(QueryValidationError):
        query_files(
            {"l": lft, "r": rgt}, "select * from l inner join r on l.k = r.k"
        )


def test_duplicate_result_column(sources):
    with pytest.raises(QueryValidationError):
        query_files(sources, f"select emp.eid, emp.eid from emp inner join dept on {ON}")
    with pytest.raises(QueryValidationError):
        query_files(
            sources,
            f"select emp.sal, emp.sal from emp inner join dept on {ON} group by emp.sal",
        )


def test_query_files_requires_a_join(sources):
    with pytest.raises(QueryValidationError):
        query_files(sources, "select * from emp")


def test_query_file_rejects_join(tmp_path):
    # The single-file entry keeps its historical pre-read syntax-error
    # classification for anything join-shaped.
    missing = tmp_path / "missing.caef"
    with pytest.raises(QuerySyntaxError):
        query_file(missing, "select * from input join dept on input.x = dept.y")


# ---------------------------------------------------------------------------
# Syntax errors (all decided before files are read)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "select * from a join b",  # missing ON
        "select * from a inner b on a.x = b.y",  # missing JOIN keyword
        "select * from a left b on a.x = b.y",
        "select * from a inner join b",  # missing ON clause
        "select * from a join b where a.x = b.y",  # ON replaced by WHERE
        "select * from a join b on a.x b.y",  # missing '='
        "select * from a join b on a.x = b.y and a.z = b.w",  # compound ON
        "select * from a join b on a.x = b.y join c on a.x = c.q",  # second join
        "select * from a right join b on a.x = b.y",  # unsupported join type
        "select * from a full join b on a.x = b.y",
        "select * from a cross join b",
        "select a.x from a join b on a.x = b.y alias",  # alias not supported
        "select * from a join b on a.x = ",  # dangling ON
        "select * from a join on a.x = b.y",  # missing right table
    ],
)
def test_join_syntax_errors_before_files(sql):
    missing = {"a": pathlib.Path("/no/such/a.caef"), "b": pathlib.Path("/no/such/b.caef")}
    with pytest.raises(QuerySyntaxError):
        query_files(missing, sql)


def test_lexical_error_before_files():
    missing = {"a": pathlib.Path("/no/such/a"), "b": pathlib.Path("/no/such/b")}
    with pytest.raises(QuerySyntaxError):
        query_files(missing, "select * from a join b on a.x = @")


# ---------------------------------------------------------------------------
# sources validation (ValueError, never touches the filesystem)
# ---------------------------------------------------------------------------


class _UnreadPath(os.PathLike):
    def __fspath__(self):  # pragma: no cover - must never be called
        raise AssertionError("filesystem must not be accessed during validation")


def test_sources_must_be_nonempty_mapping():
    with pytest.raises(ValueError):
        query_files([("a", _UnreadPath())], "select * from a join b on a.x = b.y")
    with pytest.raises(ValueError):
        query_files({}, "select * from a join b on a.x = b.y")


def test_sources_keys_must_be_nonempty_strings():
    with pytest.raises(ValueError):
        query_files({1: _UnreadPath()}, "select * from a join b on a.x = b.y")
    with pytest.raises(ValueError):
        query_files({"": _UnreadPath()}, "select * from a join b on a.x = b.y")


def test_sources_values_must_be_path_objects():
    with pytest.raises(ValueError):
        query_files({"a": "/tmp/x"}, "select * from a join b on a.x = b.y")
    with pytest.raises(ValueError):
        query_files({"a": 1}, "select * from a join b on a.x = b.y")
    with pytest.raises(ValueError):
        query_files({"a": None}, "select * from a join b on a.x = b.y")


# ---------------------------------------------------------------------------
# File-level error propagation
# ---------------------------------------------------------------------------


def test_corrupt_join_file_is_format_error(sources, tmp_path):
    bad = tmp_path / "bad.caef"
    bad.write_bytes(b"not a columnar file")
    with pytest.raises(ColumnarFormatError):
        query_files({"emp": sources["emp"], "dept": bad}, f"select * from emp join dept on {ON}")


def test_missing_join_file_is_os_error(sources, tmp_path):
    with pytest.raises(OSError):
        query_files(
            {"emp": sources["emp"], "dept": tmp_path / "gone.caef"},
            f"select * from emp join dept on {ON}",
        )


def test_syntax_and_validation_errors_precede_file_read():
    missing = {"emp": pathlib.Path("/no/emp"), "dept": pathlib.Path("/no/dept")}
    with pytest.raises(QuerySyntaxError):
        query_files(missing, "select * from emp join dept")
    with pytest.raises(QueryValidationError):
        query_files(missing, "select * from nowhere join dept on nowhere.x = dept.y")


# ---------------------------------------------------------------------------
# Backward compatibility of the single-file grammar
# ---------------------------------------------------------------------------


def test_columns_named_like_join_words_still_work(tmp_path):
    schema = Schema(
        [ColumnSchema("left", "int64"), ColumnSchema("on", "int64", nullable=True)]
    )
    p = tmp_path / "w.caef"
    write_file(p, Table(schema, {"left": [1, 2], "on": [3, 4]}))
    assert query_file(p, "select left, on from input").column_names == ("left", "on")
    assert query_file(p, "select left from input where on is not null").column("left") == [1, 2]


def test_stray_dot_is_syntax_error_before_read(tmp_path):
    with pytest.raises(QuerySyntaxError):
        query_file(tmp_path / "missing.caef", "select input.id from input")


# ---------------------------------------------------------------------------
# CLI: query-files
# ---------------------------------------------------------------------------


def _sources_json(sources):
    return json.dumps({name: str(path) for name, path in sources.items()})


def test_cli_query_files_inner(capsys, sources):
    code = main(
        [
            "query-files",
            _sources_json(sources),
            f"select emp.eid, dept.loc from emp inner join dept on {ON} limit 2",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert list(payload) == ["columns", "rows"]
    assert payload["columns"] == [
        {"name": "emp.eid", "type": "int64", "nullable": False},
        {"name": "dept.loc", "type": "utf8", "nullable": False},
    ]
    assert payload["rows"] == [[1, "NYC"], [1, "LON"]]
    assert out.count("\n") == 1


def test_cli_query_files_left_nullable(capsys, sources):
    code = main(
        [
            "query-files",
            _sources_json(sources),
            f"select dept.dname from emp left join dept on {ON} order by emp.eid limit 1",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["columns"] == [
        {"name": "dept.dname", "type": "utf8", "nullable": True}
    ]


def test_cli_query_files_byte_deterministic(sources):
    outs = set()
    sql = f"select * from emp left join dept on {ON} order by emp.eid, dept.loc nulls first"
    for _ in range(3):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert main(["query-files", _sources_json(sources), sql]) == 0
        outs.add(buf.getvalue())
    assert len(outs) == 1


def test_cli_query_files_bad_sources_json_exit_2(capsys):
    code = main(["query-files", "{not json", "select * from a join b on a.x = b.y"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_query_files_syntax_error_exit_2(capsys, sources):
    code = main(["query-files", _sources_json(sources), "select * from emp join dept"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_query_files_validation_error_exit_2(capsys, sources):
    code = main(
        [
            "query-files",
            _sources_json(sources),
            "select eid from emp join dept on emp.dept = dept.dname",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""


def test_cli_query_files_format_error_exit_2(capsys, sources, tmp_path):
    bad = tmp_path / "bad.caef"
    bad.write_bytes(b"garbage")
    src = {"emp": str(sources["emp"]), "dept": str(bad)}
    code = main(["query-files", json.dumps(src), f"select * from emp join dept on {ON}"])
    assert code == 2
    assert capsys.readouterr().out == ""


def test_cli_query_files_os_error_exit_1(capsys, tmp_path):
    src = {"a": str(tmp_path / "nope1"), "b": str(tmp_path / "nope2")}
    code = main(
        ["query-files", json.dumps(src), "select * from a join b on a.x = b.y"]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.strip()


def test_cli_existing_commands_still_work(capsys):
    assert main(["version"]) == 0


def test_cli_query_files_subprocess(sources):
    env = dict(
        os.environ,
        PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "columnar_analytics.cli",
            "query-files",
            _sources_json(sources),
            f"select emp.eid from emp join dept on {ON} where dept.loc = 'SFO'",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["rows"] == [[3]]
