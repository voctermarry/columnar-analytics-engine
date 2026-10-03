"""Tests for the two-file join query layer and the ``query-files`` CLI command."""

from __future__ import annotations

import json

import pytest

from columnar_analytics import (
    ColumnSchema,
    ColumnarFormatError,
    QuerySyntaxError,
    QueryValidationError,
    Schema,
    Table,
    explain_files,
    query_file,
    query_files,
    write_file,
)
from columnar_analytics.cli import main


LEFT_SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("name", "utf8", nullable=True),
    ]
)
LEFT_DATA = {
    "id": [1, 2, 3, 4],
    "k": [10, 20, 10, None],
    "name": ["a", "b", "c", "d"],
}

RIGHT_SCHEMA = Schema(
    [
        ColumnSchema("rid", "int64"),
        ColumnSchema("k", "int64", nullable=True),
        ColumnSchema("tag", "utf8", nullable=True),
    ]
)
RIGHT_DATA = {
    "rid": [100, 200, 300],
    "k": [10, 10, 30],
    "tag": ["x", "y", "z"],
}


@pytest.fixture()
def paths(tmp_path):
    left = tmp_path / "left.caef"
    right = tmp_path / "right.caef"
    write_file(left, Table(LEFT_SCHEMA, LEFT_DATA))
    write_file(right, Table(RIGHT_SCHEMA, RIGHT_DATA))
    return {"l": left, "r": right}


def joined_rows(result):
    return [
        [result._columns[c][i] for c in range(len(result.schema.columns))]
        for i in range(result.row_count)
    ]


# ---------------------------------------------------------------------------
# INNER JOIN
# ---------------------------------------------------------------------------


def test_inner_join_star(paths):
    result = query_files(paths, "SELECT * FROM l INNER JOIN r ON l.k = r.k")
    assert result.column_names == ("l.id", "l.k", "l.name", "r.rid", "r.k", "r.tag")
    # Left row order, right row order within one left row; NULL k never matches.
    assert joined_rows(result) == [
        [1, 10, "a", 100, 10, "x"],
        [1, 10, "a", 200, 10, "y"],
        [3, 10, "c", 100, 10, "x"],
        [3, 10, "c", 200, 10, "y"],
    ]
    # INNER JOIN keeps the right-side nullability from the file schema.
    assert [c.nullable for c in result.schema.columns] == [False, True, True, False, True, True]


def test_inner_join_projection_qualified_names(paths):
    result = query_files(paths, "SELECT r.tag, l.id FROM l INNER JOIN r ON l.k = r.k")
    assert result.column_names == ("r.tag", "l.id")
    assert result.column("l.id") == [1, 1, 3, 3]
    assert result.column("r.tag") == ["x", "y", "x", "y"]


def test_inner_join_where_order_limit(paths):
    result = query_files(
        paths,
        "SELECT l.id, r.rid FROM l INNER JOIN r ON l.k = r.k "
        "WHERE r.tag != 'x' ORDER BY l.id DESC LIMIT 1",
    )
    assert joined_rows(result) == [[3, 200]]


def test_join_aggregates(paths):
    result = query_files(
        paths,
        "SELECT l.name, COUNT(*), SUM(r.rid) FROM l INNER JOIN r ON l.k = r.k "
        "GROUP BY l.name ORDER BY l.name",
    )
    assert result.column_names == ("l.name", "COUNT(*)", "SUM(r.rid)")
    assert joined_rows(result) == [["a", 2, 300], ["c", 2, 300]]


def test_join_group_by_order_by_agg(paths):
    result = query_files(
        paths,
        "SELECT l.name, COUNT(*) FROM l INNER JOIN r ON l.k = r.k "
        "GROUP BY l.name ORDER BY COUNT(*) DESC",
    )
    assert result.column("COUNT(*)") == [2, 2]


def test_left_join_unmatched_rows(paths):
    result = query_files(paths, "SELECT * FROM l LEFT JOIN r ON l.k = r.k")
    assert joined_rows(result) == [
        [1, 10, "a", 100, 10, "x"],
        [1, 10, "a", 200, 10, "y"],
        [2, 20, "b", None, None, None],
        [3, 10, "c", 100, 10, "x"],
        [3, 10, "c", 200, 10, "y"],
        [4, None, "d", None, None, None],
    ]
    # LEFT JOIN forces every right-side result column nullable.
    assert [c.nullable for c in result.schema.columns] == [False, True, True, True, True, True]


def test_left_join_explicit_projection_right_nullable(paths):
    result = query_files(paths, "SELECT l.id, r.tag FROM l LEFT JOIN r ON l.k = r.k")
    cols = {c.name: c for c in result.schema.columns}
    assert cols["l.id"].nullable is False
    assert cols["r.tag"].nullable is True


def test_right_join_star(paths):
    result = query_files(paths, "SELECT * FROM l RIGHT JOIN r ON l.k = r.k")
    assert result.column_names == ("l.id", "l.k", "l.name", "r.rid", "r.k", "r.tag")
    # Right-file order; within one right row the matches expand in left-file
    # order; the unmatched right row (rid 300) pads the left side with NULL.
    assert joined_rows(result) == [
        [1, 10, "a", 100, 10, "x"],
        [3, 10, "c", 100, 10, "x"],
        [1, 10, "a", 200, 10, "y"],
        [3, 10, "c", 200, 10, "y"],
        [None, None, None, 300, 30, "z"],
    ]
    # RIGHT JOIN forces every left-side result column nullable; the right
    # side keeps its file-declared nullability.
    assert [c.nullable for c in result.schema.columns] == [True, True, True, False, True, True]


def test_right_join_explicit_projection_left_nullable(paths):
    result = query_files(paths, "SELECT l.id, r.rid FROM l RIGHT JOIN r ON l.k = r.k")
    cols = {c.name: c for c in result.schema.columns}
    assert cols["l.id"].nullable is True
    assert cols["r.rid"].nullable is False
    assert joined_rows(result) == [
        [1, 100],
        [3, 100],
        [1, 200],
        [3, 200],
        [None, 300],
    ]


def test_full_outer_join_star(paths):
    result = query_files(paths, "SELECT * FROM l FULL OUTER JOIN r ON l.k = r.k")
    # Matches and unmatched left rows follow LEFT JOIN order; unmatched
    # right rows are then appended in right-file order.
    assert joined_rows(result) == [
        [1, 10, "a", 100, 10, "x"],
        [1, 10, "a", 200, 10, "y"],
        [2, 20, "b", None, None, None],
        [3, 10, "c", 100, 10, "x"],
        [3, 10, "c", 200, 10, "y"],
        [4, None, "d", None, None, None],
        [None, None, None, 300, 30, "z"],
    ]
    # FULL OUTER JOIN forces every result column on both sides nullable.
    assert [c.nullable for c in result.schema.columns] == [True, True, True, True, True, True]


def test_full_outer_join_keywords_case_insensitive(paths):
    result = query_files(paths, "select * from l full outer join r on l.k = r.k")
    assert result.row_count == 7


def test_right_join_where_order_limit(paths):
    result = query_files(
        paths,
        "SELECT l.id, r.rid FROM l RIGHT JOIN r ON l.k = r.k "
        "WHERE r.tag IS NOT NULL ORDER BY r.rid ASC, l.id ASC",
    )
    assert joined_rows(result) == [
        [1, 100],
        [3, 100],
        [1, 200],
        [3, 200],
        [None, 300],
    ]


def test_right_join_padded_null_participates_in_where(paths):
    # The padded NULL on the left follows three-valued logic in WHERE.
    result = query_files(
        paths,
        "SELECT r.rid FROM l RIGHT JOIN r ON l.k = r.k WHERE l.id IS NULL",
    )
    assert joined_rows(result) == [[300]]
    result = query_files(
        paths,
        "SELECT r.rid FROM l RIGHT JOIN r ON l.k = r.k WHERE l.id = 1",
    )
    assert joined_rows(result) == [[100], [200]]


def test_full_outer_join_aggregates(paths):
    result = query_files(
        paths,
        "SELECT l.name, COUNT(*), SUM(r.rid) FROM l FULL OUTER JOIN r ON l.k = r.k "
        "GROUP BY l.name ORDER BY l.name",
    )
    # The unmatched right row forms a NULL-keyed group; COUNT(*) counts it,
    # SUM skips no rid there since the right side is populated; unmatched
    # left rows (name b, d) form their own groups with SUM(r.rid) NULL.
    assert joined_rows(result) == [
        ["a", 2, 300],
        ["b", 1, None],
        ["c", 2, 300],
        ["d", 1, None],
        [None, 1, 300],
    ]


def test_full_outer_join_distinct_and_order(paths):
    result = query_files(
        paths,
        "SELECT DISTINCT l.k, r.k FROM l FULL OUTER JOIN r ON l.k = r.k "
        "ORDER BY l.k NULLS LAST, r.k NULLS LAST",
    )
    # Padded NULLs on either side compare equal in DISTINCT; rk NULLS LAST
    # keeps (None, None) behind (None, 30) inside the NULL l.k group.
    assert joined_rows(result) == [
        [10, 10],
        [20, None],
        [None, 30],
        [None, None],
    ]


def test_right_and_full_duplicate_keys_full_combination(tmp_path):
    left = tmp_path / "dl.caef"
    right = tmp_path / "dr.caef"
    write_file(left, Table(Schema([ColumnSchema("k", "int64")]), {"k": [1, 1, 2]}))
    write_file(right, Table(Schema([ColumnSchema("k", "int64")]), {"k": [1, 1, 3]}))
    sources = {"a": left, "b": right}
    right_rows = joined_rows(
        query_files(sources, "SELECT * FROM a RIGHT JOIN b ON a.k = b.k")
    )
    assert right_rows == [
        [1, 1], [1, 1], [1, 1], [1, 1],
        [None, 3],
    ]
    full_rows = joined_rows(
        query_files(sources, "SELECT * FROM a FULL OUTER JOIN b ON a.k = b.k")
    )
    assert full_rows == [
        [1, 1], [1, 1], [1, 1], [1, 1],
        [2, None],
        [None, 3],
    ]


def test_null_keys_never_match_right_and_full(tmp_path):
    left = tmp_path / "nl.caef"
    right = tmp_path / "nr.caef"
    write_file(left, Table(Schema([ColumnSchema("k", "int64", nullable=True)]), {"k": [None, 1]}))
    write_file(right, Table(Schema([ColumnSchema("k", "int64", nullable=True)]), {"k": [None, 2]}))
    sources = {"a": left, "b": right}
    assert joined_rows(query_files(sources, "SELECT * FROM a RIGHT JOIN b ON a.k = b.k")) == [
        [None, None],
        [None, 2],
    ]
    assert joined_rows(query_files(sources, "SELECT * FROM a FULL OUTER JOIN b ON a.k = b.k")) == [
        [None, None],
        [1, None],
        [None, None],
        [None, 2],
    ]


def test_join_keywords_case_insensitive_and_quoted(paths):
    result = query_files(paths, 'select "l"."id" from "l" inner join "r" on "l"."k" = "r"."k"')
    assert result.column_names == ("l.id",)
    assert result.column("l.id") == [1, 1, 3, 3]


def test_join_int64_float64_keys(tmp_path):
    left = tmp_path / "li.caef"
    right = tmp_path / "rf.caef"
    write_file(left, Table(Schema([ColumnSchema("k", "int64")]), {"k": [1, 2]}))
    write_file(right, Table(Schema([ColumnSchema("k", "float64")]), {"k": [2.0, 3.0]}))
    result = query_files(
        {"a": left, "b": right}, "SELECT a.k, b.k FROM a INNER JOIN b ON a.k = b.k"
    )
    assert joined_rows(result) == [[2, 2.0]]


def test_join_utf8_keys(tmp_path):
    left = tmp_path / "lu.caef"
    right = tmp_path / "ru.caef"
    write_file(left, Table(Schema([ColumnSchema("k", "utf8")]), {"k": ["x", "y"]}))
    write_file(right, Table(Schema([ColumnSchema("k", "utf8")]), {"k": ["y"]}))
    result = query_files(
        {"a": left, "b": right}, "SELECT a.k FROM a INNER JOIN b ON a.k = b.k"
    )
    assert result.column("a.k") == ["y"]


def test_query_files_single_table_no_join(paths):
    result = query_files(paths, "SELECT id, name FROM l WHERE id >= 2 ORDER BY id")
    assert result.column_names == ("id", "name")
    assert result.column("id") == [2, 3, 4]


def test_query_files_single_table_qualified(paths):
    result = query_files(paths, "SELECT l.id FROM l WHERE l.id = 1")
    assert result.column_names == ("id",)
    assert result.column("id") == [1]


def test_query_files_unreferenced_source_not_read(tmp_path):
    left = tmp_path / "left.caef"
    write_file(left, Table(LEFT_SCHEMA, LEFT_DATA))
    sources = {"l": left, "missing": tmp_path / "does-not-exist.caef"}
    result = query_files(sources, "SELECT id FROM l WHERE id = 1")
    assert result.column("id") == [1]


# ---------------------------------------------------------------------------
# Syntax errors (raised before any file is read)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM l JOIN r ON l.k = r.k",                # missing INNER/LEFT/...
        "SELECT * FROM l CROSS JOIN r ON l.k = r.k",          # unsupported join type
        "SELECT * FROM l FULL JOIN r ON l.k = r.k",           # FULL without OUTER
        "SELECT * FROM l FULL OUTER r ON l.k = r.k",          # missing JOIN
        "SELECT * FROM l RIGHT OUTER JOIN r ON l.k = r.k",    # RIGHT takes no OUTER
        "SELECT * FROM l LEFT OUTER JOIN r ON l.k = r.k",     # LEFT takes no OUTER
        "SELECT * FROM l FULL OUTER JOIN r",                  # missing ON
        "SELECT * FROM l INNER r ON l.k = r.k",               # missing JOIN
        "SELECT * FROM l INNER JOIN r",                       # missing ON
        "SELECT * FROM l INNER JOIN r ON l.k",                # incomplete ON
        "SELECT * FROM l INNER JOIN r ON l.k > r.k",          # non-equality ON
        "SELECT * FROM l INNER JOIN r ON k = r.k",            # unqualified ON key
        "SELECT * FROM l INNER JOIN r ON l.k = r.k AND l.id = 1",  # compound ON
        "SELECT * FROM l x INNER JOIN r ON l.k = r.k",        # alias
    ],
)
def test_join_syntax_errors_before_file_access(sql):
    missing = {"l": "/nonexistent/l.caef", "r": "/nonexistent/r.caef"}
    with pytest.raises(QuerySyntaxError):
        query_files(missing, sql)


@pytest.mark.parametrize(
    "sql",
    [
        # A repeated table is no longer a syntax problem (chains are legal);
        # it is rejected as a validation error before any file is read.
        "SELECT * FROM l INNER JOIN r ON l.k = r.k INNER JOIN r ON l.k = r.k",
        "SELECT * FROM l INNER JOIN r ON l.k = r.k LEFT JOIN r ON l.k = r.k",
    ],
)
def test_duplicate_table_in_chain_is_validation_error(sql):
    missing = {"l": "/nonexistent/l.caef", "r": "/nonexistent/r.caef"}
    with pytest.raises(QueryValidationError):
        query_files(missing, sql)


def test_query_file_still_rejects_join(paths):
    with pytest.raises(QuerySyntaxError):
        query_file(paths["l"], "SELECT * FROM input INNER JOIN r ON input.k = r.k")


def test_query_file_still_rejects_qualified(paths):
    with pytest.raises(QuerySyntaxError):
        query_file(paths["l"], "SELECT input.id FROM input")


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_unknown_from_table(paths):
    with pytest.raises(QueryValidationError):
        query_files(paths, "SELECT * FROM nope")


def test_unknown_join_table(paths):
    with pytest.raises(QueryValidationError):
        query_files(paths, "SELECT * FROM l INNER JOIN nope ON l.k = nope.k")


def test_duplicate_table(paths):
    with pytest.raises(QueryValidationError):
        query_files(paths, "SELECT * FROM l INNER JOIN l ON l.k = l.k")


def test_unqualified_column_in_join(paths):
    with pytest.raises(QueryValidationError):
        query_files(paths, "SELECT id FROM l INNER JOIN r ON l.k = r.k")


def test_unqualified_column_in_join_where(paths):
    with pytest.raises(QueryValidationError):
        query_files(paths, "SELECT l.id FROM l INNER JOIN r ON l.k = r.k WHERE id = 1")


def test_unknown_column_in_join(paths):
    with pytest.raises(QueryValidationError):
        query_files(paths, "SELECT l.nope FROM l INNER JOIN r ON l.k = r.k")


def test_unknown_table_qualifier(paths):
    with pytest.raises(QueryValidationError):
        query_files(paths, "SELECT z.id FROM l INNER JOIN r ON l.k = r.k")


def test_on_key_sides_are_swap_symmetric(paths):
    # The ON equality accepts its two sides in either written order.
    forward = "SELECT * FROM l INNER JOIN r ON l.k = r.k"
    reversed_sql = "SELECT * FROM l INNER JOIN r ON r.k = l.k"
    assert joined_rows(query_files(paths, reversed_sql)) == joined_rows(
        query_files(paths, forward)
    )
    # A swapped ON in a later chain step works too, with the explain plan
    # always reporting the preceding table as "left" and the new one as
    # "right".
    plan = explain_files(paths, "SELECT * FROM l LEFT JOIN r ON r.k = l.k")
    join = next(op for op in plan["operators"] if op["operator"] == "Join")
    assert join["left"] == {"table": "l", "column": "k"}
    assert join["right"] == {"table": "r", "column": "k"}


def test_on_key_same_table(paths):
    with pytest.raises(QueryValidationError):
        query_files(paths, "SELECT * FROM l INNER JOIN r ON l.k = l.id")


def test_on_unknown_key_column(paths):
    with pytest.raises(QueryValidationError):
        query_files(paths, "SELECT * FROM l INNER JOIN r ON l.nope = r.k")


def test_incompatible_key_types(tmp_path):
    left = tmp_path / "a.caef"
    right = tmp_path / "b.caef"
    write_file(left, Table(Schema([ColumnSchema("k", "utf8")]), {"k": ["x"]}))
    write_file(right, Table(Schema([ColumnSchema("k", "int64")]), {"k": [1]}))
    with pytest.raises(QueryValidationError):
        query_files({"a": left, "b": right}, "SELECT * FROM a INNER JOIN b ON a.k = b.k")


def test_duplicate_result_column(paths):
    with pytest.raises(QueryValidationError):
        query_files(paths, "SELECT l.id, l.id FROM l INNER JOIN r ON l.k = r.k")


def test_duplicate_result_column_case_variant(paths):
    with pytest.raises(QueryValidationError):
        query_files(paths, "SELECT L.id, l.id FROM l INNER JOIN r ON l.k = r.k")


# ---------------------------------------------------------------------------
# sources validation (ValueError, no file access)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sources",
    [
        "not-a-mapping",
        42,
        {},
        {"": "x.caef"},
        {1: "x.caef"},
        {"t": 123},
        {"t": None},
    ],
)
def test_invalid_sources_value_error(sources):
    with pytest.raises(ValueError):
        query_files(sources, "SELECT * FROM t")


def test_invalid_sources_no_file_access():
    # An invalid key must raise ValueError before the (nonexistent) path is
    # ever touched -- an OSError here would prove a file access happened.
    with pytest.raises(ValueError):
        query_files({"": "/nonexistent/x.caef"}, "SELECT * FROM t")
    with pytest.raises(ValueError):
        query_files({"t": 123}, "SELECT * FROM t")


# ---------------------------------------------------------------------------
# File errors
# ---------------------------------------------------------------------------


def test_missing_file_oserror(tmp_path):
    sources = {"l": tmp_path / "nope.caef"}
    with pytest.raises(OSError):
        query_files(sources, "SELECT * FROM l")


def test_corrupt_referenced_file(paths, tmp_path):
    bad = tmp_path / "bad.caef"
    bad.write_bytes(b"garbage")
    with pytest.raises(ColumnarFormatError):
        query_files({"l": paths["l"], "r": bad}, "SELECT * FROM l INNER JOIN r ON l.k = r.k")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_query_files_success(paths, capsys):
    sources_json = json.dumps({"l": str(paths["l"]), "r": str(paths["r"])})
    sql = "SELECT l.id, r.tag FROM l LEFT JOIN r ON l.k = r.k"
    assert main(["query-files", sources_json, sql]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert list(payload) == ["columns", "rows"]
    assert [c["name"] for c in payload["columns"]] == ["l.id", "r.tag"]
    assert payload["columns"][1]["nullable"] is True
    assert payload["rows"] == [
        [1, "x"],
        [1, "y"],
        [2, None],
        [3, "x"],
        [3, "y"],
        [4, None],
    ]
    # Byte-identical across runs without explicit ORDER BY.
    assert main(["query-files", sources_json, sql]) == 0
    assert capsys.readouterr().out == out


def test_cli_query_files_bad_sql_exit_2(paths, capsys):
    sources_json = json.dumps({"l": str(paths["l"]), "r": str(paths["r"])})
    assert main(["query-files", sources_json, "SELECT * FROM l JOIN r"]) == 2
    assert capsys.readouterr().err


def test_cli_query_files_invalid_sources_json_exit_2(capsys):
    assert main(["query-files", "{not json", "SELECT * FROM l"]) == 2
    assert capsys.readouterr().err


def test_cli_query_files_bad_sources_shape_exit_2(capsys):
    assert main(["query-files", "[]", "SELECT * FROM l"]) == 2
    assert capsys.readouterr().err


def test_cli_query_files_missing_file_exit_1(tmp_path, capsys):
    sources_json = json.dumps({"l": str(tmp_path / "nope.caef")})
    assert main(["query-files", sources_json, "SELECT * FROM l"]) == 1
    assert capsys.readouterr().err


def test_cli_query_still_works(paths, capsys):
    rc = main(["query", str(paths["l"]), "SELECT id FROM input WHERE id = 1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert json.loads(out)["rows"] == [[1]]
