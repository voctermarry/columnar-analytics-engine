"""Regression tests for the batch column-vector expression path.

These pin the invariants the vectorised WHERE / sort-key / projection /
pre-DISTINCT projection evaluator must keep: three-valued logic over a whole
batch of rows (and empty / all-NULL batches), per-row CASE routing and
AND/OR short-circuiting, the exact evaluation timing around ORDER BY /
LIMIT / DISTINCT, nullable bool predicates and float64 signed-zero
deduplication.  Every case goes through the public entry points and is
exercised on both a v1 and a v2 (partitioned) file.
"""

from __future__ import annotations

import pytest

from columnar_analytics import (
    ColumnSchema,
    QueryValidationError,
    Schema,
    Table,
    query_file,
    write_file,
    write_partitioned_file,
)


SCHEMA = Schema(
    [
        ColumnSchema("id", "int64"),
        ColumnSchema("n", "int64", nullable=True),
        ColumnSchema("f", "float64", nullable=True),
        ColumnSchema("flag", "bool"),
        ColumnSchema("zflag", "bool", nullable=True),
        ColumnSchema("s", "utf8", nullable=True),
    ]
)


@pytest.fixture(params=["v1", "v2"])
def path(request, tmp_path):
    p = tmp_path / f"t_{request.param}.caef"
    table = Table(
        SCHEMA,
        {
            # The /(id-3) divisor is zero for exactly one row (id == 3),
            # giving the laziness tests a concrete row to route around.
            "id": [1, 2, 3, 4, 5],
            "n": [10, None, 30, 40, None],
            "f": [1.5, -0.0, None, 0.0, 1e308],
            "flag": [True, False, True, False, True],
            "zflag": [True, None, False, None, True],
            "s": ["a", "b", None, "a", "c"],
        },
    )
    if request.param == "v1":
        write_file(p, table)
    else:
        write_partitioned_file(p, table, 2)
    return p


def _column(table, name):
    return table.column(name)


# ---------------------------------------------------------------------------
# Empty and all-NULL batches
# ---------------------------------------------------------------------------


def test_empty_table_batch(path, tmp_path):
    p = tmp_path / "empty.caef"
    empty = Table(SCHEMA, {name: [] for name in SCHEMA.names})
    write_file(p, empty)
    result = query_file(
        p,
        "select id, n * 2 + 1 as x, case when flag then 1.0 else 2.0 end as b "
        "from input where f is not null order by id limit 10",
    )
    assert result.row_count == 0
    assert [(c.name, c.type, c.nullable) for c in result.schema.columns] == [
        ("id", "int64", False),
        ("x", "int64", True),
        ("b", "float64", False),
    ]


def test_empty_table_distinct(path, tmp_path):
    p = tmp_path / "empty2.caef"
    empty = Table(SCHEMA, {name: [] for name in SCHEMA.names})
    write_file(p, empty)
    result = query_file(p, "select distinct n, f, zflag from input order by n")
    assert result.row_count == 0


def test_all_null_column_propagates_through_batch(path):
    p = path
    result = query_file(
        p,
        "select id, n + 1 as a, n / 2 as b, -n as c, case when n is null then 0 else n end as d "
        "from input order by id",
    )
    assert _column(result, "a") == [11, None, 31, 41, None]
    assert _column(result, "b") == [5.0, None, 15.0, 20.0, None]
    assert _column(result, "c") == [-10, None, -30, -40, None]
    assert _column(result, "d") == [10, 0, 30, 40, 0]


# ---------------------------------------------------------------------------
# Three-valued logic over a mixed batch, incl. a nullable bool column
# ---------------------------------------------------------------------------


def test_nullable_bool_direct_predicate(path):
    result = query_file(path, "select id from input where zflag order by id")
    assert _column(result, "id") == [1, 5]
    result = query_file(path, "select id from input where not zflag order by id")
    assert _column(result, "id") == [3]
    result = query_file(
        path, "select id from input where zflag is null order by id"
    )
    assert _column(result, "id") == [2, 4]


def test_and_or_null_propagation_batch(path):
    # flag in (True, False, ...); n NULL for id 2/5.  Arithmetic on a NULL
    # yields UNKNOWN, never raises, and only TRUE rows survive WHERE.
    result = query_file(
        path, "select id from input where flag and n > 5 order by id"
    )
    assert _column(result, "id") == [1, 3]
    result = query_file(
        path, "select id from input where (not flag) or n is null order by id"
    )
    assert _column(result, "id") == [2, 4, 5]


def test_comparison_null_operand_is_unknown(path):
    result = query_file(path, "select id from input where f >= 0.0 order by id")
    # f: 1.5, -0.0, NULL, 0.0, 1e308 -> -0.0 >= 0 and 0.0 >= 0 both TRUE.
    assert _column(result, "id") == [1, 2, 4, 5]


# ---------------------------------------------------------------------------
# Per-row CASE routing inside one batch (errors live only in unhit branches)
# ---------------------------------------------------------------------------


def test_case_routes_each_row_without_evaluating_unhit_branch(path):
    # 1 / (id - 3) blows up only for id == 3.  Rows 1,2 take the first THEN
    # (the divisor branch is never evaluated for them); rows 4,5 take the
    # ELSE after the condition is FALSE.  The dangerous expression sits in an
    # always-unhit WHEN for every selected row, so nothing raises.
    result = query_file(
        path,
        "select case when id < 3 then id else 1 / (id - 3) end as x "
        "from input where id < 3 order by id",
    )
    assert _column(result, "x") == [1.0, 2.0]
    # Same batch, but id == 3 now selects the divisor branch and must raise.
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "select case when id != 3 then id else 1 / (id - 3) end as x "
            "from input",
        )


def test_case_first_true_only_within_batch(path):
    # The second WHEN carries an int64 overflow in its result, but its
    # condition is always FALSE, so that result is never evaluated: rows that
    # match the first WHEN skip it, the rest fall through to ELSE.
    result = query_file(
        path,
        "select case when flag then 1 when false then 9223372036854775807 + 1 "
        "else 2 end as x from input order by id",
    )
    assert _column(result, "x") == [1, 2, 1, 2, 1]


# ---------------------------------------------------------------------------
# AND / OR short-circuiting must suppress a divisor error per row
# ---------------------------------------------------------------------------


def test_and_short_circuit_skips_error_rows(path):
    # id != 3 is FALSE exactly on the row where 1/(id-3) divides by zero; AND
    # never evaluates the right side for that row.  Of the remaining rows only
    # id 4/5 have a positive quotient.
    result = query_file(
        path, "select id from input where id != 3 and 1 / (id - 3) > 0 order by id"
    )
    assert _column(result, "id") == [4, 5]
    # Without the guard the same batch raises on the id == 3 row.
    with pytest.raises(QueryValidationError):
        query_file(path, "select id from input where 1 / (id - 3) > 0")


def test_or_short_circuit_keeps_error_rows_unevaluated(path):
    # id = 3 is TRUE on the left, so the dividing right side is skipped on
    # precisely that row (which would otherwise divide by zero); the other
    # rows still evaluate the right side normally.
    result = query_file(
        path, "select id from input where id = 3 or 1 / (id - 3) > 0 order by id"
    )
    assert _column(result, "id") == [3, 4, 5]


# ---------------------------------------------------------------------------
# Evaluation timing: WHERE -> sort keys -> LIMIT -> projection, and the
# DISTINCT variant WHERE -> project -> dedup -> sort -> LIMIT
# ---------------------------------------------------------------------------


def test_projection_after_limit_skips_cut_row_error(path):
    # DESC ordering puts the safe rows first; LIMIT keeps only id 5 and 4, so
    # the id == 3 divisor row is cut before the SELECT expression is evaluated.
    result = query_file(
        path,
        "select 1 / (id - 3) as q from input order by id desc limit 2",
    )
    assert _column(result, "q") == [0.5, 1.0]
    # Without LIMIT the same projection still hits id == 3 and raises.
    with pytest.raises(QueryValidationError):
        query_file(path, "select 1 / (id - 3) as q from input order by id desc")


def test_sort_key_evaluated_before_limit(path):
    # The alias sort key runs over every WHERE-selected row before LIMIT, so
    # the id == 3 error surfaces even though LIMIT would keep one row.
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "select 1 / (id - 3) as q from input order by q limit 1",
        )


def test_distinct_projects_before_limit(path):
    # DISTINCT evaluates the projection for every WHERE survivor before the
    # LIMIT; the error on id == 3 must still surface.
    with pytest.raises(QueryValidationError):
        query_file(
            path,
            "select distinct 1 / (id - 3) as q from input order by q limit 1",
        )


def test_distinct_signed_zero_and_first_occurrence(path):
    # f carries -0.0 (id 2) and 0.0 (id 4): they deduplicate to one value and
    # the first occurrence wins.
    result = query_file(path, "select distinct f from input")
    values = _column(result, "f")
    zero_rows = [v for v in values if v == 0.0]
    assert len(zero_rows) == 1  # +0.0 / -0.0 collapse
    # First occurrence order: 1.5 (id1), -0.0 (id2), NULL (id3), 1e308 (id5).
    assert values == [1.5, -0.0, None, 1e308]


def test_outer_style_nulls_and_type_promotion(path):
    # float promotion of an int result in a mixed int/float CASE, only for the
    # rows routed to the int branch.
    result = query_file(
        path,
        "select case when id = 3 then 7 else 0.5 end as x from input order by id",
    )
    assert _column(result, "x") == [0.5, 0.5, 7.0, 0.5, 0.5]
    assert all(isinstance(v, float) for v in _column(result, "x"))
