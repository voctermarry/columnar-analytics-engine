"""Tests for the in-memory table model: schema, validation and projection."""

import math

import pytest

from columnar_analytics import (
    BOOL,
    FLOAT64,
    INT64,
    UTF8,
    Field,
    Schema,
    Table,
)


def make_schema():
    return Schema(
        [
            Field("flag", BOOL),
            Field("n", INT64),
            Field("ratio", FLOAT64),
            Field("label", UTF8, nullable=False),
        ]
    )


def make_table():
    schema = make_schema()
    return Table(
        schema,
        {
            "flag": [True, False, None, True],
            "n": [1, -2, 3, None],
            "ratio": [1.5, -0.25, 3.0, None],
            "label": ["a", "b", "c", "中文"],
        },
    )


def test_field_rejects_unknown_type():
    with pytest.raises(ValueError):
        Field("x", "decimal")


def test_field_rejects_empty_and_non_string_name():
    with pytest.raises(ValueError):
        Field("", INT64)
    with pytest.raises(ValueError):
        Field(123, INT64)  # type: ignore[arg-type]


def test_schema_rejects_duplicate_and_empty():
    with pytest.raises(ValueError):
        Schema([Field("a", INT64), Field("a", FLOAT64)])
    with pytest.raises(ValueError):
        Schema([])


def test_schema_preserves_order():
    schema = make_schema()
    assert schema.names == ["flag", "n", "ratio", "label"]


def test_table_requires_all_columns():
    schema = make_schema()
    with pytest.raises(ValueError, match="missing columns"):
        Table(
            schema,
            {
                "flag": [True],
                "n": [1],
                "ratio": [1.0],
            },
        )

    with pytest.raises(ValueError, match="unexpected columns"):
        Table(
            schema,
            {
                "flag": [True],
                "n": [1],
                "ratio": [1.0],
                "label": ["x"],
                "extra": [9],
            },
        )


def test_table_requires_equal_lengths():
    schema = make_schema()
    with pytest.raises(ValueError, match="has 3 rows"):
        Table(
            schema,
            {
                "flag": [True, False],
                "n": [1, 2, 3],
                "ratio": [1.0, 2.0, 3.0],
                "label": ["a", "b", "c"],
            },
        )


def test_table_rejects_wrong_types():
    schema = make_schema()
    base = {
        "flag": [True],
        "n": [1],
        "ratio": [1.0],
        "label": ["x"],
    }

    bad_bool = dict(base)
    bad_bool["flag"] = ["true"]
    with pytest.raises(ValueError, match="bool"):
        Table(schema, bad_bool)

    bad_int = dict(base)
    bad_int["n"] = [1.5]
    with pytest.raises(ValueError, match="int64"):
        Table(schema, bad_int)

    bad_int_bool = dict(base)
    bad_int_bool["n"] = [True]
    with pytest.raises(ValueError, match="int64"):
        Table(schema, bad_int_bool)

    bad_float_str = dict(base)
    bad_float_str["ratio"] = ["1.0"]
    with pytest.raises(ValueError, match="float64"):
        Table(schema, bad_float_str)

    bad_str = dict(base)
    bad_str["label"] = [b"x"]
    with pytest.raises(ValueError, match="utf8"):
        Table(schema, bad_str)


def test_table_rejects_nan_and_infinity():
    schema = make_schema()
    base = {
        "flag": [True],
        "n": [1],
        "ratio": [0.0],
        "label": ["x"],
    }
    for bad in (math.nan, math.inf, -math.inf):
        cols = dict(base)
        cols["ratio"] = [bad]
        with pytest.raises(ValueError, match="NaN or infinite"):
            Table(schema, cols)


def test_non_nullable_column_rejects_none():
    schema = make_schema()
    with pytest.raises(ValueError, match="not nullable"):
        Table(
            schema,
            {
                "flag": [True],
                "n": [1],
                "ratio": [1.0],
                "label": [None],
            },
        )


def test_int64_range_enforced():
    schema = Schema([Field("n", INT64)])
    with pytest.raises(ValueError, match="int64 range"):
        Table(schema, {"n": [2**63]})
    with pytest.raises(ValueError, match="int64 range"):
        Table(schema, {"n": [-(2**63) - 1]})


def test_columns_returned_are_copies():
    table = make_table()
    column = table.column("n")
    column.append(999)
    assert table.column("n") == [1, -2, 3, None]


def test_project_reorders_and_subsets():
    table = make_table()
    projected = table.project(["label", "n"])
    assert projected.schema.names == ["label", "n"]
    assert projected.column("label") == ["a", "b", "c", "中文"]
    assert projected.column("n") == [1, -2, 3, None]
    assert projected.row_count == 4


def test_project_unknown_raises_keyerror():
    table = make_table()
    with pytest.raises(KeyError):
        table.project(["n", "missing"])


def test_project_duplicate_raises_valueerror():
    table = make_table()
    with pytest.raises(ValueError, match="duplicate"):
        table.project(["n", "n"])


def test_unicode_column_names():
    schema = Schema([Field("列名", INT64, nullable=False)])
    table = Table(schema, {"列名": [1, 2, 3]})
    assert table.project(["列名"]).column("列名") == [1, 2, 3]
