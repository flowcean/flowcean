from typing import cast

import polars as pl


def is_timeseries_feature(
    target: pl.DataFrame | pl.LazyFrame | pl.Schema,
    name: str,
) -> bool:
    """Check if the given column is a time series feature.

    A time series feature contains a list of structs with fields _time_ and
    _value_.

    Args:
        target: The LazyFrame, DataFrame or schema to check.
        name: The column to check.

    Returns:
        True if the column is a time series feature, False otherwise.
    """
    if isinstance(target, pl.Schema):
        data_type = target.get(name)
    elif isinstance(target, pl.DataFrame):
        data_type = target.schema.get(name)
    elif isinstance(target, pl.LazyFrame):
        data_type = target.select(name).collect_schema().dtypes()[0]

    if data_type is None or data_type.base_type() != pl.List:
        return False

    inner_type: pl.DataType = cast(
        "pl.DataType",
        cast("pl.List", data_type).inner,
    )
    if inner_type.base_type() != pl.Struct:
        return False

    field_names = [
        field.name for field in cast("pl.Struct", inner_type).fields
    ]
    return "time" in field_names and "value" in field_names
