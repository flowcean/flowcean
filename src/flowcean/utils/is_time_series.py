from typing import cast

import polars as pl


def is_timeseries_feature(df: pl.LazyFrame, name: str) -> bool:
    """Check if the given column is a time series feature.

    A time series feature contains a list of structs with fields _time_ and
    _value_.

    Args:
        df: The DataFrame to check.
        name: The column to check.

    Returns:
        True if the column is a time series feature, False otherwise.
    """
    data_type = df.select(name).collect_schema().dtypes()[0]

    if data_type.base_type() != pl.List:
        return False

    inner_type: pl.DataType = cast(pl.DataType, cast(pl.List, data_type).inner)
    if inner_type.base_type() != pl.Struct:
        return False

    field_names = [field.name for field in cast(pl.Struct, inner_type).fields]
    return "time" in field_names and "value" in field_names
