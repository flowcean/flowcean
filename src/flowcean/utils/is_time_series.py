from typing import cast

import polars as pl


def is_timeseries_feature(df: pl.DataFrame, column_name: str) -> bool:
    data_type = df.select(column_name).dtypes[0]

    if data_type.base_type() != pl.List:
        return False

    inner_type: pl.DataType = cast(pl.DataType, cast(pl.List, data_type).inner)
    if inner_type.base_type() != pl.Struct:
        return False

    field_names = [field.name for field in cast(pl.Struct, inner_type).fields]
    return "time" in field_names and "value" in field_names
