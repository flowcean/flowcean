from typing import cast

import polars as pl


def is_timeseries_feature(
    df: pl.DataFrame | pl.LazyFrame, column_name: str
) -> bool:
    data_type = df.collect_schema()[column_name]

    if data_type.base_type() != pl.List:
        return False

    inner_type: pl.DataType = cast(pl.DataType, cast(pl.List, data_type).inner)
    if inner_type.base_type() != pl.Struct:
        return False

    field_names = [field.name for field in cast(pl.Struct, inner_type).fields]
    return "time" in field_names and "value" in field_names
