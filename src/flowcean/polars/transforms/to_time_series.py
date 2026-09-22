from typing import override

import polars as pl

from flowcean.core import Transform


class ToTimeSeries(Transform):
    """Collect rows sharing one time axis into a multivariate time series.

    The result has one row and one column named ``name``, with dtype
    ``List(Struct({time: <time dtype>, value: Struct({...})}))``. Each sample's
    ``value`` contains all input columns except ``time_feature``, retaining
    their names and dtypes. Even a single value column remains struct-valued.
    Input row order is preserved; timestamps are not sorted.

    Empty input produces one empty list with the same nested schema. If the
    input contains only the time column, each sample has an empty value
    struct. A missing time column raises Polars' ``ColumnNotFoundError``.

    For multiple clocks, select each clock and its associated value columns
    in separate branches, apply this transform once per branch with distinct
    output names, then combine the one-row results horizontally. No input
    columns are retained outside the resulting series.

    Args:
        time_feature: The single input column containing timestamps.
            Mappings from value columns to clocks are not supported.
        name: Name of the resulting time series column.
    """

    time_feature: str
    _output_name: str

    def __init__(
        self,
        time_feature: str,
        *,
        name: str = "time_series",
    ) -> None:
        super().__init__()
        if not isinstance(time_feature, str):
            msg = "time_feature must be a single column name"
            raise TypeError(msg)
        self.time_feature = time_feature
        self._output_name = name

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        value_features = [
            feature
            for feature in data.collect_schema().names()
            if feature != self.time_feature
        ]
        values = (
            pl.struct(value_features)
            if value_features
            else pl.lit({}, dtype=pl.Struct({}))
        )
        return data.select(
            pl.struct(
                pl.col(self.time_feature).alias("time"),
                values.alias("value"),
            )
            .implode()
            .alias(self._output_name),
        )
