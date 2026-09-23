from typing import override

import polars as pl

from flowcean.core import Transform


class ToTimeSeries(Transform):
    """Keep synchronized measurements together as one time series.

    Collects all rows into a list of ``{time, value}`` samples in one output
    row and column. A single non-time column supplies ``value`` directly;
    multiple columns form a struct. Dtypes and input order are preserved.
    Empty input produces an empty list; time-only input uses empty structs.

    Select the columns that belong together before applying this transform.
    For separate clocks or signals, transform selected branches independently
    and combine their outputs horizontally.

    Args:
        time_feature: Input timestamp column.
        name: Output column name.
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
        if len(value_features) == 1:
            values = pl.col(value_features[0])
        elif value_features:
            values = pl.struct(value_features)
        else:
            values = pl.lit({}, dtype=pl.Struct({}))
        return data.select(
            pl.struct(
                pl.col(self.time_feature).alias("time"),
                values.alias("value"),
            )
            .implode()
            .alias(self._output_name),
        )
