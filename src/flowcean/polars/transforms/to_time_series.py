from typing import override

import polars as pl

from flowcean.core import Transform


class ToTimeSeries(Transform):
    """Turn a table of timestamped measurements into a single time series.

    Each row becomes a ``{time, value}`` sample, using ``time_feature`` as
    the timestamp and the remaining columns as measurements. A single
    measurement becomes the value directly; multiple measurements are grouped
    in a struct with their column names. Their dtypes stay the same.

    The result is one row with a column named ``name`` containing the samples
    as a list, in their original order.

    Args:
        time_feature: Input timestamp column.
        name: Output column name.

    Example:
        ```python
        >>> data = pl.LazyFrame(
        ...     {
        ...         "t": [0, 1],
        ...         "temperature": [20, 22],
        ...         "pressure": [1000, 1005],
        ...     }
        ... )
        >>> result = ToTimeSeries("t", name="sensors")(data).collect()
        >>> result["sensors"][0][0]
        {'time': 0, 'value': {'temperature': 20, 'pressure': 1000}}

        ```
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
