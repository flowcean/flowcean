import polars as pl
from typing_extensions import override

from flowcean.core.transform import Transform
from flowcean.polars.is_time_series import is_timeseries_feature


class SliceTimeSeries(Transform):
    """Slices time series features based on a counter column.

    Extracts time series segments corresponding to each timestamp in the
    counter column. This transformation enables slicing time series data
    into fixed-duration windows relative to reference timestamps.

    The following example demonstrates how to use `SliceTimeSeries` in a
    `run.py` file. Given the input data:

    | feature_a                   | feature_b                   | const |
    | --------------------------- | --------------------------- | ----- |
    | list[struct[time,struct[]]] | list[struct[time,struct[]]] | int   |
    | [{12:26:01.0, {1.2}},       | [{12:26:05.0, {1.0}},       | 1     |
    |  {12:26:02.0, {2.4}},       |  {12:26:10.0, {2.0}}]       |       |
    |  {12:26:03.0, {3.6}},       |                             |       |
    |  {12:26:04.0, {4.8}}]       |                             |       |

    The `SliceTimeSeries` transform extracts a time window of 2 seconds
    before each `feature_b` timestamp from `feature_a`:

    ```
        ...
        environment.load()
        data = environment.get_data()
        transform = SliceTimeSeries(
            counter_col="feature_b",
            duration=2,
        )
        transformed_data = transform.transform(data)
        ...
    ```

    The resulting DataFrame after the transformation:

    | feature_a                   | feature_b                   | const |
    | --------------------------- | --------------------------- | ----- |
    | list[struct[time,struct[]]] | list[struct[time,struct[]]] | int   |
    | [{12:26:03.0, {3.6}},       | [{12:26:05.0, {1.0}}],      | 1     |
    |  {12:26:04.0, {4.8}},       |                             |       |
    |  {12:26:05.0, {5.0}}]       |                             |       |
    | [{12:26:08.0, {8.0}},       | [{12:26:10.0, {2.0}}],      | 1     |
    |  {12:26:09.0, {9.0}},       |                             |       |
    |  {12:26:10.0, {10.0}}]      |                             |       |
    """

    def __init__(self, counter_col: str, duration: int = 1) -> None:
        """Initialize SliceTimeseriesTransform.

        Args:
            counter_col: The name of the counter column used to slice the time
                series.
            duration: The duration in seconds for slicing the time series.
                Default is 1 second.
        """
        super().__init__()

        self.dur = duration
        self.counter_col = counter_col

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        df = data.collect()
        n_rows = df.height
        remaining_time_series_columns = [
            col
            for col in data.columns
            if is_timeseries_feature(data, col) and self.counter_col != col
        ]
        const_col = df.select(
            pl.exclude(*remaining_time_series_columns, self.counter_col),
        )
        df = (
            df.select(pl.exclude(const_col.columns))
            .explode(self.counter_col)
            .explode(*remaining_time_series_columns)
        )

        df = df.select(
            pl.when(
                pl.col(remaining_time_series_columns)
                .struct.field("time")
                .is_between(
                    pl.col(self.counter_col).struct.field("time")
                    - pl.duration(seconds=self.dur),
                    pl.col(self.counter_col).struct.field("time"),
                ),
            )
            .then(
                pl.col(remaining_time_series_columns),
            )
            .otherwise(
                None,
            ),
            pl.col(self.counter_col),
        )
        df = (
            df.unique(maintain_order=True)
            .group_by(self.counter_col, maintain_order=True)
            .agg(remaining_time_series_columns)
        )
        df = df.select(
            pl.col(self.counter_col).map_elements(lambda x: [x]),
            pl.col(remaining_time_series_columns).list.drop_nulls(),
        ).with_columns(
            const_col.select(
                pl.col(col_name).repeat_by(int(df.height / n_rows)).flatten()
                for col_name in const_col.columns
            ),
        )
        return df.select(data.collect_schema().names()).lazy()
