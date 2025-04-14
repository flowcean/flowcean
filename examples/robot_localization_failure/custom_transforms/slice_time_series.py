import polars as pl
from typing_extensions import override

from flowcean.core.transform import Transform
from flowcean.polars.is_time_series import is_timeseries_feature


class SliceTimeSeries(Transform):
    """Slices time series features based on a counter column.

    Slice a dataframe based on a counter column.

    feature = feature[
        counter_column.time[i-1] < time < counter_column.time[i]
    ]

    Example:
    Consider the following DataFrame and a time window of 2 seconds:

    | feature_a                   | counter_column              | const |
    | --------------------------- | --------------------------- | ----- |
    | list[struct[time,struct[]]] | list[struct[time,struct[]]] | int   |
    | [{12:26:01.0, {1.1}},       | [{12:26:05.0, {3.0}},       | 1     |
    |  {12:26:02.0, {2.2}},       |  {12:26:08.0, {6.0}},       |       |
    |  {12:26:03.0, {3.3}},       |                             |       |
    |  {12:26:04.0, {4.4}},       |                             |       |
    |  {12:26:02.0, {5.5}},       |                             |       |
    |  {12:26:03.0, {6.6}},       |                             |       |
    |  {12:26:04.0, {7.7}}]       |                             |       |

    The SliceTimeSeries transform is called with
    '''
        environment.load()
        data = environment.get_data()
        transform = SliceTimeSeries(
            counter_column="counter_column",
            duration=2,
        )
        transformed_data = transform.transform(data)
    '''

    The resulting DataFrame after the transformation:

    | feature_a                   | counter_column              | const |
    | --------------------------- | --------------------------- | ----- |
    | list[struct[time,struct[]]] | list[struct[time,struct[]]] | int   |
    | [{12:26:03.0, {1.1}},       | [{12:26:05.0, {3.0}}],      | 1     |
    |  {12:26:04.0, {2.2}}]       |                             |       |
    | [{12:26:08.0, {3.3}},
       {12:26:08.0, {4.4}},       | [{12:26:10.0, {6.0}}],      | 1     |
    |  {12:26:09.0, {5.5}}]       |                             |       |
    """

    def __init__(self, counter_column: str, deadzone: int) -> None:
        """Initialize SliceTimeseriesTransform.

        Args:
            counter_column: The name of the counter column used to slice the
                time series.
            deadzone: Additional nanoseconds to consider when slicing the time
                series.
        """
        super().__init__()

        self.deadzone = deadzone
        self.counter_column = counter_column

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        names = data.collect_schema().names()
        counter_column_dtype = data.collect_schema()[self.counter_column]

        # Collect names of all other time series columns
        remaining_time_series_columns = [
            col
            for col in data.collect_schema().names()
            if is_timeseries_feature(data, col) and self.counter_column != col
        ]

        # Collect names of all constant columns
        const_columns = (
            data.select(
                pl.exclude(
                    *remaining_time_series_columns,
                    self.counter_column,
                ),
            )
            .collect_schema()
            .names()
        )

        # Explode all time series columns
        data = data.explode(self.counter_column).explode(
            *remaining_time_series_columns,
        )

        # Filter all time series columns based on the counter column
        data = data.select(
            pl.when(
                pl.col(remaining_time_series_columns)
                .struct.field("time")
                .cast(int)
                <= pl.col(self.counter_column).struct.field("time").cast(int)
                + self.deadzone,
            )
            .then(
                pl.col(remaining_time_series_columns),
            )
            .otherwise(
                None,
            ),
            pl.col(self.counter_column),
            pl.col(const_columns),
        )

        data = data.unique(
            remaining_time_series_columns,
            keep="first",
            maintain_order=True,
        )

        data = data.select(
            pl.when(
                pl.col(remaining_time_series_columns).struct.field("time")
                <= pl.col(self.counter_column).struct.field("time"),
            )
            .then(
                pl.col(remaining_time_series_columns),
            )
            .otherwise(
                None,
            ),
            pl.col(self.counter_column),
            pl.col(const_columns),
        )

        # Recreate time series based on the counter column
        data = (
            data.unique(maintain_order=True)
            .group_by(
                [self.counter_column, *const_columns],
                maintain_order=True,
            )
            .agg(remaining_time_series_columns)
        )

        # Remove Clean up the DataFrame
        return data.select(
            pl.col(self.counter_column).map_elements(
                lambda x: [x],
                return_dtype=counter_column_dtype,
            ),
            pl.col(remaining_time_series_columns).list.drop_nulls(),
            pl.col(
                const_columns,
            ),
        ).select(names)
