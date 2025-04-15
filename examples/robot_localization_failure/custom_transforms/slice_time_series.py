import polars as pl
from typing_extensions import override

from flowcean.core.transform import Transform


class SliceTimeSeries(Transform):
    """Slices time series features based on a counter column and deadzone."""

    def __init__(self, counter_column: str, deadzone: int) -> None:
        super().__init__()
        self.counter_column = counter_column
        self.deadzone = deadzone

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        # extract the counter column
        collected_data = data.collect()
        extracted_counter = (
            collected_data.select([self.counter_column])
            .explode([self.counter_column])
            .unnest([self.counter_column])
            .unnest(["value"])
            .rename({"data": "value"})
        )
        print(f"Counter column: {extracted_counter}")

        # Compute the difference in the value column
        df_with_diff = extracted_counter.with_columns(
            value_diff=pl.col("value").diff(),
        )
        # Filter for rows where value increases (diff > 0)
        increases = df_with_diff.filter(pl.col("value_diff") > 0)
        # Get timestamps of the increases
        timestamp_of_increases = increases["time"].to_list()
        intervals = []
        for i in range(len(timestamp_of_increases) - 1):
            # Get the start and end timestamps for each slice
            if i == 0:  # First slice
                # Start 10 seconds before the first increase
                start = timestamp_of_increases[i] - 10_000_000_000
                end = timestamp_of_increases[i]
            else:
                start = timestamp_of_increases[i] + self.deadzone
                end = timestamp_of_increases[i + 1]

            intervals.append((start, end))
        dataframes = []
        for start, end in intervals:
            for feature in data.collect_schema().names():
                time_expression = (
                    pl.element().struct.field("time").cast(pl.Float64)
                )
                data.with_columns(
                    pl.col(feature).list.eval(
                        pl.element().filter(
                            time_expression.ge(start).and_(
                                time_expression.le(end),
                            ),
                        ),
                    ),
                )
            dataframes.append(data)
        #  Stack all the dataframes
        return pl.concat(dataframes)
