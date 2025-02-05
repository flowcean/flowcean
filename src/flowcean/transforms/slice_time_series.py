import polars as pl
from flowcean.utils import is_timeseries_feature
from typing import override
from flowcean.core.transform import Transform

class SliceTimeSeries(Transform):

    def __init__(self, counter_col : str, dur_in_sec : int = 1) -> None:
        super().__init__()

        self.dur = dur_in_sec
        self.counter_col = counter_col

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        df = data.collect()
        remaining_time_series_columns = [
            col for col in data.columns if is_timeseries_feature(data, col) and self.counter_col != col
        ]
        const_col = df.select(pl.exclude(*remaining_time_series_columns, self.counter_col))
        
        df = df.explode(self.counter_col).explode(*remaining_time_series_columns)

        for col_name in remaining_time_series_columns:
            df = df.filter(
                pl.col(col_name).struct.field("time").is_between(
                    pl.col(self.counter_col).struct.field("time") - pl.duration(seconds=self.dur),
                    pl.col(self.counter_col).struct.field("time")
                )
            )

        df = df.group_by(self.counter_col, maintain_order=True).agg(remaining_time_series_columns)
        df = df.select(
                pl.col(self.counter_col).map_elements(lambda x:[x]), pl.col(remaining_time_series_columns)
            ).with_columns(
                const_col.select(
                    pl.col(col_name).repeat_by(df.height).flatten() for col_name in const_col.columns
                )
            )
        return df.select(data.collect_schema().names()).lazy()
