import polars as pl
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
        remaining_columns = df.select(pl.exclude(self.counter_col)).columns
        df = df.explode(self.counter_col).explode(*remaining_columns)
        
        for col_name in remaining_columns:
            df = df.filter(
                pl.col(col_name).struct.field("time").is_between(
                    pl.col(self.counter_col).struct.field("time") - pl.duration(seconds=self.dur),
                    pl.col(self.counter_col).struct.field("time")
                )
            )        

        df = df.group_by(self.counter_col, maintain_order=True).agg(remaining_columns)
        df = df.select(pl.col(self.counter_col).map_elements(lambda x:[x]), pl.col(remaining_columns))
        return df.select(data.collect_schema().names()).lazy()