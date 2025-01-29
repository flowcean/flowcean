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
        col_names = df.select(pl.exclude(self.counter_col)).columns
        df = df.explode(self.counter_col).explode(*col_names)
        
        for col_name in df.columns:
            if not col_name == self.counter_col:
                df = df.filter(
                    pl.col(col_name).struct.field("time").is_between(
                        pl.col(self.counter_col).struct.field("time") - pl.duration(seconds=self.dur),
                        pl.col(self.counter_col).struct.field("time")
                    )
                )        

        # print(df)
        df = df.group_by(self.counter_col).agg(col_names).sort(self.counter_col)
        # df = df.sort(self.counter_col)
        # df = df.select(pl.col(self.counter_col).implode(), pl.col(*col_names))
        # df = df.select(self.counter_col).get_column()
        # print(df)
        # print(df.select(data.collect_schema().names()))
        
        return df.select(data.collect_schema().names()).lazy()