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

        df = df.group_by(self.counter_col, maintain_order=True).agg(col_names)
        df = df.select(pl.col(self.counter_col).map_elements(lambda x:[x]), pl.col(col_names))
        # print(df)       
        return df.select(data.collect_schema().names()).lazy()