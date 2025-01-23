import polars as pl
from typing import override
from flowcean.core.transform import Transform

class SliceTimeSeries(Transform):
    
    def __init__(self, num_slices : int, slice_length : int, series_name : str) -> None:
        super().__init__()
        
        self.n_slices = num_slices
        self.len = slice_length
    
    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        
        collected_data = data.collect()

        print(collected_data.select())

        data = pl.LazyFrame()
        return data