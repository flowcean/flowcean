import polars as pl
from typing import override
from flowcean.core.transform import Transform

class SliceTimeSeries(Transform):
    
    def __init__(self, num_slices : int, slice_length : int) -> None:
        super().__init__()
        
        self.n_slices = num_slices
        self.len = slice_length
    
    @override
    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        
        
        return None