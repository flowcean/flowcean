from typing import override

import polars as pl

from flowcean.core.transform import Transform


class Rechunk(Transform):
    """Rechunks a dataframe.

    Rearranges a dataframe so that it resides in a contiguous block of memory.
    This improves the performance of any subsequent transform performed on the
    rechunked dataframe. However, this operation can be costly depending on the
    size of the dataframe, so it should be used with care and only when deemed
    necessary.
    """

    def __init__(self) -> None:
        """Initializes the Rechunk transform."""
        super().__init__()

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.rechunk()
