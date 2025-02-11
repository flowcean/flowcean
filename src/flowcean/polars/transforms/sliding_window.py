import polars as pl
from typing_extensions import override

from flowcean.core import Transform


class SlidingWindow(Transform):
    """Transforms the data with a sliding window.

    The sliding window transform transforms the data by creating a sliding
    window over the row dimension. The data is then transformed by creating a
    new column for each column in the original data. The new columns are
    named by appending the index of the row in the sliding window to the
    original column name.
    As an example, consider the following data:

     x | y  | z
    ---|----|-----
     1 | 10 | 100
     2 | 20 | 200
     3 | 30 | 300
     4 | 40 | 400
     5 | 50 | 500

    If we apply a sliding window with a window size of 3, we get the following

    x_0 | y_0 | z_0 | x_1 | y_1 | z_1 | x_2 | y_2 | z_2
    ----|-----|-----|-----|-----|-----|-----|-----|-----
     1  | 10  | 100 | 2   | 20  | 200 | 3   | 30  | 300
     2  | 20  | 200 | 3   | 30  | 300 | 4   | 40  | 400
     3  | 30  | 300 | 4   | 40  | 400 | 5   | 50  | 500

    Args:
        window_size: size of the sliding window.
    """

    def __init__(self, window_size: int) -> None:
        self.window_size = window_size

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.select(
            [
                pl.all()
                .shift(self.window_size - 1 - i)
                .slice(self.window_size - 1)
                .name.suffix(f"_{i}")
                for i in range(self.window_size)
            ],
        )
