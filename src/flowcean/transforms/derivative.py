from typing import override

import polars as pl

from flowcean.core.transform import Transform


class Derivative(Transform):
    def __init__(self, column: str) -> None:
        self.column = column

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.with_columns(
            pl.col(self.column).map_elements(
                lambda x: pl.Series(
                    name=self.column,
                    values=[0] + [
                        float(x[i] - x[i-1]) for i in range(1, len(x))
                    ],
                    dtype=pl.Float64,
                ),
            ),
        )
