from typing import override

import polars as pl

from flowcean.core.transform import Transform


class Derive(Transform):
    def __init__(self, column: str) -> None:
        self.column = column

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.with_columns(
            (pl.col(self.column).diff()).alias(self.column),
        )
