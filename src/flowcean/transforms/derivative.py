from typing import override

import polars as pl

from flowcean.core.transform import Transform


class Derivative(Transform):
    """Calculate the derivative applied to a flatten LazyFrame."""

    def __init__(self, column: str) -> None:
        self.column = column

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        columns = [
            col for col in data.collect_schema().names()
            if col.startswith(self.column)
        ]
        if columns:
            derivatives = [
                (pl.col(columns[i + 1]) - pl.col(columns[i])).alias(columns[i])
                for i in range(len(columns) - 1)
            ]
            return data.with_columns(derivatives).drop(columns[-1])
        return data
