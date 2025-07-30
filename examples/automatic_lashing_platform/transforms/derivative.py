import polars as pl
from typing_extensions import override

from flowcean.core.transform import Transform


class Derivative(Transform):
    """Calculate the derivative applied to a flatten LazyFrame."""

    def __init__(self, column: str) -> None:
        self.column = column

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        columns = [
            col
            for col in data.collect_schema().names()
            if col.startswith(self.column)
        ]
        if columns:
            derivatives = [
                (
                    pl.lit(0)
                    if i == 0
                    else (pl.col(columns[i]) - pl.col(columns[i - 1]))
                ).alias(columns[i])
                for i in range(len(columns))
            ]
            return data.with_columns(derivatives)
        return data
