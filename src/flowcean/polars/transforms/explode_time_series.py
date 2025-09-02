import polars as pl
from polars._typing import ColumnNameOrSelector
from typing_extensions import override

from flowcean.core import Transform


class ExplodeTimeSeries(Transform):
    features: ColumnNameOrSelector

    def __init__(self, features: ColumnNameOrSelector) -> None:
        super().__init__()
        self.features = features

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return (
            data.explode(self.features).unnest(self.features).unnest("value")
        )
