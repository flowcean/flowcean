import polars as pl
from typing_extensions import override

from flowcean.core import Transform


class ExplodeTimeSeries(Transform):
    feature: str | dict[str, str]

    def __init__(self, feature: str) -> None:
        super().__init__()
        self.feature = feature

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.explode(self.feature).unnest(self.feature).unnest("value")
