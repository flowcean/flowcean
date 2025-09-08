import polars as pl
from polars._typing import ColumnNameOrSelector
from typing_extensions import override

from flowcean.core import Transform


class ExplodeTimeSeries(Transform):
    """Transform that explodes nested time series data into individual rows.

    Each time series is represented as a list of structs, where each struct
    contains a ``timestamp`` and a ``value``. The ``value`` is itself a struct
    holding multiple feature values at that timestamp. This transform expands
    the list into separate rows, then unnests the nested structs into columns.

    Example:
        Input DataFrame:
            ┌─────┬────────────────────────────────────────┐
            │ id  │ series                                 │
            │ --- │ ---                                    │
            │ i64 │ list[struct[timestamp: str, value]]    │
            ├─────┼────────────────────────────────────────┤
            │ 1   │ [{t1, {a=1, b=10}}, {t2, {a=2, b=20}}] │
            │ 2   │ [{t3, {a=3, b=30}}, {t4, {a=4, b=40}}] │
            └─────┴────────────────────────────────────────┘

        After applying ``ExplodeTimeSeries("series")``:
            ┌─────┬────────────┬─────┬─────┐
            │ id  │ timestamp  │  a  │  b  │
            │ --- │ ---        │ --- │ --- │
            │ i64 │ str        │ i64 │ i64 │
            ├─────┼────────────┼─────┼─────┤
            │ 1   │ t1         │  1  │ 10  │
            │ 1   │ t2         │  2  │ 20  │
            │ 2   │ t3         │  3  │ 30  │
            │ 2   │ t4         │  4  │ 40  │
            └─────┴────────────┴─────┴─────┘
    """

    features: ColumnNameOrSelector

    def __init__(self, features: ColumnNameOrSelector) -> None:
        super().__init__()
        self.features = features

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return (
            data.explode(self.features).unnest(self.features).unnest("value")
        )
