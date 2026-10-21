from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import override

import polars as pl

from flowcean.core import Metric


class PolarsMetric[Result](
    Metric[pl.DataFrame | pl.LazyFrame, pl.DataFrame, Result],
    ABC,
):
    """Prepare selected tabular columns for an eager metric computation."""

    def __init__(self, *, features: Sequence[str] | None = None) -> None:
        self.features = features

    @override
    def prepare(self, data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
        if self.features is not None:
            data = data.select(self.features)
        if isinstance(data, pl.LazyFrame):
            return data.collect(engine="streaming")
        return data
