"""Shared eager/lazy prediction boundary for tabular models."""

from abc import ABC
from typing import override

import polars as pl

from flowcean.core.model import Model


class PolarsModel(Model[pl.LazyFrame, pl.LazyFrame], ABC):
    """Accept eager or lazy input, with exclusively lazy attached transforms.

    Normalization happens before preprocessing, including probability
    prediction paths that call preprocess directly. Strategies can therefore
    attach ordinary LazyFrame transforms without widening their contracts.
    """

    @override
    def preprocess(
        self, input_features: pl.DataFrame | pl.LazyFrame
    ) -> pl.LazyFrame:
        return super().preprocess(input_features.lazy())

    @override
    def predict(
        self, input_features: pl.DataFrame | pl.LazyFrame
    ) -> pl.LazyFrame:
        return super().predict(input_features.lazy())

    @override
    def __call__(
        self, input_features: pl.DataFrame | pl.LazyFrame
    ) -> pl.LazyFrame:
        return self.predict(input_features)
