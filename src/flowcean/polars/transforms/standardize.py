from __future__ import annotations

from dataclasses import dataclass
from typing import Self, override

import polars as pl
from polars._typing import PythonLiteral

from flowcean.core import InvertibleTransform


@dataclass
class Standardize(InvertibleTransform[pl.LazyFrame, pl.LazyFrame]):
    r"""Standardize features by removing the mean and scaling to unit variance.

    A sample $x$ is standardized as:

    $$
        z = \frac{(x - \mu)}{\sigma}
    $$

    where

    - $\mu$ is the mean of the samples
    - $\sigma$ is the standard deviation of the samples.

    Attributes:
        mean: The mean $\mu$ of each feature.
        std: The standard deviation $\sigma$
            of each feature.
    """

    mean: dict[str, float] | None = None
    std: dict[str, float] | None = None

    @override
    def fit(self, data: pl.LazyFrame) -> Self:
        df = data.collect(engine="streaming")

        self.mean = {
            c: _as_float(df[c].mean()) for c in data.collect_schema().names()
        }
        self.std = {
            c: _as_float(df[c].std()) for c in data.collect_schema().names()
        }
        self.counts = len(df)
        return self

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        if self.mean is None or self.std is None:
            message = "Standardize transform has not been fitted"
            raise RuntimeError(message)

        return data.select(
            [
                (pl.col(c) - (self.mean.get(c) or 0.0))
                / (self.std.get(c) or 1.0)
                for c in data.collect_schema().names()
            ],
        )

    @override
    def inverse(self) -> Standardize:
        if self.mean is None or self.std is None:
            message = "Standardize transform has not been fitted"
            raise RuntimeError(message)

        if self.mean.keys() != self.std.keys():
            message = (
                "Standardize mean and std must have matching feature names"
            )
            raise ValueError(message)
        return Standardize(
            mean={c: -m / (self.std[c] or 1.0) for c, m in self.mean.items()},
            std={c: 1.0 / (s or 1.0) for c, s in self.std.items()},
        )


def _as_float(value: PythonLiteral | None) -> float:
    if value is None:
        message = "value cannot be None"
        raise ValueError(message)
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    message = "value must be a float"
    raise ValueError(message)
