from dataclasses import dataclass

import polars as pl
from polars._typing import PythonLiteral
from typing_extensions import override

from flowcean.core import FitOnce, Transform


@dataclass
class Standardize(Transform, FitOnce):
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
    def fit(self, data: pl.LazyFrame) -> None:
        df = data.collect(streaming=True)

        self.mean = {
            c: _as_float(df[c].mean()) for c in data.collect_schema().names()
        }
        self.std = {
            c: _as_float(df[c].std()) for c in data.collect_schema().names()
        }
        self.counts = len(df)

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
    def inverse(self) -> Transform:
        if self.mean is None or self.std is None:
            message = "Standardize transform has not been fitted"
            raise RuntimeError(message)

        return Standardize(
            mean={
                c: -m * s
                for c, m, s in zip(
                    self.mean,
                    self.mean.values(),
                    self.std.values(),
                    strict=True,
                )
            },
            std={c: 1.0 / s for c, s in self.std.items()},
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
