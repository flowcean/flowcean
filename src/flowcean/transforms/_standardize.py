from typing import override

import polars as pl
from polars.type_aliases import PythonLiteral

from flowcean.core.transform import FitOnce, Transform


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
        counts: Number of samples already learned
    """

    mean: dict[str, float] | None = None
    std: dict[str, float] | None = None
    counts: int | None = None

    def __init__(self) -> None:
        super().__init__()

    @override
    def fit(self, data: pl.DataFrame) -> None:
        self.mean = {c: _as_float(data[c].mean()) for c in data.columns}
        self.std = {c: _as_float(data[c].std()) for c in data.columns}
        self.counts = len(data)

    @override
    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        if self.mean is None or self.std is None:
            message = "Standardize transform has not been fitted"
            raise RuntimeError(message)

        return data.select(
            [
                (pl.col(c) - (self.mean.get(c) or 0.0))
                / (self.std.get(c) or 1.0)
                for c in data.columns
            ],
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
