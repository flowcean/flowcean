import typing

import polars as pl

from agenc.core import Transform


class Standardize(Transform):
    r"""Standardize features by removing the mean and scaling to unit variance.

    A sample :math:`x` is standardized as:

    .. math::
        z = \frac{(x - \mu)}{\sigma}

    where
        :math:`\mu` is the mean of the samples
        :math:`\sigma` is the standard deviation of the samples.

    Args:
        mean: The mean :math:`\mu` of each feature.
        std: The standard deviation :math:`\sigma` of each feature.
    """

    mean: None | dict[str, typing.Any]
    std: None | dict[str, typing.Any]

    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def fit(self, data: pl.DataFrame) -> None:
        self.mean = {c: data[c].mean() for c in data.columns}
        self.std = {c: data[c].std() for c in data.columns}

    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardize transform has not been fitted.")

        print(self.mean)
        print(self.std)

        return data.select(
            [
                (pl.col(c) - (self.mean.get(c) or 0.0))
                / (self.std.get(c) or 1.0)
                for c in data.columns
            ],
        )
