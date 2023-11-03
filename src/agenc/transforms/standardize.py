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

    def __init__(self, mean: dict[str, float], std: dict[str, float]):
        self.mean = mean
        self.std = std

    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.select(
            [
                (pl.col(c) - (self.mean.get(c) or 0.0))
                / (self.std.get(c) or 1.0)
                for c in data.columns
            ],
        )
