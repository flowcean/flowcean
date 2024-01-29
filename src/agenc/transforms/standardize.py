import polars as pl
from polars.type_aliases import PythonLiteral

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

    mean: None | dict[str, float]
    std: None | dict[str, float]

    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def fit(self, data: pl.DataFrame) -> None:
        self.mean = {c: __as_float(data[c].mean()) for c in data.columns}
        self.std = {c: __as_float(data[c].std()) for c in data.columns}

    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardize transform has not been fitted.")

        return data.select(
            [
                (pl.col(c) - (self.mean.get(c) or 0.0))
                / (self.std.get(c) or 1.0)
                for c in data.columns
            ],
        )


def __as_float(value: PythonLiteral | None) -> float:
    if value is None:
        raise ValueError("Value cannot be None.")
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    raise ValueError("Value must be a float.")
