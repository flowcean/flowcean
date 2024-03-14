import polars as pl
from polars.type_aliases import PythonLiteral
from typing_extensions import override

from agenc.core import Transform, UnsupervisedLearner


class Standardize(Transform, UnsupervisedLearner):
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

    mean: None | dict[str, float] = None
    std: None | dict[str, float] = None
    counts: None | int = None

    @override
    def fit(self, data: pl.DataFrame) -> None:
        self.mean = {c: _as_float(data[c].mean()) for c in data.columns}
        self.std = {c: _as_float(data[c].std()) for c in data.columns}
        self.counts = len(data)

    @override
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
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
