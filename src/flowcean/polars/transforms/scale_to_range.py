from dataclasses import dataclass

import polars as pl
from polars._typing import PythonLiteral
from typing_extensions import Self, override

from flowcean.core import Data, Invertible, Transform
from flowcean.polars import is_timeseries_feature


@dataclass
class ScaleToRange(Invertible, Transform):
    r"""Scale features to a fixed range using a linear mapping.

    A sample $x$ is scaled as:

    $$
        z = x \cdot m + b
    $$

    where

    - $m$ is the scaling factor
    - $b$ is the offset.

    When instantiating this transform directly, the scaling factor $m$ and
    offset $b$ for each feature are calculated during training from the data.
    To specify the scaling factor $m$ and offset $b$ directly, use the
    `from_limits` method.

    Attributes:
        m: The scaling factor $m$ of each feature.
        b: The offset $b$ of each feature.
    """

    features: list[str] | None = None
    m: dict[str, float] | None = None
    b: dict[str, float] | None = None

    lower_range: float = -1.0
    upper_range: float = 1.0

    def __init__(
        self,
        *,
        features: list[str] | None = None,
        lower_range: float = -1.0,
        upper_range: float = 1.0,
    ) -> None:
        self.features = features
        self.lower_range = lower_range
        self.upper_range = upper_range

    @override
    def fit(self, data: pl.LazyFrame) -> Self:
        schema = data.collect_schema()
        target_features = self.features or [
            name
            for name in schema.names()
            if not is_timeseries_feature(schema, name)
        ]

        # Get the min and max values of the features
        min_max_values = data.select(
            [pl.col(c).min().alias(f"{c}_min") for c in target_features]
            + [pl.col(c).max().alias(f"{c}_max") for c in target_features],
        ).collect(engine="streaming")

        # Calculate the scaling factor and offset for each feature
        self.m = {
            feature: _as_float(
                (self.upper_range - self.lower_range)
                / (
                    min_max_values[f"{feature}_max"]
                    - min_max_values[f"{feature}_min"]
                ),
            )
            for feature in target_features
        }

        self.b = {
            feature: -_as_float(
                min_max_values[f"{feature}_min"] * self.m[feature]
                - self.lower_range,
            )
            for feature in target_features
        }
        return self

    @override
    def fit_incremental(self, data: Data) -> Self:
        msg = "Incremental fitting is not supported for ScaleToRange transform"
        raise NotImplementedError(msg)

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        if self.m is None or self.b is None:
            message = "ScaleToRange transform has not been fitted"
            raise RuntimeError(message)

        return data.with_columns(
            [pl.col(c) * self.m.get(c) + self.b.get(c) for c in self.m],
        )

    @override
    def inverse(self) -> Transform:
        if self.m is None or self.b is None:
            message = "ScaleToRange transform has not been fitted"
            raise RuntimeError(message)
        inversed_scale = ScaleToRange(
            features=self.features,
            lower_range=self.lower_range,
            upper_range=self.upper_range,
        )
        inversed_scale.m = {feature: 1.0 / m for feature, m in self.m.items()}
        inversed_scale.b = {
            feature: -b / self.m[feature] for feature, b in self.b.items()
        }

        return inversed_scale

    @classmethod
    def from_limits(
        cls,
        feature_limits: dict[str, tuple[float, float]],
        *,
        lower_range: float = -1.0,
        upper_range: float = 1.0,
    ) -> Self:
        """Creates a new ScaleToRange transform based on the given limits.

        Args:
            feature_limits: A dictionary mapping each features name to its
                (min_value, max_value) tuple.
            lower_range: The lower bound of the range to scale to.
            upper_range: The upper bound of the range to scale to.

        """
        transform = cls(
            features=list(feature_limits.keys()),
            lower_range=lower_range,
            upper_range=upper_range,
        )

        transform.m = {
            feature: (upper_range - lower_range) / (max_value - min_value)
            for feature, (max_value, min_value) in feature_limits.items()
        }

        transform.b = {
            feature: min_value
            * (upper_range - lower_range)
            / (max_value - min_value)
            + lower_range
            for feature, (max_value, min_value) in feature_limits.items()
        }

        return transform


def _as_float(value: PythonLiteral | pl.Series | None) -> float:
    if isinstance(value, pl.Series):
        value = value.item()
    if value is None:
        message = "value cannot be None"
        raise ValueError(message)
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    message = "value must be a float"
    raise ValueError(message)
