from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self, override

import polars as pl
from polars._typing import PythonLiteral

from flowcean.core import InvertibleTransform
from flowcean.polars import is_timeseries_feature


@dataclass
class ScaleToRange(InvertibleTransform[pl.LazyFrame, pl.LazyFrame]):
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
        _validate_range(lower_range, upper_range)
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

        fitted = self.from_limits(
            {
                feature: (
                    _as_float(min_max_values[f"{feature}_min"]),
                    _as_float(min_max_values[f"{feature}_max"]),
                )
                for feature in target_features
            },
            lower_range=self.lower_range,
            upper_range=self.upper_range,
        )
        self.m, self.b = fitted.m, fitted.b
        return self

    def _parameters(self) -> tuple[dict[str, float], dict[str, float]]:
        if self.m is None or self.b is None:
            message = "ScaleToRange transform has not been fitted"
            raise RuntimeError(message)
        if self.m.keys() != self.b.keys():
            message = (
                "ScaleToRange coefficients must have matching feature names"
            )
            raise ValueError(message)
        if any(
            not math.isfinite(value) or value == 0 for value in self.m.values()
        ):
            message = "ScaleToRange requires finite nonzero scaling factors"
            raise ValueError(message)
        if any(not math.isfinite(value) for value in self.b.values()):
            message = "ScaleToRange requires finite offsets"
            raise ValueError(message)
        return self.m, self.b

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        m, b = self._parameters()
        return data.with_columns([pl.col(c) * m[c] + b[c] for c in m])

    @override
    def inverse(self) -> ScaleToRange:
        m, b = self._parameters()
        inversed_scale = ScaleToRange(
            features=self.features,
            lower_range=self.lower_range,
            upper_range=self.upper_range,
        )
        inversed_scale.m = {
            feature: 1.0 / value for feature, value in m.items()
        }
        inversed_scale.b = {
            feature: -value / m[feature] for feature, value in b.items()
        }
        inversed_scale._parameters()
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

        for feature, (minimum, maximum) in feature_limits.items():
            if (
                not math.isfinite(minimum)
                or not math.isfinite(maximum)
                or minimum >= maximum
            ):
                message = f"Feature {feature!r} must have finite, strictly increasing limits"
                raise ValueError(message)
        transform.m = {
            feature: (upper_range - lower_range) / (maximum - minimum)
            for feature, (minimum, maximum) in feature_limits.items()
        }
        transform.b = {
            feature: lower_range - minimum * transform.m[feature]
            for feature, (minimum, _) in feature_limits.items()
        }
        transform._parameters()
        return transform


def _validate_range(lower: float, upper: float) -> None:
    if not math.isfinite(lower) or not math.isfinite(upper) or lower == upper:
        message = "Output range bounds must be finite and distinct"
        raise ValueError(message)


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
