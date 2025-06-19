import logging
from collections.abc import Iterable
from typing import Literal, TypeAlias

import polars as pl
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)

DiscreteDerivativeKind: TypeAlias = Literal["forward", "backward", "central"]


class DiscreteDerivative(Transform):
    """Calculates the discrete derivative of time series features.

    Calculates the discrete derivative of time series features using either
    forward, backward, or central differences.
    """

    def __init__(
        self,
        features: str | Iterable[str],
        *,
        method: DiscreteDerivativeKind = "central",
    ) -> None:
        """Initializes the DiscreteDerivative transform.

        Args:
            features: Features that shall be differentiated. Result features
                will be named `<feature>_derivative`.
            method: Method to use for calculating the derivative. Valid options
                are "forward", "backward", and "central".
                Defaults to "central".
        """
        self.features = [features] if isinstance(features, str) else features
        self.method = method

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        feature_names = data.collect_schema().names()
        for feature in self.features:
            value_feature = f"{feature}_value"
            time_feature = f"{feature}_t"
            dt_feature = f"{feature}_dt"
            dvalue_feature = f"{feature}_derivative"
            index_feature = "_index"

            data = (
                data.with_columns(
                    [
                        pl.col(feature)
                        .list.eval(pl.element().struct.field("value"))
                        .alias(value_feature),
                        pl.col(feature)
                        .list.eval(pl.element().struct.field("time"))
                        .alias(time_feature),
                    ],
                )
                .with_row_index(name=index_feature)
                .explode([value_feature, time_feature])
            )
            if self.method in {"forward", "backward"}:
                shift = -1 if self.method == "forward" else 1
                data = data.with_columns(
                    [
                        pl.col(value_feature)
                        .diff(n=shift)
                        .over(index_feature),
                        pl.col(time_feature)
                        .diff(n=shift)
                        .alias(dt_feature)
                        .over(index_feature),
                    ],
                )
            elif self.method == "central":
                data = data.with_columns(
                    [
                        pl.col(value_feature).shift(-1).over(index_feature)
                        - pl.col(value_feature).shift(1).over(index_feature),
                        (
                            pl.col(time_feature).shift(-1).over(index_feature)
                            - pl.col(time_feature).shift(1).over(index_feature)
                        ).alias(dt_feature),
                    ],
                )
            else:
                logger.warning("Unknown derivative method %s", self.method)

            # Calculate the derivative
            data = data.drop_nulls().with_columns(
                (pl.col(value_feature) / pl.col(dt_feature)).alias(
                    dvalue_feature,
                ),
            )

            # Collapse the data back to time series format
            data = data.group_by(pl.col(index_feature)).agg(
                pl.struct(
                    pl.col(time_feature).alias("time"),
                    pl.col(dvalue_feature).alias("value"),
                )
                .implode()
                .alias(dvalue_feature),
                # Retain the original feature values
                *[pl.col(feature).first() for feature in feature_names],
            )

            # Drop the index feature. The other features are implicitly
            # dropped by the group_by operation.
            data = data.drop(index_feature)

        return data
