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
        derivative_suffix: str = "_derivative",
    ) -> None:
        """Initializes the DiscreteDerivative transform.

        Args:
            features: Features that shall be differentiated. Result features
                will be named `<feature>_derivative`.
            method: Method to use for calculating the derivative. Valid options
                are "forward", "backward", and "central".
                Defaults to "central".
            derivative_suffix: Suffix to append to the feature name for the
                resulting derivative feature. Defaults to "_derivative".
        """
        self.features = [features] if isinstance(features, str) else features
        self.method = method
        self.derivative_suffix = derivative_suffix

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        for feature in self.features:
            value_feature = f"_{feature}_value"
            time_feature = f"_{feature}_t"
            dt_feature = f"_{feature}_dt"
            dvalue_feature = f"{feature}{self.derivative_suffix}"
            index_feature = "_index"

            working_df = (
                data.select(
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
                working_df = working_df.with_columns(
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
                working_df = working_df.with_columns(
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
            working_df = working_df.drop_nulls().with_columns(
                (pl.col(value_feature) / pl.col(dt_feature)).alias(
                    dvalue_feature,
                ),
            )

            # Collapse the data back to time series format
            working_df = working_df.group_by(pl.col(index_feature)).agg(
                pl.struct(
                    pl.col(time_feature).alias("time"),
                    pl.col(dvalue_feature).alias("value"),
                )
                .implode()
                .alias(dvalue_feature),
            )

            # Join the newly created derivative feature back to the
            # original data
            data = pl.concat(
                [data, working_df.drop(index_feature)],
                how="horizontal",
            )

        return data
