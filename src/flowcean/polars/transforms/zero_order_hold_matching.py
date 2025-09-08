import logging
from collections.abc import Iterable

import polars as pl

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


def zero_order_hold_align(
    data: pl.LazyFrame,
    columns: Iterable[str],
    name: str,
) -> pl.LazyFrame:
    """Perform zero-order-hold alignment of multiple time series features.

    Args:
        data: Input DataFrame containing struct-type time series columns.
        columns: Names of struct columns to align using zero-order-hold.
        name: Name of the output struct column.

    Returns:
        zero-order-hold aligned time series
    """
    exploded = (
        data.with_row_index()
        .explode(column)
        .select(
            pl.col("index"),
            pl.col(column).struct.field("time"),
            pl.col(column)
            .struct.field("value")
            .name.prefix_fields(f"{column}/")
            .struct.unnest(),
        )
        for column in columns
    )

    return (
        pl.concat(exploded, how="align")
        .with_columns(pl.exclude("index", "time").forward_fill().over("index"))
        .drop_nulls()
        .select(
            pl.col("index"),
            pl.struct(
                pl.col("time"),
                pl.struct(
                    pl.exclude("index", "time"),
                ).alias("value"),
            ).alias(name),
        )
        .group_by("index", maintain_order=True)
        .agg(pl.all().implode())
        .drop("index")
    )


class ZeroOrderHold(Transform):
    """Aligns multiple time series features using zero-order-hold."""

    def __init__(
        self,
        features: list[str],
        name: str = "aligned",
        *,
        drop: bool = True,
    ) -> None:
        """Initialize the ZeroOrderHoldMatching transform.

        Args:
            features: List of topics to align.
            name: Name of the output time series feature.
            drop: Whether to drop the original features after alignment.
        """
        super().__init__()
        self.features = features
        self.name = name
        self.drop = drop

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug(
            "Aligning features %s using zero-order-hold",
            self.features,
        )
        aligned = zero_order_hold_align(
            data,
            columns=self.features,
            name=self.name,
        )
        if self.drop:
            data = data.drop(self.features)
        return pl.concat([data, aligned], how="horizontal")
