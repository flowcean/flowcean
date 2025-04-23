import logging
from collections.abc import Iterable

import polars as pl

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


def zero_order_hold_align(
    df: pl.LazyFrame,
    columns: Iterable[str],
    name: str,
) -> pl.LazyFrame:
    """Perform zero-order-hold alignment of multiple time-series features.

    Args:
        df: Input DataFrame containing struct-type time-series columns.
        columns: Names of struct columns to align using zero-order-hold.
        name: Name of the output struct column.

    Returns:
        DataFrame with an additional column containing zero-order-hold aligned
        time-series.
    """
    exploded = (
        df.with_row_index()
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

    aligned = (
        pl.concat(exploded, how="align")
        .with_columns(pl.exclude("index", "time").forward_fill().over("index"))
        .drop_nulls()
        .select(
            pl.struct(
                pl.col("time"),
                pl.struct(
                    pl.exclude("index", "time"),
                ).alias("value"),
            )
            .implode()
            .over("index")
            .alias(name),
        )
    )

    return pl.concat([df, aligned], how="horizontal")


class ZeroOrderHoldMatching(Transform):
    def __init__(
        self,
        topics: list[str],
        name: str = "aligned",
    ) -> None:
        """Initialize the ZeroOrderHoldMatching transform.

        Args:
            topics: List of topics to align.
            name: Name of the output time-series column.
        """
        super().__init__()
        self.topics = topics
        self.name = name

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("Applying ZeroOrderHoldMatching transform")
        return zero_order_hold_align(
            data,
            columns=self.topics,
            name=self.name,
        )
