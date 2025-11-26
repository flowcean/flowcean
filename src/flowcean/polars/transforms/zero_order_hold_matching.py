import logging
from collections.abc import Iterable

import polars as pl

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


def zero_order_hold_align(
    data: pl.LazyFrame,
    columns: Iterable[str],
    name: str,
    reference_column: str | None = None,
) -> pl.LazyFrame:
    """Zero-order hold alignment of struct time series columns.

    If reference_column is provided:
        • Only generate timestamps from the reference column
        • Forward-fill all other columns to match only those timestamps

    If reference_column is None:
        • Behaves like the original implementation (full alignment)
    """
    # ---------------------------------------------------------------
    # CASE 1 – No reference column → use original behavior
    # ---------------------------------------------------------------
    if reference_column is None:
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
            .with_columns(
                pl.exclude("index", "time").forward_fill().over("index"),
            )
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

    # ---------------------------------------------------------------
    # CASE 2 – Reference-based ZOH alignment
    # ---------------------------------------------------------------

    if reference_column not in columns:
        msg = f"reference_column {reference_column} not in columns"
        raise ValueError(
            msg,
        )

    # Extract reference timeline
    ref = (
        data.with_row_index()
        .explode(reference_column)
        .select(
            pl.col("index"),
            pl.col(reference_column).struct.field("time").alias("time"),
        )
    )

    # For each column: explode → forward-fill onto ref.index → take latest value
    aligned_cols = []
    for col in columns:
        df = (
            data.with_row_index()
            .explode(col)
            .select(
                pl.col("index"),
                pl.col(col).struct.field("time").alias("t"),
                pl.col(col)
                .struct.field("value")
                .name.prefix_fields(f"{col}/")
                .struct.unnest(),
            )
        )

        # Join to reference timeline (as-of join)
        # Drop the 'index' column from df before joining to avoid duplicates
        df = ref.join_asof(
            df.sort("t").drop("index"),
            left_on="time",
            right_on="t",
        ).sort("index")

        # Forward-fill the values only along the reference sampling times
        df = df.with_columns(pl.exclude("index", "time", "t").forward_fill())

        aligned_cols.append(df)

    # Combine all columns horizontally
    # We need to be careful here - the first dataframe has 'index' and 'time'
    # Subsequent ones should only contribute their feature columns
    final = aligned_cols[0]
    for df in aligned_cols[1:]:
        # Only take the feature columns (exclude 'index', 'time', 't')
        feature_cols = [
            c for c in df.columns if c not in ["index", "time", "t"]
        ]
        final = pl.concat([final, df.select(feature_cols)], how="horizontal")

    # Group by index and aggregate into time series
    return (
        final.select(
            pl.col("index"),
            pl.struct(
                pl.col("time"),
                pl.struct(pl.exclude("index", "time", "t")).alias("value"),
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
        features: list[str] | None = None,
        name: str = "aligned",
        *,
        drop: bool = True,
        reference_column: str | None = None,
    ) -> None:
        """Initialize the ZeroOrderHoldMatching transform.

        Args:
            features: List of topics to align.
            name: Name of the output time series feature.
            drop: Whether to drop the original features after alignment.
            reference_column: Column whose timestamps drive the alignment.
        """
        super().__init__()
        self.features = features
        self.name = name
        self.drop = drop
        self.reference_column = reference_column

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug(
            "Aligning features %s using zero-order-hold (reference=%s)",
            self.features,
            self.reference_column,
        )
        if self.features is None:  # take all features in data
            logger.info(
                "No features specified for ZOH alignment; using all features",
            )
            self.features = data.columns
        aligned = zero_order_hold_align(
            data,
            columns=self.features,
            name=self.name,
            reference_column=self.reference_column,
        )
        if self.drop:
            data = data.drop(self.features)
        return pl.concat([data, aligned], how="horizontal")
