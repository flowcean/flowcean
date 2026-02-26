import logging
from collections.abc import Iterable

import polars as pl

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


def resample_to_reference(
    data: pl.LazyFrame,
    features: Iterable[str],
    reference: str,
    name: str,
) -> pl.LazyFrame:
    """Resample columns to a reference timeline holding the last value.

    Args:
        data: Input LazyFrame containing time series columns.
        features: Features to resample. The reference column will be included
            in the output automatically; other features will be resampled to
            match the reference timeline.
        reference: Column whose timestamps define the output timeline.
        name: Name for the output aligned time series column.

    Returns:
        LazyFrame with a single time series column containing resampled data.
    """
    # Extract reference timeline
    ref = (
        data.with_row_index()
        .explode(reference)
        .select(
            pl.col("index"),
            pl.col(reference).struct.field("time").alias("time"),
        )
    )

    # For each feature: explode → forward-fill onto reference timeline
    aligned_cols = []
    for col in features:
        # Skip resampling the reference to itself
        if col == reference:
            continue

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

        # Join to reference timeline (as-of join, stratified by index)
        df = ref.join_asof(
            df.sort("index", "t"),
            left_on="time",
            right_on="t",
            by="index",
        ).sort("index")

        # Forward-fill the values along the reference sampling times
        df = df.with_columns(pl.exclude("index", "time", "t").forward_fill())

        aligned_cols.append(df)

    # Start with reference timeline and add reference column values
    ref_df = (
        data.with_row_index()
        .explode(reference)
        .select(
            pl.col("index"),
            pl.col(reference).struct.field("time").alias("time"),
            pl.col(reference)
            .struct.field("value")
            .name.prefix_fields(f"{reference}/")
            .struct.unnest(),
        )
    )

    # Concatenate all aligned columns horizontally
    final = ref_df
    for df in aligned_cols:
        feature_cols = [
            c
            for c in df.collect_schema().names()
            if c not in ["index", "time", "t"]
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


class ResampleToReference(Transform):
    """Resamples time series features to a reference timeline.

    This transform takes multiple time series features and resamples them all
    to match the sampling times of a specified reference feature. Values are
    forward-filled (holding the last value) to the reference timestamps.

    Example:
        If reference has timestamps [0, 5, 10] and another feature has
        timestamps [0, 3, 7, 12], the output will have timestamps [0, 5, 10]
        with the other feature's values forward-filled to those times.
    """

    def __init__(
        self,
        reference: str,
        features: list[str] | None = None,
        name: str = "resampled",
        *,
        drop: bool = True,
    ) -> None:
        """Initialize the ResampleToReference transform.

        Args:
            reference: Column whose timestamps define the output timeline.
                This column will be included in the output automatically.
            features: List of time series features to resample. If None, uses
                all features in the data. The reference column is always
                included in the output regardless.
            name: Name of the output time series feature.
            drop: Whether to drop the original features after resampling.
        """
        super().__init__()
        self.reference = reference
        self.features = features
        self.name = name
        self.drop = drop

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug(
            "Resampling features %s to reference column '%s'",
            self.features,
            self.reference,
        )

        if self.features is None:
            logger.info(
                "No features specified for resampling; using all features",
            )
            features = data.collect_schema().names()
        else:
            features = list(self.features)
            # Add reference if not already in features
            if self.reference not in features:
                features.append(self.reference)

        resampled = resample_to_reference(
            data,
            features=features,
            reference=self.reference,
            name=self.name,
        )

        if self.drop:
            data = data.drop(features)

        return pl.concat([data, resampled], how="horizontal")
