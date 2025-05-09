import polars as pl
from typing_extensions import override

from flowcean.core.transform import Transform


class ShiftTimestamps(Transform):
    """Shift the timestamps of a series features."""

    def __init__(self, shift: float, feature: str) -> None:
        """Initialize the transform.

        Args:
            shift: Time in seconds to add (positive values) or subtract
                (negative values) to the timestamps of the feature.
            feature: The name of the feature to shift the timestamps for.
        """
        super().__init__()
        self.shift = int(
            shift * 1_000_000_000,
        )
        self.feature = feature

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        # Convert shift from seconds to nanoseconds
        collected_data = data.collect()

        if len(collected_data) == 0:
            return collected_data.lazy()

        # Initialize with the first transformed row
        transformed_data = self._transform_row(collected_data.slice(0, 1))

        # Process the remaining rows
        for i in range(1, len(collected_data)):
            transformed_data_slice = self._transform_row(
                collected_data.slice(i, 1),
            )
            transformed_data = transformed_data.vstack(transformed_data_slice)
        return transformed_data.lazy()

    def _transform_row(self, data: pl.DataFrame) -> pl.DataFrame:
        shifted = (
            data.select(self.feature)
            .explode(self.feature)
            .unnest(self.feature)
            .with_columns(pl.col("time").cast(pl.Int64) + self.shift)
        )
        # Nest into a struct
        nested = shifted.select(
            [
                pl.struct(
                    pl.col("time"),
                    pl.col("value"),
                )
                .implode()
                .alias(self.feature),
            ],
        )
        # Add back to the original DataFrame
        return pl.concat(
            [data.drop(self.feature), nested],
            how="horizontal",
        )
