import logging

import polars as pl

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class ZeroOrderHoldMatching(Transform):
    def __init__(self, output_shift_seconds: float = 0.0) -> None:
        """Initialize the transform.

        Args:
            output_shift_seconds: Time in seconds to shift the output column
                (isDelocalized_value) back in time. Positive values shift it
                later in time (inputs predict future).
                Default is 0 (no shift).
        """
        self.output_shift_seconds = output_shift_seconds

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("Applying ZeroOrderHoldMatching transform")

        collected_data = data.collect()
        if collected_data.height == 0:
            return collected_data.lazy()  # Return empty if no data

        # Convert shift from seconds to nanoseconds
        shift_ns = int(
            self.output_shift_seconds * 1_000_000_000,
        )  # seconds to nanoseconds

        # List to store processed DataFrames for each slice
        processed_slices = []

        # Step 1: Process each row (slice) separately
        for slice_idx, row in enumerate(collected_data.rows(named=True)):
            row_df = pl.DataFrame([row])

            # Explode each column and extract time and value
            exploded_dfs = []
            for col in row_df.columns:
                temp_df = (
                    row_df.select(col)
                    .explode(col)
                    .unnest(col)
                    .with_columns(
                        pl.col("time").cast(pl.Int64),
                    )
                )
                exploded_df = temp_df.rename(
                    {
                        c: f"{col}_value"
                        for c in temp_df.columns
                        if c != "time"
                    },
                )
                exploded_dfs.append(exploded_df)

            # Step 2: Get all unique timestamps for this slice
            all_times = (
                pl.concat(
                    [
                        df.select(pl.col("time").cast(pl.Int64))
                        for df in exploded_dfs
                    ],
                )
                .unique()
                .sort("time")
            )

            # Step 3: Forward fill values to match all timestamps
            result = all_times
            output_df = None  # To store the shifted isDelocalized_value
            for exploded_df in exploded_dfs:
                col_name = next(c for c in exploded_df.columns if c != "time")
                if col_name == "isDelocalized_value" and shift_ns != 0:
                    # Shift output forward by subtracting shift_ns from time
                    output_df = exploded_df.with_columns(
                        (pl.col("time") - shift_ns).alias("shifted_time"),
                    ).select(
                        pl.col("shifted_time").alias("time"),
                        pl.col("isDelocalized_value"),
                    )
                else:
                    # Join and forward fill other columns
                    result = result.join(
                        exploded_df,
                        on="time",
                        how="left",
                    ).with_columns(
                        pl.col(col_name).forward_fill(),
                    )

            # Step 4: Handle the shifted output column
            if output_df is not None:
                result = result.join(
                    output_df,
                    on="time",
                    how="left",
                ).with_columns(
                    pl.col("isDelocalized_value").forward_fill(),
                )
            elif "isDelocalized_value" not in result.columns:
                # If no shift and column wasn't processed, join it unshifted
                output_df = next(
                    df
                    for df in exploded_dfs
                    if "isDelocalized_value" in df.columns
                )
                result = result.join(
                    output_df,
                    on="time",
                    how="left",
                ).with_columns(
                    pl.col("isDelocalized_value").forward_fill(),
                )

            # Add slice identifier
            result = result.with_columns(
                pl.lit(slice_idx).alias("slice_id"),
            )
            processed_slices.append(result)

        # Step 5: Concatenate all processed slices vertically
        final_df = pl.concat(processed_slices)
        return final_df.lazy()
