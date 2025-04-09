import logging

import polars as pl

from flowcean.core.transform import Transform

logger = logging.getLogger(__name__)


class ZeroOrderHoldMatching(Transform):
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("Applying ZeroOrderHoldMatching transform")

        collected_data = data.collect()
        # Check if there's more than one row
        if collected_data.height == 0:
            return collected_data.lazy()  # Return empty if no data

        # List to store processed DataFrames for each recording
        processed_recordings = []

        # Step 1: Process each row (recording) separately
        for row_idx, row in enumerate(collected_data.rows(named=True)):
            # Convert the row to a single-row DataFrame
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

            # Step 2: Get all unique timestamps for this recording
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
            for exploded_df in exploded_dfs:
                col_name = [c for c in exploded_df.columns if c != "time"][0]
                result = result.join(
                    exploded_df,
                    on="time",
                    how="left",
                ).with_columns(
                    pl.col(col_name).forward_fill(),
                )

            # Add a recording identifier
            result = result.with_columns(
                pl.lit(row_idx).alias("recording_id"),
            )
            processed_recordings.append(result)

        # Step 4: Concatenate all processed recordings vertically
        final_df = pl.concat(processed_recordings)
        return final_df.lazy()
