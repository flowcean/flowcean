import polars as pl

from flowcean.core import Transform


class ZeroOrderHoldMatching(Transform):
    def __init__(self, columns=None, all_times=None):
        self.columns = columns
        self.all_times = all_times  # Precomputed timestamps

    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        if self.columns is None:
            self.columns = data.collect_schema().names()

        # Use provided all_times or compute from this batch
        if self.all_times is None:
            exploded_dfs = [
                data.select(col)
                .lazy()
                .explode(col)
                .unnest(col)
                .with_columns(pl.col("time").cast(pl.Int64))
                for col in self.columns
            ]
            all_times = (
                pl.concat(
                    [df.select("time") for df in exploded_dfs],
                    how="vertical",
                )
                .unique()
                .sort("time")
            )
        else:
            all_times = self.all_times

        # Process this batch’s columns with the full timestamp set
        result = all_times
        for col in self.columns:
            exploded_df = (
                data.select(col)
                .lazy()
                .explode(col)
                .unnest(col)
                .with_columns(pl.col("time").cast(pl.Int64))
            )
            field_names = [
                c for c in exploded_df.collect_schema().names() if c != "time"
            ]
            rename_dict = {field: f"{col}_{field}" for field in field_names}
            exploded_df = exploded_df.rename(rename_dict)
            added_columns = [
                c for c in exploded_df.collect_schema().names() if c != "time"
            ]
            result = result.join(
                exploded_df,
                on="time",
                how="left",
            ).with_columns(
                [pl.col(col).forward_fill() for col in added_columns],
            )

        return result
