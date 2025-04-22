import polars as pl


def slice_time_series_join(
    df: pl.DataFrame,
    time_series_col: str,
    slices_col: str,
    result_col: str | None = None,
) -> pl.DataFrame:
    """Slice a nested time series by exploding and joining ranges.

    This approach:
      1. Adds a row identifier to preserve original rows.
      2. Explodes the `slices` list into individual ranges with a `slice_id`.
      3. Explodes the `time_series` list into individual samples with `time` and `value`.
      4. Cross-joins samples and ranges on `row_id`, then filters samples by range bounds.
      5. Groups back by row and slice, collecting samples into sub-series.
      6. Re-nests sub-series lists in the original row order.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with two list columns: one with structs `{'time','value'}`,
        the other with structs `{'from','to'}`.
    time_series_col : str
        Name of the column containing the list<struct<time,value>>.
    slices_col : str
        Name of the column containing the list<struct<from,to>>.
    result_col : str | None
        Optional name for the output nested column. If None, defaults to
        `{time_series_col}_sliced`.

    Returns
    -------
    pl.DataFrame
        A DataFrame with all original columns plus a new nested column of sub-series.
    """
    # 1) label rows
    df_id = df.with_row_index("row_id")

    # 2) explode ranges and tag with slice index
    ranges = (
        df_id.explode(slices_col)
        .with_columns(
            slice_id=pl.cum_count(),
            start=pl.col(slices_col).struct.field("from"),
            end=pl.col(slices_col).struct.field("to"),
        )
        .select("row_id", "slice_id", "start", "end")
    )

    # 3) explode time series samples
    samples = (
        df_id.explode(time_series_col)
        .with_columns(
            time=pl.col(time_series_col).struct.field("time"),
            value=pl.col(time_series_col).struct.field("value"),
        )
        .select("row_id", "time", "value")
    )

    # 4) cross-join and filter by time bounds
    joined = samples.join(ranges, on="row_id", how="cross").filter(
        (pl.col("time") >= pl.col("start")) & (pl.col("time") < pl.col("end"))
    )

    # 5) collect back into sub-series lists
    collected = (
        joined.select(
            "row_id", "slice_id", pl.struct(["time", "value"]).alias("pair")
        )
        .groupby(["row_id", "slice_id"])
        .agg(pl.col("pair").list().alias("subseries"))
        .sort(["row_id", "slice_id"])
        .groupby("row_id")
        .agg(pl.col("subseries").list())
    )

    # 6) join result back to original and rename
    out_col = result_col or f"{time_series_col}_sliced"
    result = (
        df_id.join(collected, on="row_id")
        .drop("row_id")
        .with_columns(pl.col("subseries").alias(out_col))
        .drop("subseries")
    )

    return result


df = pl.DataFrame(
    {
        "a": [[{"time": i * 2, "value": i * 10} for i in range(100_000)]],
        "slices": [
            [
                {"from": 10, "to": 20},
                {"from": 20, "to": 36},
                {"from": 36, "to": 58},
                {"from": 58, "to": 75},
                {"from": 75, "to": 100},
            ],
        ],
        "map": [True],
    },
)

df_sliced = slice_time_series_join(df, "a", "slices")
