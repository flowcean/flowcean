from collections.abc import Iterable

import polars as pl
from run import load_or_cache_ros_data

data = load_or_cache_ros_data()

topics = [
    "/amcl_pose",
    # "/momo/pose",
    # "/scan",
    "/map",
    # "/particle_cloud",
    # not really inputs:
    # "/delocalizations",
    # "/position_error",
    # "/heading_error",
]

data_multiple = pl.concat([data] * 2)


# def explode_timeseries_expr(columns: list[str]) -> Iterable[pl.Expr]:
#     for column in columns:
#         yield pl.col(column).explode().struct.unnest()


def explode_timeseries(df: pl.LazyFrame, column: str) -> pl.LazyFrame:
    exploded = df.select(column).with_row_index().explode(column)
    unnested = exploded.unnest(column)
    renamed = unnested.with_columns(
        pl.col("value").name.prefix_fields(f"{column}/"),
    )
    unnested_2 = renamed.unnest("value")
    return unnested_2.with_row_index(f"{column}_i")


def join_topics(df: pl.LazyFrame, topics: list[str]) -> pl.LazyFrame:
    it = iter(topics)
    topic = next(it)
    joined = explode_timeseries(df, topic).select(
        ["index", "time", f"{topic}_i"],
    )

    for topic in it:
        joined = joined.join(
            explode_timeseries(df, topic).select(
                ["index", "time", f"{topic}_i"],
            ),
            on=["index", "time"],
            how="full",
            coalesce=True,
        )

    return joined.sort(["index", "time"])


def join_topics_2(df: pl.LazyFrame, topics: list[str]) -> pl.LazyFrame:
    joined = df
    for topic in topics:
        joined = joined.join(
            explode_timeseries(data_multiple, topic).drop(["index", "time"]),
            on=f"{topic}_i",
        )

    return joined.sort(["index", "time"])


joined = join_topics(data_multiple, topics)
filled = joined.select(
    pl.all().forward_fill().over("index"),
).drop_nulls()
filled_full = join_topics_2(filled, topics)
