import polars as pl
from run import load_or_cache_ros_data

data = load_or_cache_ros_data()

topics = [
    "/amcl_pose",
    "/momo/pose",
    "/scan",
    "/particle_cloud",
    # "/map",
    # "/delocalizations",
    # "/position_error",
    # "/heading_error",
]

data_multiple = pl.concat([data] * 2)


def explode_timeseries(df: pl.LazyFrame, column: str) -> pl.LazyFrame:
    exploded = df.select(column).with_row_index().explode(column)
    unnested = exploded.unnest(column)
    renamed = unnested.with_columns(
        pl.col("value").name.map_fields(lambda x: f"{column}/{x}"),
    )
    return renamed.unnest("value")


def join_topics(df: pl.LazyFrame, topics: list[str]) -> pl.LazyFrame:
    it = iter(topics)
    joined = explode_timeseries(df, next(it))

    for topic in it:
        joined = joined.join(
            explode_timeseries(df, topic),
            on=["index", "time"],
            how="full",
            coalesce=True,
        )

    return joined.sort(["index", "time"])


joined = join_topics(data_multiple, topics)
filled = joined.select(
    pl.all().forward_fill().over("index"),
).drop_nulls()
