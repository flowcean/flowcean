import polars as pl

pl.Config().set_tbl_rows(1_000)

df = pl.DataFrame(
    {
        "foo": [
            [
                {"time": 0.0, "value": 1},
                {"time": 1.0, "value": 2},
                {"time": 2.5, "value": 3},
                {"time": 3.0, "value": 4},
                {"time": 4.5, "value": 5},
                {"time": 5.0, "value": 6},
                {"time": 6.0, "value": 7},
            ],
            [
                {"time": 0.0, "value": 4},
                {"time": 3.5, "value": 5},
                {"time": 6.0, "value": 6},
            ],
        ],
        "bar": [17, 42],
    },
).lazy()
print(df.collect())
df = (
    df.select(pl.row_index("_index"), pl.col("foo"))
    .explode("foo")
    .unnest("foo")
    # .with_columns(pl.col("time").cast(pl.Duration("ms")))
)
print(df.collect(engine="streaming"))

d = (
        df.select(pl.arange(

df = (
    df.collect(engine="streaming")
    .upsample(
        time_column="time",
        every="2ms",
        group_by="_index",
        maintain_order=True,
    )
    .lazy()
)
print(df.collect(engine="streaming"))

df = (
    df.select(
        pl.col("_index").forward_fill(),
        pl.struct(
            pl.col("time"),
            pl.col("value").interpolate(),
        ),
    )
    .group_by("_index", maintain_order=True)
    .agg(pl.all().implode())
)
print(df.collect(engine="streaming"))
