import polars as pl

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


def slice_time_series(feature: pl.Expr, slices: pl.Expr) -> pl.Expr:
    def is_between(element: pl.Expr, reference: pl.Expr) -> pl.Expr:
        return element.filter(
            element.struct.field("time").is_between(
                reference.struct.field("from"),
                reference.struct.field("to"),
                closed="none",
            ),
        )

    return pl.concat_list(
        pl.struct(
            feature.alias("feature"),
            slices.alias("slices"),
        ),
    ).list.eval(
        is_between(
            pl.element().struct.field("feature").explode(),
            pl.element().struct.field("slices"),
        ),
    )


after = (
    df.sample(1, with_replacement=True)
    .explode("slices")
    .with_columns(slice_time_series(pl.col("a"), pl.col("slices")).alias("a"))
)
