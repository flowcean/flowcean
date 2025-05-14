from typing import Iterable
import polars as pl

N = 100
df = pl.DataFrame(
    {
        "a": [
            [
                {"time": i * 2, "value": {"data": i * 10}}
                for i in range(1, N // 2)
            ],
        ],
        "b": [
            [{"time": i * 3, "value": {"data": i * 1}} for i in range(N // 3)],
        ],
        "c": [
            [
                {"time": i * 7, "value": {"foo": i * 1, "bar": i / 2}}
                for i in range(10, N // 7)
            ],
        ],
        "map": [True],
        "event_counter": [
            [
                {"time": 2, "value": {"data": 0}},
                {"time": 3, "value": {"data": 0}},
                {"time": 4, "value": {"data": 0}},
                {"time": 10, "value": {"data": 1}},
                {"time": 11, "value": {"data": 1}},
                {"time": 20, "value": {"data": 2}},
                {"time": 22, "value": {"data": 2}},
                {"time": 36, "value": {"data": 3}},
                {"time": 40, "value": {"data": 3}},
                {"time": 58, "value": {"data": 4}},
                {"time": 63, "value": {"data": 4}},
                {"time": 75, "value": {"data": 5}},
                {"time": 84, "value": {"data": 5}},
                {"time": 100, "value": {"data": 6}},
            ],
        ],
    },
)
print(df)

# zoh = ZeroOrderHoldMatching(["a", "b"])
# after = zoh(df)


pl.Config().set_tbl_rows(100)

# df = df.sample(3, with_replacement=True)
df = df.lazy()
after = df.select(
    pl.col("event_counter")
    .list.eval(
        pl.element().struct.with_fields(
            pl.field("value").struct.field("data").diff().alias("value"),
        ),
    )
    .list.eval(
        pl.element()
        .struct.field("time")
        .filter(pl.element().struct.field("value") > 0),
    ),
).collect()
print(after)


def zero_order_hold_align(
    df: pl.LazyFrame,
    columns: Iterable[str],
    new_column: str = "aligned",
) -> pl.LazyFrame:
    """Perform zero-order-hold alignment of multiple time-series features.

    Args:
        df: Input DataFrame containing struct-type time-series columns.
        columns: Names of struct columns to align using zero-order-hold.
        new_column: Name of the output struct column. Defaults to "aligned".

    Returns:
        DataFrame with an additional column containing zero-order-hold aligned
        time-series.
    """
    exploded = (
        df.with_row_index()
        .explode(column)
        .select(
            pl.col("index"),
            pl.col(column).struct.field("time"),
            pl.col(column)
            .struct.field("value")
            .name.prefix_fields(f"{column}/")
            .struct.unnest(),
        )
        for column in columns
    )

    aligned = (
        pl.concat(exploded, how="align")
        .with_columns(pl.exclude("index", "time").forward_fill().over("index"))
        .drop_nulls()
        .select(
            pl.struct(
                pl.col("time"),
                pl.struct(
                    pl.exclude("index", "time"),
                ).alias("value"),
            )
            .implode()
            .over("index")
            .alias(new_column),
        )
    )

    return pl.concat([df, aligned], how="horizontal")


# after = zero_order_hold_align(
#     df,
#     columns=["a", "b", "c"],
#     new_column="aligned",
# ).collect(engine="streaming")
# print(after)


# columns = ["a", "b", "c"]
# name = "aligned"
#
#
# def prefix_values(expr: pl.Expr, prefix: str) -> pl.Expr:
#     return expr.struct.with_fields(
#         pl.field("value").name.prefix_fields(f"{prefix}/"),
#     )
#
#
# after = pl.concat(
#     (
#         df.with_row_index()
#         .explode(column)
#         .select(
#             pl.col("index"),
#             prefix_values(pl.col(column), column).struct.field("time"),
#             prefix_values(pl.col(column), column)
#             .struct.field("value")
#             .struct.unnest(),
#         )
#         for column in columns
#     ),
#     how="align",
# ).select(
#     pl.struct(
#         pl.col("time"),
#         pl.struct(pl.exclude("index", "time")).alias("value"),
#     )
#     .implode()
#     .over("index")
#     .alias(name),
# )
#
# full = pl.concat([df, after], how="horizontal")
#
# collected = full.collect(engine="streaming")
# print(collected.schema)
# print(collected)

# after = (
#     df
#     # .sample(3, with_replacement=True)
#     .with_columns(
#         slice_time_series(
#             pl.col("a"),
#             collect_slices(
#                 pl.col("event_counter"),
#                 condition=pl.element()
#                 .struct.field("value")
#                 .struct.field("data")
#                 .diff()
#                 > 0,
#             ),
#         ).alias(
#             "a",
#         ),
#     )
#     .drop("event_counter")
#     .explode("a")
# )
# print(after)
