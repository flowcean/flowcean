from collections.abc import Callable

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


def slice_time_series(time_series: pl.Expr, slices: pl.Expr) -> pl.Expr:
    def list_eval_with(
        listcol: pl.Expr | str,
        operation: Callable[..., pl.Expr],
        *ref_cols: str | pl.Expr,
    ) -> pl.Expr:
        arguments = [pl.element().struct[0].explode()] + [
            pl.element().struct[i + 1] for i in range(len(ref_cols))
        ]
        return pl.concat_list(pl.struct(listcol, *ref_cols)).list.eval(
            operation(*arguments),
        )

    def is_between(
        element: pl.Expr,
        lower: pl.Expr,
        upper: pl.Expr,
    ) -> pl.Expr:
        return element.filter(
            element.struct.field("time").is_between(
                lower,
                upper,
                closed="left",
            ),
        )

    return list_eval_with(
        slices,
        lambda from_to, time_series: list_eval_with(
            time_series,
            lambda sample, from_to: is_between(
                sample,
                from_to.struct.field("from"),
                from_to.struct.field("to"),
            ),
            from_to,
        ),
        time_series,
    )


after = df.sample(2, with_replacement=True).with_columns(
    slice_time_series(pl.col("a"), pl.col("slices")).alias("a"),
)
