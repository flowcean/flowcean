from typing import Callable
import polars as pl
from sympy import field

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


def list_eval_ref(
    listcol: pl.Expr | str,
    op: Callable[..., pl.Expr],
    *ref_cols: str | pl.Expr,
) -> pl.Expr:
    # if len(ref_cols) == 0:
    #     ref_cols = tuple([x for x in signature(op).parameters.keys()][1:])

    args_to_op = [pl.element().struct[0].explode()] + [
        pl.element().struct[i + 1] for i in range(len(ref_cols))
    ]
    return pl.concat_list(pl.struct(listcol, *ref_cols)).list.eval(
        op(*args_to_op),
    )


def slice_time_series(feature: pl.Expr, slices: pl.Expr) -> pl.Expr:
    def is_between(
        element: pl.Expr,
        lower: pl.Expr,
        upper: pl.Expr,
    ) -> pl.Expr:
        return element.filter(
            element.struct.field("time").is_between(
                lower,
                upper,
                closed="none",
            ),
        )

    # return slices.list.eval(pl.struct(pl.element()))
    return (
        list_eval_ref(
            pl.col("slices"),
            lambda slice, a: slice.struct.with_fields(a.alias("feature")),
            pl.col("a"),
        ).list.eval(
            list_eval_ref(
                pl.element().struct.field("feature"),
                lambda feature, f, t: is_between(feature, f, t),
                pl.element().struct.field("from"),
                pl.element().struct.field("to"),
            ),
        )
        # .list.eval(
        #     pl.struct(
        #         pl.element().struct.field("feature").explode(),
        #         pl.element().struct.field("").alias("slices"),
        #     ),
        # )
        # .list.eval(
        #     pl.element().struct.field("feature").list.eval(
        #     is_between(
        #         pl.element().struct.field("feature").explode(),
        #         pl.element().struct.field("slices"),
        #     ),
        # )
    )
    # return pl.concat_list(
    #     pl.struct(
    #         feature.alias("feature"),
    #         slices.alias("slices"),
    #     ),
    # ).list.eval(
    #     is_between(
    #         pl.element().struct.field("feature").explode(),
    #         pl.element().struct.field("slices"),
    #     ),
    # )


after = df.sample(2, with_replacement=True).select(
    slice_time_series(pl.col("a"), pl.col("slices")).alias("a"),
)
