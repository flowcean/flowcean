from collections.abc import Callable

import polars as pl


def slice_time_series(
    time_series: pl.Expr,
    slices: pl.Expr,
) -> pl.Expr:
    """Slice a list-of-structs time series into multiple segments.

    For each struct in `slices` (with fields `from` and `to`), this function
    extracts all samples from `time_series` (structs with `time` and `value`)
    whose `time` lies in `[from, to)`.

    Args:
        time_series: Expression of a list column containing structs with fields
            `time` and `value`.
        slices: Expression of a list column containing structs with fields
            `from` and `to`.

    Returns:
        A list-of-lists expression: for each row, an outer list over slices,
        each element is the sub-series (a list of structs) for that slice.
    """

    def _list_eval(
        list_col: pl.Expr | str,
        func: Callable[..., pl.Expr],
        *refs: pl.Expr | str,
    ) -> pl.Expr:
        """Evaluate `func` on elements of a list column, with references."""
        elements = pl.element().struct[0].explode()
        references = [pl.element().struct[i + 1] for i in range(len(refs))]
        return pl.concat_list(pl.struct(list_col, *refs)).list.eval(
            func(elements, *references),
        )

    def _filter_between(
        sample: pl.Expr,
        lower: pl.Expr,
        upper: pl.Expr,
    ) -> pl.Expr:
        return sample.filter(
            sample.struct.field("time").is_between(
                lower,
                upper,
                closed="left",
            ),
        )

    return _list_eval(
        slices,
        lambda slice_range, series: _list_eval(
            series,
            lambda sample, slice_range: _filter_between(
                sample,
                slice_range.struct.field("from"),
                slice_range.struct.field("to"),
            ),
            slice_range,
        ),
        time_series,
    )


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
df = df.with_columns(
    a_sliced=slice_time_series(pl.col("a"), pl.col("slices")).alias(
        "a_sliced",
    ),
)
