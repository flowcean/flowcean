from collections.abc import Callable

import polars as pl
from typing_extensions import override

from flowcean.core.transform import Transform


def collect_slices(
    counter: pl.Expr,
    condition: pl.Expr,
    name: str = "slices",
) -> pl.Expr:
    """Convert a list of timestamped events into `from`-`to` slice ranges.

    This function constructs a list of structs, where each struct represents a
    time slice between two event timestamps. The beginning of the first slice
    is always 0, and each subsequent slice starts where the previous ended.

    Example:
        Given a list of timestamps: [5, 9, 15]
        The output will be:
            [
                {"from": 0, "to": 5},
                {"from": 5, "to": 9},
                {"from": 9, "to": 15}
            ]

    Args:
        counter: A list of structs, each containing at least a `time` field.
        condition: A boolean expression applied to filter the events.
        name: The desired name for the resulting slice column.

    Returns:
        Expression that generates the list of `from`-`to` slice structs.
    """
    event_times = counter.list.eval(
        pl.element().struct.field("time").filter(condition),
    )

    start_times = event_times.list.shift(n=1).list.eval(
        pl.element().fill_null(0),
    )
    end_times = event_times

    return (
        pl.concat_list(
            pl.struct(
                start_times.alias("from"),
                end_times.alias("to"),
            ),
        )
        .list.eval(
            pl.struct(
                pl.element().struct.field("from").explode(),
                pl.element().struct.field("to").explode(),
            ),
        )
        .alias(name)
    )


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


class SliceTimeSeries(Transform):
    """Slices time series features based on a counter column and deadzone."""

    def __init__(self, time_series: str, counter_feature: str) -> None:
        super().__init__()
        self.time_series = time_series
        self.counter_feature = counter_feature

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        return data.with_columns(
            slice_time_series(
                pl.col(self.time_series),
                collect_slices(
                    pl.col(self.counter_feature),
                    condition=pl.element()
                    .struct.field("value")
                    .struct.field("data")
                    .diff()
                    > 0,
                ),
            ).alias(
                self.time_series,
            ),
        ).explode(self.time_series)
