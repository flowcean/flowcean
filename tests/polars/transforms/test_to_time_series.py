import unittest

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.core import Transform
from flowcean.polars import (
    DataFrame,
    ExplodeTimeSeries,
    JoinedOfflineEnvironment,
    MatchSamplingRate,
    Mean,
    Median,
    Mode,
    Resample,
    Select,
    SignalFilter,
    TimeWindow,
    ToTimeSeries,
    is_timeseries_feature,
)
from flowcean.polars.time_series_type import (
    get_time_series_time_type,
    get_time_series_value_type,
)


class TestToTimeSeries(unittest.TestCase):
    def test_multivariate_time_series(self) -> None:
        dataset = DataFrame(
            pl.DataFrame(
                {
                    "time_feature": pl.Series([0, 1], dtype=pl.Int32),
                    "feature_a": [42, 43],
                    "feature_b": [3.5, None],
                    "label": ["first", "second"],
                },
            ).lazy(),
        )

        result = (dataset | ToTimeSeries("time_feature")).observe().collect()

        assert_frame_equal(
            result,
            pl.DataFrame(
                {
                    "time_series": [
                        [
                            {
                                "time": 0,
                                "value": {
                                    "feature_a": 42,
                                    "feature_b": 3.5,
                                    "label": "first",
                                },
                            },
                            {
                                "time": 1,
                                "value": {
                                    "feature_a": 43,
                                    "feature_b": None,
                                    "label": "second",
                                },
                            },
                        ],
                    ],
                },
                schema={
                    "time_series": pl.List(
                        pl.Struct(
                            {
                                "time": pl.Int32,
                                "value": pl.Struct(
                                    {
                                        "feature_a": pl.Int64,
                                        "feature_b": pl.Float64,
                                        "label": pl.String,
                                    },
                                ),
                            },
                        ),
                    ),
                },
            ),
        )

    def test_custom_name_and_single_value_remains_scalar(self) -> None:
        result = ToTimeSeries("t", name="signal")(
            pl.LazyFrame(
                {"t": [0, 1], "reading": [42, None]},
                schema={"t": pl.Int64, "reading": pl.Int32},
            ),
        ).collect()

        assert_frame_equal(
            result,
            pl.DataFrame(
                {
                    "signal": [
                        [
                            {"time": 0, "value": 42},
                            {"time": 1, "value": None},
                        ],
                    ],
                },
                schema={
                    "signal": pl.List(
                        pl.Struct({"time": pl.Int64, "value": pl.Int32}),
                    ),
                },
            ),
        )

    def test_preserves_unsorted_and_duplicate_timestamps(self) -> None:
        data = pl.LazyFrame(
            {"time": [2, 0, 2, 1], "value": [42, 43, 44, 45]},
        )
        result = ToTimeSeries("time")(data)
        assert_frame_equal(
            result.explode("time_series").unnest("time_series").collect(),
            data.collect(),
        )

    def test_multiple_clocks_are_composed_externally(self) -> None:
        def source() -> DataFrame:
            return DataFrame(
                pl.LazyFrame(
                    {
                        "clock_a": [0, 1],
                        "a": [10, 20],
                        "clock_b": [5, 7],
                        "b": [30, 40],
                        "c": [50, 60],
                    },
                ),
            )

        combined = JoinedOfflineEnvironment(
            (
                source()
                | Select(["clock_a", "a"])
                | ToTimeSeries("clock_a", name="series_a"),
                source()
                | Select(["clock_b", "b", "c"])
                | ToTimeSeries("clock_b", name="series_b"),
            ),
        )

        assert_frame_equal(
            combined.observe().collect(),
            pl.DataFrame(
                {
                    "series_a": [
                        [
                            {"time": 0, "value": 10},
                            {"time": 1, "value": 20},
                        ],
                    ],
                    "series_b": [
                        [
                            {"time": 5, "value": {"b": 30, "c": 50}},
                            {"time": 7, "value": {"b": 40, "c": 60}},
                        ],
                    ],
                },
            ),
        )

    def test_empty_input_preserves_schema(self) -> None:
        result = ToTimeSeries("t")(
            pl.LazyFrame(schema={"t": pl.Float64, "reading": pl.Int32}),
        ).collect()
        assert_frame_equal(
            result,
            pl.DataFrame(
                {"time_series": [[]]},
                schema={
                    "time_series": pl.List(
                        pl.Struct(
                            {
                                "time": pl.Float64,
                                "value": pl.Int32,
                            },
                        ),
                    ),
                },
            ),
        )

    def test_empty_multivariate_input_preserves_schema(self) -> None:
        result = ToTimeSeries("t")(
            pl.LazyFrame(
                schema={"t": pl.Float64, "a": pl.Int32, "b": pl.String},
            ),
        ).collect()
        assert_frame_equal(
            result,
            pl.DataFrame(
                {"time_series": [[]]},
                schema={
                    "time_series": pl.List(
                        pl.Struct(
                            {
                                "time": pl.Float64,
                                "value": pl.Struct(
                                    {"a": pl.Int32, "b": pl.String},
                                ),
                            },
                        ),
                    ),
                },
            ),
        )

    def test_time_only_input(self) -> None:
        result = ToTimeSeries("t")(pl.LazyFrame({"t": [0, 1]})).collect()
        assert_frame_equal(
            result,
            pl.DataFrame(
                {
                    "time_series": [
                        [
                            {"time": 0, "value": {}},
                            {"time": 1, "value": {}},
                        ],
                    ],
                },
            ),
        )

    def test_empty_time_only_input(self) -> None:
        result = ToTimeSeries("t")(
            pl.LazyFrame(schema={"t": pl.Int32}),
        ).collect()
        assert_frame_equal(
            result,
            pl.DataFrame(
                {"time_series": [[]]},
                schema={
                    "time_series": pl.List(
                        pl.Struct({"time": pl.Int32, "value": pl.Struct({})}),
                    ),
                },
            ),
        )

    def test_missing_time_column(self) -> None:
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            ToTimeSeries("missing")(pl.LazyFrame({"value": [1]})).collect()

    def test_rejects_mapping(self) -> None:
        with pytest.raises(TypeError, match="single column name"):
            ToTimeSeries({"value": "t"})  # type: ignore[arg-type]

    def test_struct_aware_helpers_and_transforms(self) -> None:
        data = pl.LazyFrame(
            {"t": [0, 1, 2], "a": [10, 20, 30], "b": [1, 2, 3]},
        )
        series = ToTimeSeries("t")(data)
        schema = series.collect_schema()
        assert is_timeseries_feature(schema, "time_series")
        assert get_time_series_value_type(schema["time_series"]) == pl.Struct(
            {"a": pl.Int64, "b": pl.Int64},
        )
        assert get_time_series_time_type(schema["time_series"]) == pl.Int64
        window = TimeWindow(time_start=1, time_end=2)(series)
        assert_frame_equal(
            ExplodeTimeSeries("time_series")(window).collect(),
            data.filter(pl.col("t") >= 1).rename({"t": "time"}).collect(),
        )

    def test_match_sampling_rate_with_multivariate_values(self) -> None:
        reference = ToTimeSeries("t", name="reference")(
            pl.LazyFrame({"t": [0, 1, 2], "r": [0, 1, 2]}),
        )
        signal = ToTimeSeries("t", name="signal")(
            pl.LazyFrame({"t": [0, 2], "a": [0.0, 4.0], "b": [2.0, 6.0]}),
        )
        result = MatchSamplingRate("reference", {"signal": "linear"})(
            pl.concat([reference, signal], how="horizontal_extend"),
        )
        assert_frame_equal(
            result.select("signal").collect(),
            ToTimeSeries("t", name="signal")(
                pl.LazyFrame(
                    {
                        "t": [0, 1, 2],
                        "a": [0.0, 2.0, 4.0],
                        "b": [2.0, 4.0, 6.0],
                    },
                ),
            ).collect(),
        )


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        (Mean("signal", replace=True), 4.0),
        (Median("signal", replace=True), 2.0),
        (Mode("signal", replace=True), 2),
    ],
)
def test_singleton_composes_with_statistics(
    transform: Transform,
    expected: float,
) -> None:
    data = DataFrame(
        pl.LazyFrame({"t": [0, 1, 2, 3, 4], "reading": [1, 2, 2, 3, 12]}),
    )
    result = (
        (data | ToTimeSeries("t", name="signal") | transform)
        .observe()
        .collect()
    )
    assert_frame_equal(result, pl.DataFrame({"signal": [expected]}))


def test_singleton_composes_with_resample() -> None:
    data = DataFrame(
        pl.LazyFrame({"t": [0.0, 1.0, 2.0], "reading": [0.0, 2.0, 4.0]}),
    )
    result = (
        (data | ToTimeSeries("t", name="signal") | Resample(0.5))
        .observe()
        .collect()
    )
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "signal": [
                    [
                        {"time": 0.0, "value": 0.0},
                        {"time": 0.5, "value": 1.0},
                        {"time": 1.0, "value": 2.0},
                        {"time": 1.5, "value": 3.0},
                        {"time": 2.0, "value": 4.0},
                    ],
                ],
            },
        ),
    )


def test_singleton_composes_with_signal_filter() -> None:
    times = [i / 100 for i in range(100)]
    data = DataFrame(pl.LazyFrame({"t": times, "reading": [1.0] * 100}))
    result = (
        (
            data
            | ToTimeSeries("t", name="signal")
            | SignalFilter(["signal"], "lowpass", 25, order=1)
        )
        .observe()
        .collect()
    )
    # At one quarter of the sampling frequency, a first-order Butterworth
    # lowpass averages adjacent samples (with zero initial state).
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "signal": [
                    [
                        {"time": t, "value": 0.5 if i == 0 else 1.0}
                        for i, t in enumerate(times)
                    ],
                ],
            },
        ),
    )


if __name__ == "__main__":
    unittest.main()
