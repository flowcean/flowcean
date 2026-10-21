"""Sampled comparisons align rows explicitly and keep physical outputs separate."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.hybrid import compare_traces


def test_named_per_signal_scores_and_inputs_are_preserved() -> None:
    reference = pl.DataFrame(
        {
            "clock": [0, 1],
            "temperature": [1, 3],
            "pressure": [2, 4],
            "id": [0, 1],
        }
    )
    prediction = pl.DataFrame(
        {"pressure": [4, 2], "clock": [0, 1], "temperature": [0, 3]}
    )
    original_reference, original_prediction = (
        reference.clone(),
        prediction.clone(),
    )
    comparison = compare_traces(
        reference,
        prediction,
        columns=["pressure", "temperature"],
        time_column="clock",
    )
    assert comparison.columns == ("pressure", "temperature")
    assert comparison.scores["signal"].to_list() == ["pressure", "temperature"]
    np.testing.assert_allclose(comparison.scores["mae"], [2, 0.5])
    np.testing.assert_allclose(comparison.scores["rmse"], [2, np.sqrt(0.5)])
    np.testing.assert_allclose(comparison.scores["max_error"], [2, 1])
    assert comparison.total_rows == comparison.scored_rows == 2
    assert comparison.excluded_rows == 0
    assert_frame_equal(reference, original_reference)
    assert_frame_equal(prediction, original_prediction)
    assert_frame_equal(
        comparison.reference,
        reference.select("clock", "pressure", "temperature"),
    )
    # Replacing an input column does not change the retained comparison data.
    prediction.replace_column(0, pl.Series("pressure", [100, 100]))
    assert comparison.prediction["pressure"].to_list() == [4, 2]


def test_exclusion_ignores_missing_warmup_but_retains_coverage_and_data() -> (
    None
):
    reference = pl.DataFrame({"t": [0, 1, 2, 3], "x": [None, 10.0, 4.0, 6.0]})
    prediction = pl.DataFrame(
        {"t": [0, 1, 2, 3], "x": [None, np.nan, 3.0, 4.0]}
    )
    comparison = compare_traces(
        reference, prediction, columns=["x"], exclude_initial=2
    )
    assert (
        comparison.total_rows,
        comparison.excluded_rows,
        comparison.scored_rows,
    ) == (4, 2, 2)
    assert_frame_equal(comparison.prediction, prediction)
    assert comparison.scores["mae"][0] == 1.5
    assert comparison.scores["rmse"][0] == pytest.approx(np.sqrt(2.5))


def test_matching_grids_use_absolute_tolerance_and_duplicate_row_order() -> (
    None
):
    reference = pl.DataFrame({"t": [0.0, 1.0, 1.0], "x": [0, 1, 10]})
    prediction = pl.DataFrame({"t": [5e-13, 1.0, 1.0], "x": [0, 10, 1]})
    comparison = compare_traces(reference, prediction, columns=["x"])
    assert comparison.scores["mae"][0] == 6
    assert comparison.prediction["t"][0] == 5e-13
    for other in (
        prediction.with_columns(pl.col("t") + 1e-6),
        prediction.reverse(),
        prediction.head(2),
        prediction.with_columns(pl.lit(float("nan")).alias("t")),
    ):
        with pytest.raises(ValueError, match="Time grids"):
            compare_traces(reference, other, columns=["x"])


def test_large_integer_timestamps_do_not_lose_alignment_or_order() -> None:
    start = 1_700_000_000_000_000_000
    reference = pl.DataFrame({"t": [start, start + 10], "x": [0, 1]})
    comparison = compare_traces(reference, reference, columns=["x"])
    assert comparison.scores["max_error"][0] == 0
    for times in ([start + 1, start + 11], [start + 1, start]):
        prediction = pl.DataFrame({"t": times, "x": [0, 1]})
        with pytest.raises(ValueError, match="Time grids"):
            compare_traces(reference, prediction, columns=["x"])


def test_large_errors_are_stable_or_rejected_if_unrepresentable() -> None:
    reference = pl.DataFrame({"t": [0, 1], "x": [1e200, 1e200]})
    prediction = reference.with_columns(pl.lit(0.0).alias("x"))
    with np.errstate(over="raise", invalid="raise"):
        comparison = compare_traces(reference, prediction, columns=["x"])
    np.testing.assert_allclose(
        comparison.scores.select("mae", "rmse", "max_error").to_numpy(),
        [[1e200, 1e200, 1e200]],
    )
    reference = pl.DataFrame({"t": [0, 1], "x": [1e308, 0.0]})
    prediction = pl.DataFrame({"t": [0, 1], "x": [-1e308, 0.0]})
    with (
        np.errstate(over="raise", invalid="raise"),
        pytest.raises(ValueError, match=r"signal 'x' at row 0.*Float64"),
    ):
        compare_traces(reference, prediction, columns=["x"])


def test_explicit_columns_must_exist_and_be_distinct_from_time() -> None:
    frame = pl.DataFrame({"t": [0.0], "named": [1.0]})
    for columns in ([], ["missing"], ["named", "named"], ["t"]):
        with pytest.raises(ValueError, match="columns"):
            compare_traces(frame, frame, columns=columns)


def test_comparison_requires_scored_rows() -> None:
    frame = pl.DataFrame({"t": [0.0], "x": [1.0]})
    with pytest.raises(ValueError, match="scored row"):
        compare_traces(frame.clear(), frame.clear(), columns=["x"])
    with pytest.raises(ValueError, match="scored row"):
        compare_traces(frame, frame, columns=["x"], exclude_initial=1)
    with pytest.raises(ValueError, match="nonnegative integer"):
        compare_traces(frame, frame, columns=["x"], exclude_initial=-1)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "1"])
def test_scored_observations_must_be_numeric_and_finite(value) -> None:
    reference = pl.DataFrame({"t": [0.0], "x": [1.0]})
    invalid = pl.DataFrame({"t": [0.0], "x": [value]})
    for left, right in ((reference, invalid), (invalid, reference)):
        with pytest.raises(ValueError, match="numeric and finite"):
            compare_traces(left, right, columns=["x"])
