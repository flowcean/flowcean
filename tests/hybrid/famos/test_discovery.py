"""Geometric segmentation, exact DTW, and ordered location discovery."""

import numpy as np
import polars as pl
import pytest

from flowcean.hybrid.famos import discover_famos_locations
from flowcean.hybrid.famos._dtw import (
    _alignment_score,
    ordered_clusters,
    segment_score,
)
from flowcean.hybrid.famos.discovery import _geometric_peaks, _merge_peaks


def test_geometry_retains_rows_warmup_and_short_unassigned_trace():
    t = np.arange(61, dtype=float)
    y = np.where(t <= 30, t, 30 - 0.5 * (t - 30))
    traces = [
        pl.DataFrame(
            {"t": t, "y": y, "u": np.zeros(len(t)), "truth": [99] * len(t)}
        ),
        pl.DataFrame(
            {"t": np.arange(5, dtype=float), "y": [-2.0] * 5, "u": [0.0] * 5}
        ),
    ]
    native_traces = [
        trace.with_columns(pl.lit(99).alias("location_id")) for trace in traces
    ]
    discovery = discover_famos_locations(
        native_traces,
        input_features=["u"],
        output_features=["y"],
        window=4,
        trim=2,
        peak_height=0.1,
        rho_min=0.001,
        rho_max=0.01,
    )
    labeled = discovery.to_labeled_frames(traces)
    for original, annotated in zip(traces, labeled, strict=True):
        assert annotated.drop("location_id").equals(original)
    with pytest.raises(ValueError, match=r"location_id.*already exists"):
        discovery.to_labeled_frames(native_traces)
    assert labeled[0]["location_id"][0] is None
    assert labeled[1]["location_id"].null_count() == 5
    assigned = [
        segment
        for segment in discovery.segments
        if segment.trace_index == 0 and segment.location_id is not None
    ]
    assert len(assigned) == 2
    assert abs(assigned[0].stop - 30) <= 1
    assert assigned[0].location_id != assigned[1].location_id
    np.testing.assert_array_equal(discovery.output_scales, [30.0])
    np.testing.assert_array_equal(discovery.input_scales, [1.0])
    assert _merge_peaks([10, 16, 22], 8) == [10, 22]


def test_derivative_recursion_preserves_outer_intervals():
    # No raw peaks: derivative scanning retains the complete interval.
    # A slope change near its start would be lost by repeated endpoint insets.
    t = np.arange(30, dtype=float)
    derivative = np.minimum(t, 5)
    peaks = _geometric_peaks([np.zeros(30), derivative], 4, 0.5)
    assert peaks == [5]


def test_exact_dtw_cost_and_raw_alignment_offsets():
    a, b = np.array([0.0, 3.0, 10.0]), np.array([0.0, 4.0, 10.0])
    assert _alignment_score(a, b) == pytest.approx(1 / 6)
    # Distinguish absolute local costs from root-sum-squared costs.
    assert _alignment_score(np.zeros(4), np.full(4, 0.1)) == pytest.approx(
        0.05
    )
    # Zero-cost non-diagonal path; score must include path-index correlation.
    correlation = np.corrcoef([0, 1, 2, 2], [0, 0, 1, 2])[0, 1]
    assert _alignment_score(
        np.array([0.0, 0.0, 2.0]), np.array([0.0, 2.0, 2.0])
    ) == pytest.approx(0.5 * (1 - correlation))
    assert segment_score(a, np.r_[a + 7, 100], raw=True) == pytest.approx(0)
    assert segment_score(a, a + 7, raw=False) > 0
    assert segment_score(np.zeros(4), np.zeros(4), raw=False) == pytest.approx(
        0
    )


def test_ordered_asymmetric_reassignment_is_not_cluster_union():
    scores = np.array(
        [
            [0, 0.8, 0.2, 0.4],
            [0.8, 0, 0.1, 0.14],
            [0.2, 0.1, 0, 0.12],
            [0.4, 0.14, 0.12, 0],
        ]
    )
    # Row 0 accepts 2, even though row 2's threshold would reject 0.
    # Row 1 then moves 2 alone on better confidence, leaving 0 behind.
    # Assigned self-pairs must not reset confidence to zero.
    assert ordered_clusters(scores, alpha=1.5, rho_min=0.01, rho_max=0.5) == [
        0,
        1,
        1,
        1,
    ]
