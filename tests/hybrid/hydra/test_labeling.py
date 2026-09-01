"""Behavioral tests for HyDRA pending-segment grouping."""

from typing import override

import numpy as np
import polars as pl

from flowcean.core import Model
from flowcean.hybrid.hydra.learner import (
    HyDRATrace,
    TraceSegment,
    find_next_pending_segment,
    label_matching_rows,
)


class ZeroDerivativeModel(Model):
    """Small deterministic flow model used to exercise grouping."""

    @override
    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        return pl.DataFrame({"dx": [0.0] * frame.height}).lazy()


def _trace(dx: list[float], labels: list[int]) -> HyDRATrace:
    frame = pl.DataFrame(
        {
            "time": [float(index) for index in range(len(dx))],
            "x": [10.0 + index for index in range(len(dx))],
            "dx": dx,
        },
    )
    return HyDRATrace(frame, np.asarray(labels, dtype=np.int64))


def test_find_next_pending_segment_scans_contiguous_runs_and_traces() -> None:
    traces = [
        _trace([0.0, 0.0], [3, 3]),
        _trace([0.0, 0.0, 0.0, 0.0], [2, -1, -1, 2]),
        _trace([0.0], [-1]),
    ]

    assert find_next_pending_segment(traces) == TraceSegment(1, 1, 2)
    assert (
        find_next_pending_segment(
            [
                trace.with_mode_labels(np.zeros(trace.height, dtype=np.int64))
                for trace in traces
            ],
        )
        is None
    )


def test_label_matching_rows_labels_only_accurate_unlabeled_rows() -> None:
    traces = [
        _trace([0.0, 0.1, 9.0, 0.0], [-1, -1, 4, -1]),
        _trace([0.2, 0.19], [-1, -1]),
    ]
    triggering_segment = TraceSegment(0, 0, 1)

    result = label_matching_rows(
        traces=traces,
        model=ZeroDerivativeModel(),
        input_columns=["time", "x"],
        output_columns=["dx"],
        threshold=0.2,
        mode_id=5,
        triggering_segment=triggering_segment,
    )

    # Accuracy is strict: the row with error exactly equal to the threshold
    # remains pending, and an existing label is never overwritten.
    assert result.traces[0].mode_labels.tolist() == [5, 5, 4, 5]
    assert result.traces[1].mode_labels.tolist() == [-1, 5]
    assert result.accepted_rows["mode"].to_list() == [5, 5, 5, 5]
    assert result.accepted_rows["dx"].to_list() == [0.0, 0.1, 0.0, 0.19]

    grouping = result.grouping
    assert grouping.mode_id == 5
    assert grouping.triggering_segment == triggering_segment
    assert grouping.traces[0].row_indices == [0, 1, 3]
    assert grouping.traces[0].accepted_mask == [True, True, True]
    assert grouping.traces[1].accepted_mask == [False, True]
    assert grouping.traces[1].errors == [0.2, 0.19]
