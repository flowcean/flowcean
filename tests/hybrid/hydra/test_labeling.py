"""Behavioral tests for interval-based HyDRA grouping and subtraction."""

from itertools import product
from typing import override

import numpy as np
import polars as pl
import pytest

from flowcean.core import Model
from flowcean.hybrid.hydra.learner import (
    TraceSegment,
    _first_pending_segment,
    _group_matching_segments,
    _segments_from_mask,
    _subtract_segments,
)


class ZeroDerivativeModel(Model):
    @override
    def _predict(
        self, input_features: pl.DataFrame | pl.LazyFrame
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        return pl.DataFrame({"dx": [0.0] * frame.height}).lazy()


def _trace(dx: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "time": [float(index) for index in range(len(dx))],
            "x": [10.0 + index for index in range(len(dx))],
            "dx": dx,
        },
    )


def test_first_pending_segment_uses_trace_and_row_order() -> None:
    pending = [
        [],
        [TraceSegment(1, 1, 3), TraceSegment(1, 5, 6)],
        [TraceSegment(2, 0, 1)],
    ]
    assert _first_pending_segment(pending) == TraceSegment(1, 1, 3)
    assert _first_pending_segment([[], [], []]) is None
    assert _first_pending_segment([]) is None


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ([], []),
        ([False], []),
        ([True], [TraceSegment(2, 7, 8)]),
        ([False, True, True], [TraceSegment(2, 8, 10)]),
        ([True, True, False], [TraceSegment(2, 7, 9)]),
        (
            [True, False, True, True, False, True],
            [
                TraceSegment(2, 7, 8),
                TraceSegment(2, 9, 11),
                TraceSegment(2, 12, 13),
            ],
        ),
    ],
)
def test_mask_runs_are_maximal_absolute_half_open_segments(
    mask: list[bool], expected: list[TraceSegment]
) -> None:
    assert (
        _segments_from_mask(np.asarray(mask), trace_index=2, start=7)
        == expected
    )


@pytest.mark.parametrize(
    ("accepted", "expected"),
    [
        ([], [TraceSegment(0, 2, 8)]),
        ([TraceSegment(0, 2, 4)], [TraceSegment(0, 4, 8)]),
        ([TraceSegment(0, 6, 8)], [TraceSegment(0, 2, 6)]),
        (
            [TraceSegment(0, 4, 6)],
            [TraceSegment(0, 2, 4), TraceSegment(0, 6, 8)],
        ),
        ([TraceSegment(0, 2, 8)], []),
        (
            [TraceSegment(0, 3, 4), TraceSegment(0, 4, 7)],
            [TraceSegment(0, 2, 3), TraceSegment(0, 7, 8)],
        ),
    ],
)
def test_subtraction_prefix_suffix_interior_full_empty_and_adjacency(
    accepted: list[TraceSegment], expected: list[TraceSegment]
) -> None:
    pending = [[TraceSegment(0, 2, 8)]]
    assert _subtract_segments(pending, accepted) == [expected]
    assert pending == [[TraceSegment(0, 2, 8)]]


def test_subtraction_preserves_gaps_and_fully_assigned_traces() -> None:
    pending = [
        [],
        [TraceSegment(1, 0, 1), TraceSegment(1, 3, 6)],
        [TraceSegment(2, 4, 5)],
    ]
    accepted = [
        TraceSegment(1, 0, 1),
        TraceSegment(1, 4, 5),
        TraceSegment(2, 4, 5),
    ]
    assert _subtract_segments(pending, accepted) == [
        [],
        [TraceSegment(1, 3, 4), TraceSegment(1, 5, 6)],
        [],
    ]


@pytest.mark.parametrize(
    "accepted",
    [
        [TraceSegment(-1, 0, 1)],
        [TraceSegment(2, 0, 1)],
        [TraceSegment(0, 0, 1)],
        [TraceSegment(1, 0, 1)],
        [TraceSegment(1, 1, 4)],
        [TraceSegment(1, 2, 5)],
        [TraceSegment(1, 3, 4)],
        [TraceSegment(1, 6, 8)],
        [TraceSegment(1, 8, 9)],
        [TraceSegment(1, 2, 2)],
        [TraceSegment(1, 3, 2)],
        [TraceSegment(1, 1, 3), TraceSegment(1, 2, 3)],
        [TraceSegment(1, 4, 5), TraceSegment(1, 1, 2)],
    ],
)
def test_subtraction_rejects_invalid_and_outside_pending_pieces(
    accepted: list[TraceSegment],
) -> None:
    pending = [[], [TraceSegment(1, 1, 3), TraceSegment(1, 4, 7)]]
    with pytest.raises(ValueError, match="Accepted segment"):
        _subtract_segments(pending, accepted)


def test_subtraction_exhaustive_short_tristate_masks() -> None:
    # 0 = outside pending, 1 = pending unaccepted, 2 = pending accepted.
    for length in range(7):
        for states in product(range(3), repeat=length):
            values = np.asarray(states)
            pending = _segments_from_mask(values != 0, trace_index=0)
            accepted = _segments_from_mask(values == 2, trace_index=0)
            expected = _segments_from_mask(values == 1, trace_index=0)
            assert _subtract_segments([pending], accepted) == [expected]


def test_grouping_accepts_only_accurate_pending_rows_without_overwrite() -> (
    None
):
    traces = [_trace([0.0, 0.1, 0.0, 0.0]), _trace([0.2, 0.19])]
    pending = [
        [TraceSegment(0, 0, 2), TraceSegment(0, 3, 4)],
        [TraceSegment(1, 0, 2)],
    ]
    accepted = _group_matching_segments(
        traces=traces,
        pending=pending,
        model=ZeroDerivativeModel(),
        input_columns=["time", "x"],
        output_columns=["dx"],
        threshold=0.2,
    )
    assert accepted == (
        TraceSegment(0, 0, 2),
        TraceSegment(0, 3, 4),
        TraceSegment(1, 1, 2),
    )
    assert _subtract_segments(pending, accepted) == [
        [],
        [TraceSegment(1, 0, 1)],
    ]


def test_grouping_predicts_whole_batches_including_fully_assigned_traces() -> (
    None
):
    class CenteredModel(Model):
        def __init__(self) -> None:
            self.batches: list[list[float]] = []

        def _predict(
            self, input_features: pl.DataFrame | pl.LazyFrame
        ) -> pl.LazyFrame:
            frame = (
                input_features.collect()
                if isinstance(input_features, pl.LazyFrame)
                else input_features
            )
            self.batches.append(frame["x"].to_list())
            x = frame["x"].to_numpy()
            return pl.DataFrame({"dx": x - x.mean()}).lazy()

    traces = [_trace([-1.5, -0.5, 0.5, 1.5]), _trace([-0.5, 0.5])]
    pending = [[TraceSegment(0, 1, 2), TraceSegment(0, 3, 4)], []]
    model = CenteredModel()
    accepted = _group_matching_segments(
        traces=traces,
        pending=pending,
        model=model,
        input_columns=["x"],
        output_columns=["dx"],
        threshold=0.01,
    )
    assert model.batches == [[10.0, 11.0, 12.0, 13.0], [10.0, 11.0]]
    assert accepted == tuple(pending[0])


def test_grouping_validates_predictions_even_for_fully_assigned_trace() -> (
    None
):
    class MalformedAssignedModel(ZeroDerivativeModel):
        def _predict(
            self, input_features: pl.DataFrame | pl.LazyFrame
        ) -> pl.LazyFrame:
            frame = (
                input_features.collect()
                if isinstance(input_features, pl.LazyFrame)
                else input_features
            )
            if frame.height == 1:
                return pl.DataFrame({"dx": [np.nan]}).lazy()
            return super()._predict(frame)

    with pytest.raises(ValueError, match="finite"):
        _group_matching_segments(
            traces=[_trace([0.0, 0.0]), _trace([0.0])],
            pending=[[TraceSegment(0, 0, 2)], []],
            model=MalformedAssignedModel(),
            input_columns=["x"],
            output_columns=["dx"],
            threshold=0.1,
        )
