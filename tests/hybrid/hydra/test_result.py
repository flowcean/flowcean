"""Sparse flow membership and explicit row-aligned result projections."""

from dataclasses import FrozenInstanceError, fields

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.core import Model
from flowcean.hybrid.hydra import LearnedFlow, LearnedFlows, TraceSegment


class UnusedModel(Model):
    def _predict(
        self, input_features: pl.DataFrame | pl.LazyFrame
    ) -> pl.LazyFrame:
        pytest.fail("Membership projections must not invoke models")


def _result(
    segments: tuple[tuple[TraceSegment, ...], ...],
    trace_lengths: tuple[int, ...] = (6, 2),
) -> LearnedFlows:
    return LearnedFlows(
        flows=tuple(LearnedFlow(UnusedModel(), pieces) for pieces in segments),
        trace_lengths=trace_lengths,
        input_features=("x",),
        output_features=("dx",),
    )


def test_result_contains_only_flows_geometry_and_discovery_metadata() -> None:
    result = _result(((TraceSegment(0, 0, 6), TraceSegment(1, 0, 2)),))
    assert {field.name for field in fields(result)} == {
        "flows",
        "trace_lengths",
        "input_features",
        "output_features",
    }
    for name in ("complete", "stop_reason", "stopped_segment"):
        assert not hasattr(result, name)
    assert {field.name for field in fields(result.flows[0])} == {
        "model",
        "segments",
    }
    with pytest.raises(FrozenInstanceError):
        result.trace_lengths = (1,)  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(FrozenInstanceError):
        result.flows[0].segments = ()  # pyright: ignore[reportAttributeAccessIssue]


def test_projections_preserve_flow_positions_and_multiple_traces() -> None:
    result = _result(
        (
            (
                TraceSegment(0, 0, 2),
                TraceSegment(0, 4, 6),
                TraceSegment(1, 0, 1),
            ),
            (TraceSegment(0, 2, 4), TraceSegment(1, 1, 2)),
        )
    )
    ids = result.to_flow_ids()
    assert [values.tolist() for values in ids] == [
        [0, 0, 1, 1, 0, 0],
        [0, 1],
    ]
    assert all(values.dtype == np.int64 and values.ndim == 1 for values in ids)
    ids[0][:] = 99
    assert result.to_flow_ids()[0].tolist() == [0, 0, 1, 1, 0, 0]
    assert result.flows[0].segments[0] == TraceSegment(0, 0, 2)

    frames = [
        pl.DataFrame(
            {
                "metadata": ["z", None, "b", "a", "c", "d"],
                "x": pl.Series([5, 1, 2, 3, 4, 0], dtype=pl.Int8),
                "flag": [True, False] * 3,
            }
        ),
        pl.DataFrame({"other": ["a", "b"], "time": [9.0, 1.0]}),
    ]
    before = [frame.clone() for frame in frames]
    labeled = result.to_labeled_frames(frames)
    assert [frame["flow_id"].to_list() for frame in labeled] == [
        [0, 0, 1, 1, 0, 0],
        [0, 1],
    ]
    for original, snapshot, projection in zip(
        frames, before, labeled, strict=True
    ):
        assert projection is not original
        assert projection.columns == [*original.columns, "flow_id"]
        assert projection["flow_id"].dtype == pl.Int64
        assert projection["flow_id"].null_count() == 0
        assert_frame_equal(projection.drop("flow_id"), original)
        assert_frame_equal(original, snapshot)
    labeled[0].replace_column(0, pl.Series("metadata", ["changed"] * 6))
    labeled[0].replace_column(3, pl.Series("flow_id", [9] * 6, dtype=pl.Int64))
    again = result.to_labeled_frames(frames)
    assert_frame_equal(again[0].drop("flow_id"), before[0])
    assert again[0]["flow_id"].to_list() == [0, 0, 1, 1, 0, 0]


@pytest.mark.parametrize(
    ("segments", "lengths"),
    [
        ((), (3,)),
        (((TraceSegment(0, 1, 3),),), (3,)),
        (((TraceSegment(0, 0, 2),),), (3,)),
        (((TraceSegment(0, 0, 1), TraceSegment(0, 2, 3)),), (3,)),
        (((TraceSegment(0, 0, 3),),), (3, 1)),
        (((TraceSegment(0, 0, 3), TraceSegment(1, 1, 2)),), (3, 2)),
    ],
)
def test_result_rejects_leading_interior_trailing_and_other_trace_gaps(
    segments: tuple[tuple[TraceSegment, ...], ...],
    lengths: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="cover every row"):
        _result(segments, lengths)


def test_result_accepts_unsorted_complete_segments() -> None:
    result = _result(
        (
            (TraceSegment(1, 1, 2), TraceSegment(0, 2, 3)),
            (TraceSegment(0, 0, 2), TraceSegment(1, 0, 1)),
        ),
        (3, 2),
    )
    assert [ids.tolist() for ids in result.to_flow_ids()] == [
        [1, 1, 0],
        [1, 0],
    ]


def test_geometry_validation_does_not_allocate_dense_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        pytest.fail("Sparse membership must not allocate dense labels")

    for name in ("full", "empty", "zeros", "ones"):
        monkeypatch.setattr(np, name, fail)
    result = _result(((TraceSegment(0, 0, 10**12),),), (10**12,))
    assert result.trace_lengths == (10**12,)


@pytest.mark.parametrize(
    "segment",
    [
        TraceSegment(-1, 0, 1),
        TraceSegment(2, 0, 1),
        TraceSegment(0, -1, 2),
        TraceSegment(0, 2, 7),
        TraceSegment(0, 1, 1),
        TraceSegment(0, 2, 1),
        TraceSegment(0, 6, 7),
        TraceSegment(True, 0, 1),
        TraceSegment(0, 0.5, 1),  # pyright: ignore[reportArgumentType]
    ],
)
def test_result_rejects_malformed_segment_geometry(
    segment: TraceSegment,
) -> None:
    with pytest.raises(ValueError, match="invalid trace bounds"):
        _result(((segment,),))


@pytest.mark.parametrize(
    "segments",
    [
        ((TraceSegment(0, 0, 2), TraceSegment(0, 1, 3)),),
        ((TraceSegment(0, 0, 2),), (TraceSegment(0, 1, 3),)),
        ((TraceSegment(0, 2, 4),), (TraceSegment(0, 0, 3),)),
        ((TraceSegment(0, 0, 2), TraceSegment(0, 0, 2)),),
    ],
)
def test_result_rejects_overlap_within_and_across_flows(
    segments: tuple[tuple[TraceSegment, ...], ...],
) -> None:
    with pytest.raises(ValueError, match="overlap"):
        _result(segments)


def test_result_rejects_empty_flow_membership() -> None:
    with pytest.raises(ValueError, match="requires accepted segments"):
        _result(((),))


@pytest.mark.parametrize("lengths", [(), (0,), (1, 0), (-1,), (True,), (1.5,)])
def test_result_rejects_invalid_trace_lengths(
    lengths: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="nonempty positive integers"):
        _result((), lengths)


def test_frame_projection_validates_count_height_reserved_column_and_type() -> (
    None
):
    result = _result(((TraceSegment(0, 0, 2),),), (2,))
    with pytest.raises(ValueError, match="Trace count"):
        result.to_labeled_frames([])
    with pytest.raises(ValueError, match="Trace height"):
        result.to_labeled_frames([pl.DataFrame({"x": [1]})])
    with pytest.raises(ValueError, match="reserved"):
        result.to_labeled_frames([pl.DataFrame({"flow_id": [0, 0]})])
    with pytest.raises(TypeError, match="Polars DataFrames"):
        result.to_labeled_frames(
            [pl.DataFrame({"x": [1, 2]}).lazy()],  # pyright: ignore[reportArgumentType]
        )


def test_frame_projection_uses_supplied_metadata_in_positional_order() -> None:
    result = _result(((TraceSegment(0, 0, 2),),), (2,))
    original = pl.DataFrame({"x": [0, 1]})
    supplied = pl.DataFrame({"description": ["first", "second"]})
    assert result.to_labeled_frames([supplied])[0].to_dict(
        as_series=False
    ) == {"description": ["first", "second"], "flow_id": [0, 0]}
    assert original.columns == ["x"]
