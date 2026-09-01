"""Behavioral tests for HyDRA trace schemas and labels."""

import numpy as np
import polars as pl
import pytest

from flowcean.hybrid.hydra.learner import HyDRATrace, TraceSegment
from flowcean.hybrid.hydra.schema import HyDRATraceSchema


def test_trace_schema_orders_features_and_validates_columns() -> None:
    schema = HyDRATraceSchema(
        time="time",
        state=("position", "velocity"),
        derivative=("d_position", "d_velocity"),
        inputs=("force", "temperature"),
    )

    assert schema.input_features == (
        "time",
        "position",
        "velocity",
        "force",
        "temperature",
    )
    # Input frame order may differ, while derivative order is significant.
    schema.validate_input_features(
        ["force", "velocity", "time", "temperature", "position"],
    )
    schema.validate_output_features(["d_position", "d_velocity"])
    schema.validate_state_derivative_width()

    with pytest.raises(ValueError, match="input_features must match"):
        schema.validate_input_features(["time", "position", "velocity"])
    with pytest.raises(ValueError, match="derivative order"):
        schema.validate_output_features(["d_velocity", "d_position"])


def test_trace_schema_rejects_duplicate_columns_and_width_mismatch() -> None:
    with pytest.raises(ValueError, match="columns must be disjoint"):
        HyDRATraceSchema(
            time="time",
            state=("x",),
            derivative=("dx",),
            inputs=("x",),
        )

    schema = HyDRATraceSchema(
        time="time",
        state=("x", "y"),
        derivative=("dx",),
    )
    with pytest.raises(ValueError, match="widths must match"):
        schema.validate_state_derivative_width()


def test_trace_labeling_is_immutable_and_segments_are_inclusive() -> None:
    frame = pl.DataFrame({"time": [0.0, 1.0, 2.0], "x": [2.0, 3.0, 4.0]})
    trace = HyDRATrace.unlabeled(frame)

    labeled = trace.with_labeled_segment(
        start_index=1,
        end_index=2,
        mode_id=7,
    )

    assert trace.mode_labels.tolist() == [-1, -1, -1]
    assert labeled.mode_labels.tolist() == [-1, 7, 7]
    assert labeled.unlabeled_indices() == [0]
    assert labeled.to_labeled_frame()["mode"].to_list() == [None, 7, 7]
    assert labeled.segment_frame(TraceSegment(0, 1, 2)).to_dict(
        as_series=False,
    ) == {"time": [1.0, 2.0], "x": [3.0, 4.0]}


def test_trace_validates_label_shape_and_storage() -> None:
    frame = pl.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="1D array"):
        HyDRATrace(frame, np.array([[0, 1]]))
    with pytest.raises(ValueError, match="match frame height"):
        HyDRATrace(frame, np.array([0]))
    with pytest.raises(ValueError, match="must not store mode labels"):
        HyDRATrace(
            frame.with_columns(pl.Series("mode", [0, 1])),
            np.array([0, 1]),
        )
