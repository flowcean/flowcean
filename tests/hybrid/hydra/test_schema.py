"""Behavioral tests for HyDRA trace schemas."""

import pytest

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
