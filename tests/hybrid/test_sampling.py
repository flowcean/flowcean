"""Sampling and table schema behavior of a continuous hybrid execution."""

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from flowcean.hybrid import HybridSystem, Location, Transition, simulate


def _trajectory(*, input_stream=None):
    first = Location(lambda: np.array([1.0, -2.0]), label="same")
    second = Location(lambda: np.array([0.0, -2.0]), label="same")
    system = HybridSystem(
        [first, second],
        [Transition(first, second, lambda t: t - 0.5)],
        first,
        np.array([1.0, 2.0]),
    )
    return simulate(system, (0, 1), input_stream=input_stream)


def test_default_schema_and_optional_columns_preserve_location_identity() -> (
    None
):
    trajectory = _trajectory(input_stream=lambda t: np.array([4.0 + t]))
    frame = trajectory.sample([0, 0.5, 1])
    assert frame.columns == ["t", "x0", "x1", "location_id", "location_time"]
    assert frame["location_id"].to_list() == [0, 1, 1]
    assert frame["location_time"].to_list() == [0, 0, 0.5]
    frame = trajectory.sample(
        [0, 0.5, 1],
        include_location_label=True,
        include_inputs=True,
        include_derivatives=True,
    )
    assert frame.columns == [
        "t",
        "x0",
        "x1",
        "location_id",
        "location_label",
        "location_time",
        "u0",
        "dx0",
        "dx1",
    ]
    assert frame["location_label"].to_list() == ["same"] * 3
    np.testing.assert_allclose(frame["u0"], [4, 4.5, 5])
    np.testing.assert_allclose(frame["dx0"], [1, 0, 0])
    np.testing.assert_allclose(frame["dx1"], [-2, -2, -2])


def test_empty_grid_keeps_state_derivative_and_location_dtypes() -> None:
    frame = _trajectory().sample(
        [], include_derivatives=True, include_location_label=True
    )
    assert frame.height == 0
    assert frame.schema == {
        "t": pl.Float64,
        "x0": pl.Float64,
        "x1": pl.Float64,
        "location_id": pl.Int64,
        "location_label": pl.String,
        "location_time": pl.Float64,
        "dx0": pl.Float64,
        "dx1": pl.Float64,
    }


def test_zero_dimensional_state_keeps_location_and_time_columns() -> None:
    location = Location(lambda: np.empty(0))
    system = HybridSystem([location], [], location, np.empty(0))
    trajectory = simulate(system, (0, 1))
    assert trajectory.evaluate(0.5).state.shape == (0,)
    for times in ([], [0, 0.5, 1]):
        frame = trajectory.sample(times, include_derivatives=True)
        assert frame.schema == {
            "t": pl.Float64,
            "location_id": pl.Int64,
            "location_time": pl.Float64,
        }
        assert frame.height == len(times)


def test_derivatives_receive_a_writable_copy_of_the_evaluated_state() -> None:
    def flow(state):
        state[:] = 100
        return np.array([1.0])

    location = Location(flow)
    trajectory = simulate(
        HybridSystem([location], [], location, np.array([2.0])), (0, 0)
    )
    frame = trajectory.sample([0, 0], include_derivatives=True)
    assert frame["x0"].to_list() == [2, 2]
    assert frame["dx0"].to_list() == [1, 1]
    np.testing.assert_allclose(trajectory.evaluate(0).state, [2])


def test_repeated_sampling_is_independent_of_grid_and_preserves_duplicates() -> (
    None
):
    trajectory = _trajectory()
    frame = trajectory.sample(t for t in [0, 0.5, 0.5, 1])
    assert frame["t"].to_list() == [0, 0.5, 0.5, 1]
    assert frame["location_id"].to_list() == [0, 1, 1, 1]
    assert_frame_equal(trajectory.sample(dt=0.25), trajectory.sample(dt=0.25))
    assert trajectory.sample(dt=0.3)["t"][-1] == 1
