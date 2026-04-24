"""Tests for hybrid system simulation in flowcean."""

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest

from flowcean.ode import (
    ContinuousDynamics,
    Guard,
    HybridSystem,
    InputStream,
    Location,
    Reset,
    Transition,
    generate_traces,
    simulate,
)


def _constant_velocity_system() -> HybridSystem:
    def flow(
        _t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([1.0], dtype=float)

    location = Location("loc", ContinuousDynamics("flow", flow))
    return HybridSystem(
        locations={"loc": location},
        transitions=[],
        initial_location="loc",
        initial_state=np.array([0.0], dtype=float),
    )


def test_ode_public_api_exports_locations_not_modes() -> None:
    """The hybrid ODE API uses location/dynamics terminology."""
    from flowcean import ode

    assert hasattr(ode, "ContinuousDynamics")
    assert hasattr(ode, "Location")
    assert not hasattr(ode, "Mode")


def test_simulate_raises_when_solver_fails() -> None:
    def singular_flow(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([1.0 / (t - 0.5)], dtype=float)

    location = Location(
        name="loc",
        dynamics=ContinuousDynamics(name="singular", flow=singular_flow),
    )
    system = HybridSystem(
        locations={"loc": location},
        transitions=[],
        initial_location="loc",
        initial_state=np.array([0.0], dtype=float),
    )

    with pytest.raises(RuntimeError, match="ODE integration failed"):
        simulate(system, t_span=(0.0, 1.0))


def test_simulate_rejects_sample_times_before_t_span() -> None:
    system = _constant_velocity_system()

    with pytest.raises(
        ValueError,
        match="sample_times must lie within t_span",
    ):
        simulate(system, t_span=(0.0, 1.0), sample_times=[-1.0, 0.0])


def test_simulate_rejects_sample_times_after_t_span() -> None:
    system = _constant_velocity_system()

    with pytest.raises(
        ValueError,
        match="sample_times must lie within t_span",
    ):
        simulate(system, t_span=(0.0, 1.0), sample_times=[0.0, 2.0])


def test_simulate_sample_dt_includes_endpoint_without_overshoot() -> None:
    trace = simulate(
        _constant_velocity_system(),
        t_span=(0.0, 1.0),
        sample_dt=0.6,
    )

    assert np.allclose(trace.t, [0.0, 0.6, 1.0])


def test_simulate_sample_dt_snaps_close_endpoint_to_t_span_end() -> None:
    trace = simulate(
        _constant_velocity_system(),
        t_span=(0.0, 1.0),
        sample_dt=np.nextafter(1.0, 0.0),
    )

    assert trace.t.tolist() == [0.0, 1.0]


def test_simulate_sample_dt_keeps_non_precision_endpoint_step() -> None:
    trace = simulate(
        _constant_velocity_system(),
        t_span=(0.0, 1.0),
        sample_dt=0.99999,
    )

    assert trace.t.tolist() == [0.0, 0.99999, 1.0]


def test_simulate_rejects_non_finite_sample_times() -> None:
    system = _constant_velocity_system()

    with pytest.raises(ValueError, match="sample_times must be finite"):
        simulate(system, t_span=(0.0, 1.0), sample_times=[0.0, np.nan, 1.0])


def test_simulate_rejects_non_finite_sample_dt() -> None:
    system = _constant_velocity_system()

    with pytest.raises(ValueError, match="sample_dt must be finite"):
        simulate(system, t_span=(0.0, 1.0), sample_dt=np.nan)


def test_locations_can_share_continuous_dynamics() -> None:
    """Several discrete locations may reference one dynamics object."""

    def flow(
        _: float,
        _state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([params["rate"]], dtype=float)

    dynamics = ContinuousDynamics(
        name="constant_rate",
        flow=flow,
        params={"rate": 1.0},
    )
    system = HybridSystem(
        locations={
            "a": Location(name="a", dynamics=dynamics),
            "b": Location(name="b", dynamics=dynamics),
        },
        transitions=[],
        initial_location="a",
        initial_state=np.array([0.0], dtype=float),
    )

    assert system.locations["a"].dynamics is dynamics
    assert system.locations["b"].dynamics is dynamics


def test_bouncing_ball_like_system() -> None:
    """Simulation yields events and expected shapes."""

    def flow(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        _height, velocity = state
        return np.array([velocity, -params["gravity"]], dtype=float)

    def guard(
        _: float,
        state: np.ndarray,
        __: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[0]

    def reset(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        height, velocity = state
        return np.array(
            [height, -params["restitution"] * velocity],
            dtype=float,
        )

    system = HybridSystem(
        locations={
            "flight": Location(
                name="flight",
                dynamics=ContinuousDynamics(name="flight_dynamics", flow=flow),
            ),
        },
        transitions=[
            Transition(
                source_location="flight",
                target_location="flight",
                guard=Guard(
                    name="ground",
                    fn=guard,
                    direction=-1,
                ),
                reset=Reset(
                    name="bounce",
                    fn=reset,
                    params={"restitution": 0.6},
                ),
            ),
        ],
        initial_location="flight",
        initial_state=np.array([1.0, 0.0], dtype=float),
        params={"gravity": 9.81},
    )

    trace = simulate(system, t_span=(0.0, 1.0), max_jumps=128)
    assert trace.t.size > 2
    assert trace.x.shape[1] == 2
    assert trace.events


def test_dense_sampling() -> None:
    """Sampling at predefined times yields matching timestamps."""

    def flow(
        _: float,
        state: np.ndarray,
        __: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([state[1], -state[0]], dtype=float)

    system = HybridSystem(
        locations={
            "linear": Location(
                name="linear",
                dynamics=ContinuousDynamics(name="linear_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="linear",
        initial_state=np.array([1.0, 0.0], dtype=float),
    )

    times = np.linspace(0.0, 2.0, 11)
    trace = simulate(system, t_span=(0.0, 2.0), sample_times=times)
    assert np.allclose(trace.t, times)


def test_guard_rejects_terminal_argument() -> None:
    """Guard no longer accepts a terminal flag."""

    def guard(
        _: float,
        state: np.ndarray,
        __: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[0]

    with pytest.raises(TypeError, match="terminal"):
        cast("Any", Guard)(
            name="ground",
            fn=guard,
            direction=-1,
            terminal=True,
        )


def test_flow_can_use_input_stream() -> None:
    """Flow callback can access current and delayed inputs."""

    def flow(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        input_stream: InputStream,
    ) -> np.ndarray:
        return np.array(
            [
                input_stream(t)[0] + input_stream(t - 0.5)[0],
            ],
            dtype=float,
        )

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(
        system,
        t_span=(0.0, 1.0),
        sample_times=np.linspace(0.0, 1.0, 6),
        input_stream=lambda t: np.array([t], dtype=float),
    )
    expected = trace.t**2 - 0.5 * trace.t
    assert np.allclose(trace.x[:, 0], expected, atol=2e-2)


def test_guard_can_use_delayed_input_stream() -> None:
    """Guard callback can use delayed inputs to trigger transitions."""

    def flow(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([0.0], dtype=float)

    def delayed_guard(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        input_stream: InputStream,
    ) -> float:
        return input_stream(t - 1.0)[0]

    system = HybridSystem(
        locations={
            "a": Location(
                name="a",
                dynamics=ContinuousDynamics(name="a_dynamics", flow=flow),
            ),
            "b": Location(
                name="b",
                dynamics=ContinuousDynamics(name="b_dynamics", flow=flow),
            ),
        },
        transitions=[
            Transition(
                source_location="a",
                target_location="b",
                guard=Guard(
                    name="delayed_crossing",
                    fn=delayed_guard,
                    direction=1,
                ),
            ),
        ],
        initial_location="a",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(
        system,
        t_span=(0.0, 2.0),
        input_stream=lambda t: np.array([t], dtype=float),
    )
    assert len(trace.events) == 1
    assert trace.events[0].time == pytest.approx(1.0, rel=1e-4, abs=1e-4)


def test_simulate_applies_transition_when_initial_guard_is_active() -> None:
    def flow(
        _t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([1.0], dtype=float)

    def guard(
        _t: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> float:
        return float(state[0])

    left = Location("left", ContinuousDynamics("left_flow", flow))
    right = Location("right", ContinuousDynamics("right_flow", flow))
    system = HybridSystem(
        locations={"left": left, "right": right},
        transitions=[
            Transition(
                source_location="left",
                target_location="right",
                guard=Guard("at_zero", guard, direction=1),
            ),
        ],
        initial_location="left",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(system, t_span=(0.0, 1.0))

    assert len(trace.events) == 1
    assert trace.events[0].time == pytest.approx(0.0)
    assert trace.events[0].source_location == "left"
    assert trace.events[0].target_location == "right"
    assert trace.location[-1] == "right"


def test_chained_immediate_transitions_keep_physical_event_time() -> None:
    def flow(
        _t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([1.0], dtype=float)

    def guard(
        _t: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> float:
        return float(state[0])

    a = Location("a", ContinuousDynamics("a_flow", flow))
    b = Location("b", ContinuousDynamics("b_flow", flow))
    c = Location("c", ContinuousDynamics("c_flow", flow))
    system = HybridSystem(
        locations={"a": a, "b": b, "c": c},
        transitions=[
            Transition(
                source_location="a",
                target_location="b",
                guard=Guard("a_to_b", guard, direction=1),
            ),
            Transition(
                source_location="b",
                target_location="c",
                guard=Guard("b_to_c", guard, direction=1),
            ),
        ],
        initial_location="a",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(system, t_span=(0.0, 1.0))

    assert len(trace.events) == 2
    assert trace.events[0].time == pytest.approx(0.0)
    assert trace.events[1].time == pytest.approx(0.0)
    assert trace.events[1].time == trace.events[0].time
    assert trace.events[0].target_location == "b"
    assert trace.events[1].target_location == "c"
    assert trace.location[-1] == "c"


def test_chained_immediate_time_guard_keeps_physical_event_time() -> None:
    def flow(
        _t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([1.0], dtype=float)

    def state_guard(
        _t: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> float:
        return float(state[0] - 0.5)

    def time_guard(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> float:
        return t - 0.5

    a = Location("a", ContinuousDynamics("a_flow", flow))
    b = Location("b", ContinuousDynamics("b_flow", flow))
    c = Location("c", ContinuousDynamics("c_flow", flow))
    system = HybridSystem(
        locations={"a": a, "b": b, "c": c},
        transitions=[
            Transition(
                source_location="a",
                target_location="b",
                guard=Guard("a_to_b", state_guard, direction=1),
            ),
            Transition(
                source_location="b",
                target_location="c",
                guard=Guard("b_to_c", time_guard, direction=1),
            ),
        ],
        initial_location="a",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(system, t_span=(0.0, 1.0))

    assert len(trace.events) == 2
    assert trace.events[0].time == trace.events[1].time
    assert trace.events[0].time == pytest.approx(0.5)
    assert trace.events[1].time == pytest.approx(0.5)
    assert trace.events[0].target_location == "b"
    assert trace.events[1].target_location == "c"


def test_direction_zero_immediate_guard_skips_target_flow() -> None:
    def flow(
        _t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([1.0], dtype=float)

    def raising_flow(
        _t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> np.ndarray:
        msg = "target flow should not be evaluated for direction-zero guard"
        raise AssertionError(msg)

    def source_guard(
        _t: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> float:
        return float(state[0] - 0.5)

    def immediate_guard(
        _t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> float:
        return 0.0

    a = Location("a", ContinuousDynamics("a_flow", flow))
    b = Location("b", ContinuousDynamics("b_flow", raising_flow))
    c = Location("c", ContinuousDynamics("c_flow", flow))
    system = HybridSystem(
        locations={"a": a, "b": b, "c": c},
        transitions=[
            Transition(
                source_location="a",
                target_location="b",
                guard=Guard("a_to_b", source_guard, direction=1),
            ),
            Transition(
                source_location="b",
                target_location="c",
                guard=Guard("b_to_c", immediate_guard, direction=0),
            ),
        ],
        initial_location="a",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(system, t_span=(0.0, 1.0))

    assert len(trace.events) == 2
    assert trace.events[0].target_location == "b"
    assert trace.events[1].target_location == "c"


def test_persistent_immediate_self_loop_chattering_hits_max_jumps() -> None:
    def flow(
        _t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([0.0], dtype=float)

    def guard(
        _t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> float:
        return 0.0

    loop = Location("loop", ContinuousDynamics("loop_flow", flow))
    system = HybridSystem(
        locations={"loop": loop},
        transitions=[
            Transition(
                source_location="loop",
                target_location="loop",
                guard=Guard("always", guard, direction=0),
            ),
        ],
        initial_location="loop",
        initial_state=np.array([0.0], dtype=float),
    )

    with pytest.raises(
        RuntimeError,
        match="Maximum number of transitions exceeded",
    ):
        simulate(system, t_span=(0.0, 1.0), max_jumps=3)


def test_directional_immediate_self_loop_chattering_hits_max_jumps() -> None:
    def flow(
        _t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([1.0], dtype=float)

    def guard(
        _t: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> float:
        return float(state[0])

    loop = Location("loop", ContinuousDynamics("loop_flow", flow))
    system = HybridSystem(
        locations={"loop": loop},
        transitions=[
            Transition(
                source_location="loop",
                target_location="loop",
                guard=Guard("positive", guard, direction=1),
            ),
        ],
        initial_location="loop",
        initial_state=np.array([0.0], dtype=float),
    )

    with pytest.raises(
        RuntimeError,
        match="Maximum number of transitions exceeded",
    ):
        simulate(system, t_span=(0.0, 1e-13), max_jumps=3)


def test_time_driven_immediate_self_loop_chattering_hits_max_jumps() -> None:
    def flow(
        _t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([0.0], dtype=float)

    def guard(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> float:
        return t

    loop = Location("loop", ContinuousDynamics("loop_flow", flow))
    system = HybridSystem(
        locations={"loop": loop},
        transitions=[
            Transition(
                source_location="loop",
                target_location="loop",
                guard=Guard("time", guard, direction=1),
            ),
        ],
        initial_location="loop",
        initial_state=np.array([0.0], dtype=float),
    )

    with pytest.raises(
        RuntimeError,
        match="Maximum number of transitions exceeded",
    ):
        simulate(system, t_span=(0.0, 1e-9), max_jumps=3)


def test_reset_can_use_input_stream() -> None:
    """Reset callback can access stream values at event times."""

    def flow(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([0.0], dtype=float)

    def guard(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return t - 0.5

    def reset(
        t: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        input_stream: InputStream,
    ) -> np.ndarray:
        updated = state.copy()
        updated[0] = input_stream(t)[0]
        return updated

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
            "done": Location(
                name="done",
                dynamics=ContinuousDynamics(name="done_dynamics", flow=flow),
            ),
        },
        transitions=[
            Transition(
                source_location="m",
                target_location="done",
                guard=Guard(
                    name="half_second",
                    fn=guard,
                    direction=1,
                ),
                reset=Reset(name="set_from_input", fn=reset),
            ),
        ],
        initial_location="m",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(
        system,
        t_span=(0.0, 1.0),
        input_stream=lambda t: np.array([3.0 * t], dtype=float),
    )
    assert len(trace.events) == 1
    assert trace.events[0].state[0] == pytest.approx(1.5, rel=1e-4, abs=1e-4)


def test_invalid_input_stream_shape_raises() -> None:
    """Non-1D input stream values are rejected."""

    def flow(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([0.0], dtype=float)

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([0.0], dtype=float),
    )

    with pytest.raises(ValueError, match="1D array"):
        simulate(
            system,
            t_span=(0.0, 1.0),
            sample_times=np.linspace(0.0, 1.0, 3),
            input_stream=lambda _: np.array([[1.0]], dtype=float),
        )


def test_invalid_input_stream_dtype_raises() -> None:
    """Non-numeric input stream values are rejected."""

    def flow(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([0.0], dtype=float)

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([0.0], dtype=float),
    )

    with pytest.raises(ValueError, match="numeric values"):
        simulate(
            system,
            t_span=(0.0, 1.0),
            sample_times=np.linspace(0.0, 1.0, 3),
            input_stream=lambda _: np.array(["x"], dtype=object),
        )


def test_changing_input_dimension_raises() -> None:
    """Input stream dimensions must stay constant during a run."""

    def flow(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([input_stream(t)[0]], dtype=float)

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([0.0], dtype=float),
    )

    with pytest.raises(ValueError, match="dimension changed"):
        simulate(
            system,
            t_span=(0.0, 1.0),
            input_stream=lambda t: (
                np.array([1.0], dtype=float)
                if t < 0.5
                else np.array([1.0, 2.0], dtype=float)
            ),
        )


def test_trace_captures_inputs_when_stream_is_provided() -> None:
    """Trace stores sampled inputs when an input stream is configured."""

    def flow(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([input_stream(t)[0]], dtype=float)

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([0.0], dtype=float),
    )

    times = np.array([0.0, 0.5, 1.0], dtype=float)
    trace = simulate(
        system,
        t_span=(0.0, 1.0),
        sample_times=times,
        input_stream=lambda t: np.array([t, t + 1.0], dtype=float),
    )
    assert trace.u is not None
    assert trace.u.shape == (times.size, 2)
    assert np.allclose(trace.u[:, 0], times)
    assert np.allclose(trace.u[:, 1], times + 1.0)


def test_trace_inputs_are_none_without_stream() -> None:
    """Trace input matrix is omitted when no stream is configured."""

    def flow(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([0.0], dtype=float)

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(system, t_span=(0.0, 1.0), sample_times=[0.0, 1.0])
    assert trace.u is None


def test_trace_records_location_labels() -> None:
    """Simulation stores active discrete locations on traces."""

    def flow(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([1.0], dtype=float)

    system = HybridSystem(
        locations={
            "active": Location(
                name="active",
                dynamics=ContinuousDynamics(name="constant", flow=flow),
            ),
        },
        transitions=[],
        initial_location="active",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(system, t_span=(0.0, 1.0), sample_times=[0.0, 1.0])

    assert trace.location.tolist() == ["active", "active"]
    assert "location" in trace.as_dict()
    assert "mode" not in trace.as_dict()


def test_trace_derivatives_are_none_by_default() -> None:
    """Trace derivative matrix is omitted unless explicitly requested."""

    def flow(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([1.0], dtype=float)

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(system, t_span=(0.0, 1.0), sample_times=[0.0, 1.0])

    assert trace.dx is None


def test_trace_captures_derivatives_when_requested() -> None:
    """Derivative samples are captured on the returned sample grid."""

    def flow(
        _: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([2.0 * state[0]], dtype=float)

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([1.0], dtype=float),
    )

    times = np.array([0.0, 0.5, 1.0], dtype=float)
    trace = simulate(
        system,
        t_span=(0.0, 1.0),
        sample_times=times,
        capture_derivatives=True,
    )

    assert trace.dx is not None
    assert trace.dx.shape == trace.x.shape
    assert np.allclose(
        trace.dx[:, 0],
        2.0 * trace.x[:, 0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_sampled_derivatives_match_sampled_state_dynamics() -> None:
    """Derivative samples are evaluated on the returned state grid."""

    def flow(
        _: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([-state[0]], dtype=float)

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([1.0], dtype=float),
    )

    times = np.linspace(0.0, 1.0, 5)
    trace = simulate(
        system,
        t_span=(0.0, 1.0),
        sample_times=times,
        capture_derivatives=True,
    )

    assert trace.dx is not None
    assert np.allclose(trace.dx[:, 0], -trace.x[:, 0], rtol=1e-6, atol=1e-6)


def test_invalid_derivative_shape_raises() -> None:
    """Captured derivatives must match the state dimension."""
    invalid_sample_time = 0.123456789

    def flow(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        if t == invalid_sample_time:
            return np.array([1.0, 2.0], dtype=float)
        return np.array([1.0], dtype=float)

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([0.0], dtype=float),
    )

    with pytest.raises(ValueError, match="state dimension"):
        simulate(
            system,
            t_span=(0.0, 0.5),
            sample_times=[0.0, invalid_sample_time],
            capture_derivatives=True,
        )


def test_scalar_flow_is_rejected_for_multi_state_solver_path() -> None:
    """Multi-state solver dynamics must not silently broadcast scalars."""

    def flow(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return 1.0

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([0.0, 0.0], dtype=float),
    )

    with pytest.raises(ValueError, match="state dimension"):
        simulate(system, t_span=(0.0, 0.1))


def test_scalar_flow_is_accepted_for_single_state_solver_path() -> None:
    """Single-state solver dynamics still accept scalar flow outputs."""

    def flow(
        _: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return -state[0]

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([1.0], dtype=float),
    )

    trace = simulate(system, t_span=(0.0, 1.0), sample_times=[0.0, 0.5, 1.0])

    assert trace.x.shape == (3, 1)
    assert trace.x[-1, 0] == pytest.approx(np.exp(-1.0), rel=1e-4)


def test_trace_captures_scalar_derivatives_for_single_state_system() -> None:
    """Scalar flow outputs are accepted for 1-state derivative capture."""

    def flow(
        _: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return 3.0 * state[0]

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([2.0], dtype=float),
    )

    trace = simulate(
        system,
        t_span=(0.0, 1.0),
        sample_times=[0.0, 0.5, 1.0],
        capture_derivatives=True,
    )

    assert trace.dx is not None
    assert trace.dx.shape == (3, 1)
    assert np.allclose(
        trace.dx[:, 0],
        3.0 * trace.x[:, 0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_generate_traces_propagates_derivative_capture() -> None:
    """Batch trace generation forwards derivative capture to each trace."""

    def flow(
        _: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([-state[0]], dtype=float)

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([1.0], dtype=float),
    )

    traces = generate_traces(
        system,
        t_span=(0.0, 0.5),
        initial_states=[
            np.array([1.0], dtype=float),
            np.array([2.0], dtype=float),
        ],
        sample_times=[0.0, 0.5],
        capture_derivatives=True,
    )

    assert len(traces) == 2
    for trace in traces:
        assert trace.dx is not None
        assert trace.dx.shape == trace.x.shape


def test_derivatives_follow_post_reset_location_and_state() -> None:
    """Boundary derivative samples match trace location/state."""

    def flow_left(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([1.0], dtype=float)

    def flow_right(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([1e12 * t + input_stream(t)[0]], dtype=float)

    def guard(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return t - 0.5

    def reset(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([10.0], dtype=float)

    system = HybridSystem(
        locations={
            "left": Location(
                name="left",
                dynamics=ContinuousDynamics(
                    name="left_dynamics",
                    flow=flow_left,
                ),
            ),
            "right": Location(
                name="right",
                dynamics=ContinuousDynamics(
                    name="right_dynamics",
                    flow=flow_right,
                ),
            ),
        },
        transitions=[
            Transition(
                source_location="left",
                target_location="right",
                guard=Guard(
                    name="switch",
                    fn=guard,
                    direction=1,
                ),
                reset=Reset(name="jump", fn=reset),
            ),
        ],
        initial_location="left",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(
        system,
        t_span=(0.0, 1.0),
        sample_times=np.array([0.0, 0.5, 1.0], dtype=float),
        input_stream=lambda t: np.array([10.0 * t], dtype=float),
        capture_inputs=True,
        capture_derivatives=True,
    )

    event_time = trace.events[0].time

    assert trace.dx is not None
    assert trace.u is not None
    assert trace.location.tolist() == ["left", "right", "right"]
    assert trace.t[0] == 0.0
    assert trace.t[1] == event_time
    assert trace.t[2] == 1.0
    assert np.all(np.diff(trace.t) > 0)
    assert trace.x[0, 0] == pytest.approx(0.0)
    assert trace.x[1, 0] == pytest.approx(10.0)
    assert trace.x[2, 0] > trace.x[1, 0]
    assert np.all(np.isfinite(trace.x))
    assert np.all(np.isfinite(trace.dx))
    assert trace.dx[0, 0] == pytest.approx(1.0)
    assert trace.dx[1, 0] > trace.dx[0, 0]
    assert trace.dx[2, 0] == pytest.approx(1e12)
    assert trace.u[:, 0].tolist() == pytest.approx(
        [0.0, 10.0 * event_time, 10.0],
    )


def test_sample_immediately_after_transition_uses_target_location() -> None:
    """Samples immediately after an event follow post-transition behavior."""
    requested_transition_sample = np.nextafter(0.5, 1.0)

    def flow_left(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([1.0], dtype=float)

    def flow_right(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([1e9], dtype=float)

    def guard(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return t - 0.5

    def reset(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([10.0], dtype=float)

    system = HybridSystem(
        locations={
            "left": Location(
                name="left",
                dynamics=ContinuousDynamics(
                    name="left_dynamics",
                    flow=flow_left,
                ),
            ),
            "right": Location(
                name="right",
                dynamics=ContinuousDynamics(
                    name="right_dynamics",
                    flow=flow_right,
                ),
            ),
        },
        transitions=[
            Transition(
                source_location="left",
                target_location="right",
                guard=Guard(
                    name="switch",
                    fn=guard,
                    direction=1,
                ),
                reset=Reset(name="jump", fn=reset),
            ),
        ],
        initial_location="left",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(
        system,
        t_span=(0.0, 1.0),
        sample_times=np.array(
            [0.0, requested_transition_sample, 1.0],
            dtype=float,
        ),
        capture_derivatives=True,
    )

    event_time = trace.events[0].time

    assert trace.dx is not None
    assert len(trace.events) == 1
    assert event_time == pytest.approx(0.5)
    assert trace.location.tolist() == ["left", "right", "right"]
    assert trace.t.shape == (3,)
    assert trace.x.shape == (3, 1)
    assert trace.dx.shape == trace.x.shape
    assert trace.t[0] == 0.0
    assert trace.t[1] >= requested_transition_sample
    assert trace.t[2] == 1.0
    assert np.all(np.diff(trace.t) > 0)
    assert np.all(trace.t >= 0.0)
    assert np.all(trace.t <= 1.0)
    assert np.all(np.isfinite(trace.x))
    assert np.all(np.isfinite(trace.dx))
    assert trace.x[0, 0] == pytest.approx(0.0)
    assert trace.x[1, 0] == pytest.approx(10.0)
    assert trace.x[2, 0] > trace.x[1, 0]
    assert trace.dx[0, 0] == pytest.approx(1.0)
    assert trace.dx[1, 0] == pytest.approx(1e9)
    assert trace.dx[2, 0] == pytest.approx(1e9)


def test_immediate_directional_guard_requires_crossing() -> None:
    def flow_a(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([0.0], dtype=float)

    def flow_b(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([0.0], dtype=float)

    def guard_a_to_b(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return t - 0.5

    def guard_b_to_c(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return 5e-13 + (t - 0.5)

    system = HybridSystem(
        locations={
            "a": Location(
                name="a",
                dynamics=ContinuousDynamics("a_dynamics", flow_a),
            ),
            "b": Location(
                name="b",
                dynamics=ContinuousDynamics("b_dynamics", flow_b),
            ),
            "c": Location(
                name="c",
                dynamics=ContinuousDynamics("c_dynamics", flow_b),
            ),
        },
        transitions=[
            Transition(
                source_location="a",
                target_location="b",
                guard=Guard("a_to_b", guard_a_to_b, direction=1),
            ),
            Transition(
                source_location="b",
                target_location="c",
                guard=Guard("b_to_c", guard_b_to_c, direction=1),
            ),
        ],
        initial_location="a",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(
        system,
        t_span=(0.0, 1.0),
        sample_times=[0.0, 1.0],
    )

    assert len(trace.events) == 1
    assert trace.events[0].target_location == "b"
    assert trace.location.tolist() == ["a", "b"]


def test_missing_input_stream_access_raises() -> None:
    """Accessing input without a stream raises a clear error."""

    def flow(
        t: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([input_stream(t)[0]], dtype=float)

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([0.0], dtype=float),
    )

    with pytest.raises(ValueError, match="input_stream is required"):
        simulate(system, t_span=(0.0, 1.0))


def test_capture_inputs_true_requires_stream() -> None:
    """capture_inputs=True without a stream is rejected."""

    def flow(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([0.0], dtype=float)

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([0.0], dtype=float),
    )

    with pytest.raises(ValueError, match="capture_inputs=True"):
        simulate(system, t_span=(0.0, 1.0), capture_inputs=True)


def test_trace_input_capture_can_be_disabled() -> None:
    """Trace input matrix can be disabled even with an input stream."""

    def flow(
        _: float,
        _state: np.ndarray,
        _params: Mapping[str, float],
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([0.0], dtype=float)

    system = HybridSystem(
        locations={
            "m": Location(
                name="m",
                dynamics=ContinuousDynamics(name="m_dynamics", flow=flow),
            ),
        },
        transitions=[],
        initial_location="m",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(
        system,
        t_span=(0.0, 1.0),
        sample_times=[0.0, 1.0],
        input_stream=lambda t: np.array([t], dtype=float),
        capture_inputs=False,
    )
    assert trace.u is None
