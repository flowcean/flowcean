"""Tests for hybrid system simulation in flowcean."""

from collections.abc import Mapping

import numpy as np
import pytest

from flowcean.ode import (
    Guard,
    HybridSystem,
    InputStream,
    Mode,
    Reset,
    Transition,
    simulate,
)


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
        modes={"flight": Mode(name="flight", flow=flow)},
        transitions=[
            Transition(
                source="flight",
                target="flight",
                guard=Guard(
                    name="ground",
                    fn=guard,
                    direction=-1,
                    terminal=True,
                ),
                reset=Reset(
                    name="bounce",
                    fn=reset,
                    params={"restitution": 0.6},
                ),
            ),
        ],
        initial_mode="flight",
        initial_state=np.array([1.0, 0.0], dtype=float),
        params={"gravity": 9.81},
    )

    trace = simulate(system, t_span=(0.0, 2.0), max_jumps=128)
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
        modes={"linear": Mode(name="linear", flow=flow)},
        transitions=[],
        initial_mode="linear",
        initial_state=np.array([1.0, 0.0], dtype=float),
    )

    times = np.linspace(0.0, 2.0, 11)
    trace = simulate(system, t_span=(0.0, 2.0), sample_times=times)
    assert np.allclose(trace.t, times)


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
        modes={"m": Mode(name="m", flow=flow)},
        transitions=[],
        initial_mode="m",
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
        modes={
            "a": Mode(name="a", flow=flow),
            "b": Mode(name="b", flow=flow),
        },
        transitions=[
            Transition(
                source="a",
                target="b",
                guard=Guard(
                    name="delayed_crossing",
                    fn=delayed_guard,
                    direction=1,
                    terminal=True,
                ),
            ),
        ],
        initial_mode="a",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(
        system,
        t_span=(0.0, 2.0),
        input_stream=lambda t: np.array([t], dtype=float),
    )
    assert len(trace.events) == 1
    assert trace.events[0].time == pytest.approx(1.0, rel=1e-4, abs=1e-4)


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
        modes={"m": Mode(name="m", flow=flow)},
        transitions=[
            Transition(
                source="m",
                target="m",
                guard=Guard(
                    name="half_second",
                    fn=guard,
                    direction=1,
                    terminal=True,
                ),
                reset=Reset(name="set_from_input", fn=reset),
            ),
        ],
        initial_mode="m",
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
        modes={"m": Mode(name="m", flow=flow)},
        transitions=[],
        initial_mode="m",
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
        modes={"m": Mode(name="m", flow=flow)},
        transitions=[],
        initial_mode="m",
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
        modes={"m": Mode(name="m", flow=flow)},
        transitions=[],
        initial_mode="m",
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
        modes={"m": Mode(name="m", flow=flow)},
        transitions=[],
        initial_mode="m",
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
        modes={"m": Mode(name="m", flow=flow)},
        transitions=[],
        initial_mode="m",
        initial_state=np.array([0.0], dtype=float),
    )

    trace = simulate(system, t_span=(0.0, 1.0), sample_times=[0.0, 1.0])
    assert trace.u is None


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
        modes={"m": Mode(name="m", flow=flow)},
        transitions=[],
        initial_mode="m",
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
        modes={"m": Mode(name="m", flow=flow)},
        transitions=[],
        initial_mode="m",
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
        modes={"m": Mode(name="m", flow=flow)},
        transitions=[],
        initial_mode="m",
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
