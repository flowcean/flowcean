"""Tests for hybrid system simulation in flowcean."""

from collections.abc import Mapping

import numpy as np

from flowcean.ode import Guard, HybridSystem, Mode, Reset, Transition, simulate


def test_bouncing_ball_like_system() -> None:
    """Simulation yields events and expected shapes."""

    def flow(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
    ) -> np.ndarray:
        _height, velocity = state
        return np.array([velocity, -params["gravity"]], dtype=float)

    def guard(_: float, state: np.ndarray, __: Mapping[str, float]) -> float:
        return state[0]

    def reset(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
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
