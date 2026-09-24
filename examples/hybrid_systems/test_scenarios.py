"""Gallery scenario configuration contracts."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

import flowcean.hybrid.benchmarks as benchmarks

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.hybrid_systems import scenarios

NAMES = (
    "Bouncing Ball",
    "Thermostat",
    "Hybrid Oscillator",
    "Switched Linear",
    "Relay Integrator",
    "Time-Varying Event Surface",
    "Time-Forced Switch",
    "Piecewise Affine",
    "Impact Oscillator",
    "PID-Controlled Plant",
    "Tank Valves",
    "Location Cycle",
    "Buck Converter",
    "Wind Turbine",
)


def test_order_uniqueness_and_turbine_identity() -> None:
    assert tuple(s.name for s in scenarios.SCENARIOS) == NAMES
    assert len(set(NAMES)) == 14
    assert scenarios.SCENARIOS[-1] is scenarios.WIND_TURBINE
    assert scenarios.WIND_TURBINE.t_span == (0, 120)
    assert scenarios.SCENARIOS[11].t_span == (0, 10)
    assert scenarios.SCENARIOS[12].t_span == (0, 0.02)


def test_factories_are_lazy_and_cycle_override_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_constructed(*_args: object, **_kwargs: object) -> None:
        pytest.fail("scenario factory was invoked while importing")

    # Reimport the catalogue while a model constructor fails if invoked.
    with monkeypatch.context() as patch:
        patch.setattr(benchmarks, "bouncing_ball", fail_if_constructed)
        importlib.reload(scenarios)
        assert scenarios.SCENARIOS[0].name == "Bouncing Ball"
    importlib.reload(scenarios)
    cycle = scenarios.SCENARIOS[11].factory()
    assert len(cycle.locations) == 6
    assert cycle.initial_state.shape == (4,)  # 3 states plus clock
    assert cycle.parameters["dwell_time"] == pytest.approx(0.4)
    assert (
        scenarios.SCENARIOS[1].factory()
        is not scenarios.SCENARIOS[1].factory()
    )


@pytest.mark.parametrize(
    ("name", "t", "expected"),
    [
        ("Thermostat", 1.0, [22 + 0.8 * np.sin(0.7)]),
        ("Impact Oscillator", 1.0, [0.5 * np.sin(1.5) + 0.2 * np.sin(0.2)]),
        (
            "Time-Varying Event Surface",
            1.0,
            [0.5 * np.sin(1) + 0.15 * np.sin(2.3)],
        ),
        ("PID-Controlled Plant", 0.0, [0.0, 2.0]),
        ("PID-Controlled Plant", np.pi / 2, [2.0, 0.0]),
        ("Wind Turbine", 0.0, [7.0]),
        ("Wind Turbine", 60.0, [15.0]),
    ],
)
def test_local_driving_signals(
    name: str, t: float, expected: list[float]
) -> None:
    scenario = next(s for s in scenarios.SCENARIOS if s.name == name)
    assert scenario.input_stream is not None
    np.testing.assert_allclose(scenario.input_stream(t), expected, atol=1e-14)
