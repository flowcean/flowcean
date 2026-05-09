from __future__ import annotations

import numpy as np

from flowcean.ode import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    InputStream,
    Location,
    Parameters,
    Transition,
)


def thermostat_target_stream(t: float) -> np.ndarray:
    target = 22.0 + 0.8 * np.sin(0.7 * t)
    return np.array([target], dtype=float)


def thermostat(
    ambient: float = 20.0,
    heating_power: float = 5.0,
    cooling_rate: float = 0.3,
    hysteresis: float = 2.0,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a thermostat benchmark system.

    Args:
        ambient: Ambient temperature.
        heating_power: Heating input strength.
        cooling_rate: Cooling coefficient.
        hysteresis: Temperature hysteresis band width.
        initial_state: Optional initial temperature.

    Returns:
        HybridSystem configured for thermostat switching.
    """

    def heating(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        temperature = state[0]
        return np.array(
            [
                -params["cooling_rate"] * (temperature - params["ambient"])
                + params["heating_power"],
            ],
        )

    def cooling(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        temperature = state[0]
        return np.array(
            [
                -params["cooling_rate"] * (temperature - params["ambient"]),
            ],
        )

    def event_surface_high(
        t: float,
        state: np.ndarray,
        params: Parameters,
        input_stream: InputStream,
    ) -> float:
        target = float(input_stream(t)[0])
        return state[0] - (target + 0.5 * params["hysteresis"])

    def event_surface_low(
        t: float,
        state: np.ndarray,
        params: Parameters,
        input_stream: InputStream,
    ) -> float:
        target = float(input_stream(t)[0])
        return state[0] - (target - 0.5 * params["hysteresis"])

    heating_dynamics = ContinuousDynamics(heating, label="heating")
    cooling_dynamics = ContinuousDynamics(cooling, label="cooling")
    heating_mode = Location(heating_dynamics, label="heating")
    cooling_mode = Location(cooling_dynamics, label="cooling")

    to_cooling = Transition(
        source=heating_mode,
        target=cooling_mode,
        event=EventSurface(
            event_surface_high,
            direction=CrossingDirection.RISING,
            label="too_hot",
        ),
    )
    to_heating = Transition(
        source=cooling_mode,
        target=heating_mode,
        event=EventSurface(
            event_surface_low,
            direction=CrossingDirection.FALLING,
            label="too_cold",
        ),
    )

    if initial_state is None:
        initial_state = np.array([ambient], dtype=float)

    return HybridSystem(
        locations=[heating_mode, cooling_mode],
        transitions=[to_cooling, to_heating],
        initial_location=heating_mode,
        initial_state=initial_state,
        parameters={
            "ambient": ambient,
            "heating_power": heating_power,
            "cooling_rate": cooling_rate,
            "hysteresis": hysteresis,
        },
    )
