"""Thermostat benchmark."""

import numpy as np

from flowcean.ode import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    InputStream,
    Reset,
    Location,
    Parameters,
    Transition,
)


def thermostat_variance_stream(t: float) -> np.ndarray:
    variance = np.sin(0.7 * t)
    return np.array([variance], dtype=float)


def thermostat_varying_delay(
    ambient: float = 20.0,
    heating_power: float = 5.0,
    cooling_rate: float = 0.3,
    threshold_low: float = 22.0,
    threshold_high: float = 25.0,
    delay_time: float = 0.25,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a thermostat benchmark system.

    Args:
        ambient: Ambient temperature.
        heating_power: Heating input strength.
        cooling_rate: Cooling coefficient.
        threshold_low: Lower temperature threshold.
        threshold_high: Upper temperature threshold.
        delay_time: Delay time for transitions.
        initial_state: Optional initial temperature.

    Returns:
        HybridSystem configured for thermostat switching.
    """
    def _reset_timer(
        _t: float,
        state: np.ndarray,
        _parameters: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        updated = state.copy()
        updated[1] = 0.0
        return updated

    def heating(
        t: float,
        state: np.ndarray,
        params: Parameters,
        input_stream: InputStream,
    ) -> np.ndarray:
        temperature = state[0]
        cooling_rate_correction = 0.1 * float(input_stream(t)[0])
        return np.array(
            [
                -(params["cooling_rate"] + cooling_rate_correction) * (temperature - params["ambient"])
                + params["heating_power"],
                1,
            ],
        )

    def cooling(
        t: float,
        state: np.ndarray,
        params: Parameters,
        input_stream: InputStream,
    ) -> np.ndarray:
        temperature = state[0]
        cooling_rate_correction = 0.1 * float(input_stream(t)[0])
        return np.array(
            [
                -(params["cooling_rate"] + cooling_rate_correction) * (temperature - params["ambient"]),
                1,
            ],
        )

    def event_surface_delay(
        t: float,
        state: np.ndarray,
        params: Parameters,
        input_stream: InputStream,
    ) -> float:
        delay_variance = 0.05 * float(input_stream(t)[0])
        return state[1] - params["delay_time"] - delay_variance
    
    def event_surface_high(
        t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0] - params["threshold_high"]

    def event_surface_low(
        t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0] - params["threshold_low"]

    heating_dynamics = ContinuousDynamics(heating, label="heating")
    cooling_dynamics = ContinuousDynamics(cooling, label="cooling")

    heating_mode = Location(heating_dynamics, label="heating")
    heating_mode_delay = Location(heating_dynamics, label="heating_delay")
    cooling_mode = Location(cooling_dynamics, label="cooling")
    cooling_mode_delay = Location(cooling_dynamics, label="cooling_delay")

    to_cooling_delay = Transition(
        source=heating_mode,
        target=heating_mode_delay,
        event=EventSurface(
            event_surface_high,
            direction=CrossingDirection.RISING,
            label="too_hot_delay",
        ),
        reset = Reset(_reset_timer, label="reset_timer"),
    )
    to_cooling = Transition(
        source=heating_mode_delay,
        target=cooling_mode,
        event=EventSurface(
            event_surface_delay,
            direction=CrossingDirection.RISING,
            label="too_hot_delay_over",
        ),
    )
    to_heating_delay = Transition(
        source=cooling_mode,
        target=cooling_mode_delay,
        event=EventSurface(
            event_surface_low,
            direction=CrossingDirection.FALLING,
            label="too_cold_delay",
        ),
        reset = Reset(_reset_timer, label="reset_timer"),
    )
    to_heating = Transition(
        source=cooling_mode_delay,
        target=heating_mode,
        event=EventSurface(
            event_surface_delay,
            direction=CrossingDirection.RISING,
            label="too_cold_delay_over",
        ),
    )

    if initial_state is None:
        initial_state = np.array([ambient,0.0], dtype=float)

    return HybridSystem(
        locations=[heating_mode, cooling_mode, heating_mode_delay, cooling_mode_delay],
        transitions=[to_cooling_delay, to_cooling, to_heating_delay, to_heating],
        initial_location=heating_mode,
        initial_state=initial_state,
        parameters={
            "ambient": ambient,
            "heating_power": heating_power,
            "cooling_rate": cooling_rate,
            "threshold_low": threshold_low,
            "threshold_high": threshold_high,
            "delay_time": delay_time,
        },
    )
