"""Thermostat benchmark."""

from collections.abc import Mapping

import numpy as np

from flowcean.ode import Guard, HybridSystem, InputStream, Mode, Transition


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
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
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
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        temperature = state[0]
        return np.array(
            [
                -params["cooling_rate"] * (temperature - params["ambient"]),
            ],
        )

    def guard_high(
        t: float,
        state: np.ndarray,
        params: Mapping[str, float],
        input_stream: InputStream,
    ) -> float:
        target = float(input_stream(t)[0])
        return state[0] - (target + 0.5 * params["hysteresis"])

    def guard_low(
        t: float,
        state: np.ndarray,
        params: Mapping[str, float],
        input_stream: InputStream,
    ) -> float:
        target = float(input_stream(t)[0])
        return state[0] - (target - 0.5 * params["hysteresis"])

    heating_mode = Mode(name="heating", flow=heating)
    cooling_mode = Mode(name="cooling", flow=cooling)

    to_cooling = Transition(
        source="heating",
        target="cooling",
        guard=Guard(name="too_hot", fn=guard_high, direction=1, terminal=True),
    )
    to_heating = Transition(
        source="cooling",
        target="heating",
        guard=Guard(
            name="too_cold",
            fn=guard_low,
            direction=-1,
            terminal=True,
        ),
    )

    if initial_state is None:
        initial_state = np.array([ambient], dtype=float)

    return HybridSystem(
        modes={"heating": heating_mode, "cooling": cooling_mode},
        transitions=[to_cooling, to_heating],
        initial_mode="heating",
        initial_state=initial_state,
        params={
            "ambient": ambient,
            "heating_power": heating_power,
            "cooling_rate": cooling_rate,
            "hysteresis": hysteresis,
        },
    )
