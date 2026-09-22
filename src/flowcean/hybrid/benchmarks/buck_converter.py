"""Hysteretic nonsynchronous buck-converter benchmark."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

from ..hybrid_system import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    InputStream,
    Location,
    Parameters,
    Reset,
    SurfaceEntryPolicy,
    Transition,
)


def _validate_parameters(
    source_voltage: float,
    inductance: float,
    capacitance: float,
    load_resistance: float,
    inductor_resistance: float,
    switch_resistance: float,
    voltage_high: float,
    voltage_low: float,
) -> dict[str, float]:
    values = {
        "source_voltage": source_voltage,
        "inductance": inductance,
        "capacitance": capacitance,
        "load_resistance": load_resistance,
        "inductor_resistance": inductor_resistance,
        "switch_resistance": switch_resistance,
        "voltage_high": voltage_high,
        "voltage_low": voltage_low,
    }
    if not all(math.isfinite(value) for value in values.values()):
        message = "buck-converter parameters must be finite"
        raise ValueError(message)
    if (
        source_voltage <= 0.0
        or inductance <= 0.0
        or capacitance <= 0.0
        or load_resistance <= 0.0
    ):
        message = (
            "source_voltage, inductance, capacitance, and load_resistance "
            "must be strictly positive"
        )
        raise ValueError(message)
    if inductor_resistance < 0.0 or switch_resistance < 0.0:
        message = (
            "inductor_resistance and switch_resistance must be nonnegative"
        )
        raise ValueError(message)
    if not 0.0 < voltage_low < voltage_high < source_voltage:
        message = (
            "voltages must satisfy 0 < voltage_low < voltage_high "
            "< source_voltage"
        )
        raise ValueError(message)
    return values


def _validate_initial_state(
    initial_state: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    values = (2.0, 7.0) if initial_state is None else initial_state
    try:
        initial = np.array(values, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        message = "initial_state must contain current and voltage"
        raise ValueError(message) from error
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        message = "initial_state must contain two finite values"
        raise ValueError(message)
    if np.any(initial < 0.0):
        message = "initial current and voltage must be nonnegative"
        raise ValueError(message)
    return initial


def _snap_low_voltage(
    time: float,
    voltage: float,
    params: Parameters,
) -> float:
    """Coalesce lower-threshold crossings unresolved in physical time."""
    low = params["voltage_low"]
    # solve_ivp locates event times with absolute and relative tolerances of
    # 4 * machine epsilon. Allow two such localization errors, converted to
    # volts using the zero-current discharge rate, plus voltage roundoff.
    # This is not a controller deadband: only zero-current boundary states
    # are projected, and voltages with resolvable discharge times are retained.
    roundoff = 8.0 * np.finfo(float).eps
    time_tolerance = roundoff * (1.0 + abs(time))
    discharge_rate = low / (params["load_resistance"] * params["capacitance"])
    voltage_tolerance = time_tolerance * discharge_rate + roundoff * low
    return low if abs(voltage - low) <= voltage_tolerance else voltage


def _select_initial_location(
    initial: np.ndarray,
    voltage_high: float,
    voltage_low: float,
    switch_on: Location,
    switch_off: Location,
    zero_current: Location,
) -> Location:
    """Choose a consistent latch and ideal-diode state from the initial state."""
    current, voltage = initial
    if current == 0.0:
        # A blocking ideal diode cannot sustain a negative inductor current.
        # Below the lower threshold the voltage controller immediately drives
        # the switch; otherwise the diode-blocking mode is physically valid.
        return switch_on if voltage <= voltage_low else zero_current
    return switch_off if voltage >= voltage_high else switch_on


def buck_converter(
    source_voltage: float = 24.0,
    inductance: float = 0.00265,
    capacitance: float = 0.0022,
    load_resistance: float = 10.0,
    *,
    inductor_resistance: float = 0.52,
    switch_resistance: float = 0.2,
    voltage_high: float = 12.1,
    voltage_low: float = 11.9,
    initial_state: Sequence[float] | np.ndarray | None = None,
) -> HybridSystem:
    """Create a physical hysteretic nonsynchronous buck converter.

    The state is ``[inductor_current, capacitor_voltage]`` in amperes and
    volts. ``switch_on`` supplies the inductor from the source,
    ``switch_off`` conducts through an ideal freewheel diode, and
    ``zero_current`` represents the diode-blocking boundary. The diode is
    ideal: it has no voltage drop and cannot conduct negative current. The
    transition into ``zero_current`` resets current exactly to zero to make
    that physical boundary robust under numerical event localization. At
    zero current, lower-threshold voltage residuals within event-time
    localization precision are also snapped to the threshold, allowing
    coincident events to settle without an artificial integration step.

    The defaults are component parameters corresponding, within the rounding
    in its published coefficients, to the BuckConverter example in FaMoS-DT:
    https://github.com/TUHH-IES/FaMoS-DT/blob/master/ExampleSystems/BuckConverter/createTrace.m
    This model is derived from the component equations below; it does not
    claim exact equivalence to any XML artifact. For a nonzero initial current,
    the latch starts on below ``voltage_high`` and off at or above it. At zero
    current, it starts in ``zero_current`` above ``voltage_low`` and otherwise
    starts on. A useful nominal simulation horizon is 0.02 seconds.

    Args:
        source_voltage: Constant input supply voltage in volts.
        inductance: Inductance in henries.
        capacitance: Output capacitance in farads.
        load_resistance: Resistive load in ohms.
        inductor_resistance: Inductor series resistance in ohms.
        switch_resistance: Closed-switch resistance in ohms.
        voltage_high: Rising output-voltage threshold that switches off.
        voltage_low: Falling output-voltage threshold that switches on.
        initial_state: Initial ``[inductor_current, capacitor_voltage]``.

    Returns:
        HybridSystem configured for hysteretic buck conversion.
    """
    values = _validate_parameters(
        source_voltage,
        inductance,
        capacitance,
        load_resistance,
        inductor_resistance,
        switch_resistance,
        voltage_high,
        voltage_low,
    )
    initial = _validate_initial_state(initial_state)
    if initial[0] == 0.0:
        initial[1] = _snap_low_voltage(0.0, float(initial[1]), values)

    def flow_switch_on(
        _time: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        current, voltage = state
        return np.array(
            [
                (
                    params["source_voltage"]
                    - (
                        params["inductor_resistance"]
                        + params["switch_resistance"]
                    )
                    * current
                    - voltage
                )
                / params["inductance"],
                (current - voltage / params["load_resistance"])
                / params["capacitance"],
            ],
            dtype=float,
        )

    def flow_switch_off(
        _time: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        current, voltage = state
        return np.array(
            [
                (-params["inductor_resistance"] * current - voltage)
                / params["inductance"],
                (current - voltage / params["load_resistance"])
                / params["capacitance"],
            ],
            dtype=float,
        )

    def flow_zero_current(
        _time: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array(
            [
                0.0,
                -state[1]
                / (params["load_resistance"] * params["capacitance"]),
            ],
            dtype=float,
        )

    def high_voltage_surface(
        _time: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return float(state[1] - params["voltage_high"])

    def low_voltage_surface(
        _time: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return float(state[1] - params["voltage_low"])

    def zero_current_surface(
        _time: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return float(state[0])

    def reset_zero_current(
        time: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        reset_state = np.array(state, dtype=float, copy=True)
        reset_state[0] = 0.0
        reset_state[1] = _snap_low_voltage(time, float(reset_state[1]), params)
        return reset_state

    switch_on = Location(
        ContinuousDynamics(flow_switch_on),
        label="switch_on",
    )
    switch_off = Location(
        ContinuousDynamics(flow_switch_off),
        label="switch_off",
    )
    zero_current = Location(
        ContinuousDynamics(flow_zero_current),
        label="zero_current",
    )

    transitions = [
        Transition(
            source=switch_on,
            target=switch_off,
            event=EventSurface(
                high_voltage_surface,
                direction=CrossingDirection.RISING,
                label="voltage_high",
            ),
        ),
        Transition(
            source=switch_off,
            target=zero_current,
            event=EventSurface(
                zero_current_surface,
                direction=CrossingDirection.FALLING,
                label="current_zero",
            ),
            reset=Reset(reset_zero_current, label="current_zero"),
        ),
        Transition(
            source=switch_off,
            target=switch_on,
            event=EventSurface(
                low_voltage_surface,
                direction=CrossingDirection.FALLING,
                label="voltage_low",
            ),
        ),
        Transition(
            source=zero_current,
            target=switch_on,
            event=EventSurface(
                low_voltage_surface,
                direction=CrossingDirection.FALLING,
                label="voltage_low",
            ),
            # If a current-zero event and the lower voltage threshold coincide,
            # enter the commanded on-state without another integration step.
            entry_policy=SurfaceEntryPolicy.TRIGGER,
        ),
    ]

    initial_location = _select_initial_location(
        initial,
        voltage_high,
        voltage_low,
        switch_on,
        switch_off,
        zero_current,
    )
    return HybridSystem(
        locations=[switch_on, switch_off, zero_current],
        transitions=transitions,
        initial_location=initial_location,
        initial_state=initial.copy(),
        parameters=values,
    )
