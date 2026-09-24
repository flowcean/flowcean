"""Reusable hybrid-system benchmark factories."""

from .bouncing_ball import bouncing_ball
from .buck_converter import buck_converter
from .hybrid_oscillator import hybrid_oscillator
from .impact_oscillator import impact_oscillator
from .mode_cycle import mode_cycle
from .pid_controlled_plant import pid_controlled_plant
from .piecewise_affine import piecewise_affine
from .relay_integrator import relay_integrator
from .switched_linear import switched_linear
from .tank_valves import tank_valves
from .thermostat import thermostat
from .time_forced_switch import time_forced_switch
from .time_varying_event_surface import time_varying_event_surface
from .wind_turbine import wind_turbine, wind_turbine_power

__all__ = [
    "bouncing_ball",
    "buck_converter",
    "hybrid_oscillator",
    "impact_oscillator",
    "mode_cycle",
    "pid_controlled_plant",
    "piecewise_affine",
    "relay_integrator",
    "switched_linear",
    "tank_valves",
    "thermostat",
    "time_forced_switch",
    "time_varying_event_surface",
    "wind_turbine",
    "wind_turbine_power",
]
