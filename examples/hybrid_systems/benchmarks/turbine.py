"""Time-varying guard benchmark."""

from collections.abc import Mapping

import numpy as np

from flowcean.ode import Guard, HybridSystem, Mode, Transition

J_H     = 115926   # kgm^2                     Hub Inertia About Shaft Axis
J_B     = 11776047 # kgm^2                     Second Mass Moment of Inertia (w.r.t. Root)
J_G     = 534.116   # kgm^2                     Generator Inertia About High-Speed Shaft

def turbine(
    damping_pitch: float = 0.7,
    freq_pitch: float = 6.2832,
    rho: float = 1.225,
    rotor_radius: float = 63.0,
    gearbox_ratio: float = 1/97,
    cTe: float = 19325.0,
    mTe: float = 480650.0,
    kTe: float = 1942400.0 ,
    xT0: float = 0.7452 ,
    # paper != matlab implementation (*vs/) (gearbox_ratio**2),
    inertia: float = J_H + 3*J_B + J_G *((1/97)**2), # gearbox_ratio: float = 1/97,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a system with time-varying guard thresholds.

    Args:
        frequency: Frequency of the guard oscillation.
        amplitude: Amplitude of the guard oscillation.
        hysteresis: Guard hysteresis width.
        drift: Drift magnitude per mode.
        damping: Damping on the second state.
        initial_state: Optional initial state.

    Returns:
        HybridSystem with time-dependent guard surfaces.
    """
    def tipspeed_ratio(omega, rotor_radius, wind_speed, dx):
        return omega * rotor_radius / (wind_speed - dx)

    def Ma(rho, rotor_radius, theta, wind_speed, omega, dx):
        # calculate aerodynamic torque
        tipspeedratio = tipspeed_ratio(omega, rotor_radius, wind_speed, dx)
        return 1/2 * rho * np.pi * rotor_radius**3 * cp(tipspeedratio, theta) / tipspeedratio * (wind_speed - dx)**2
    
    def Fa(rho, rotor_radius, theta, wind_speed, omega, dx):
        # calculate aerodynamic force
        tipspeedratio = tipspeed_ratio(omega, rotor_radius, wind_speed, dx)
        return 1/2 * rho * np.pi * rotor_radius**2 * c_thrust(tipspeedratio, theta) * (wind_speed - dx)**2
    
    def cp(tipspeed_ratio_val, theta):
        return 0.482
    
    def c_thrust(tipspeed_ratio_val, theta):
        return 0

    def flow_(
        _: float,
        state: np.ndarray, # omega, x, dx, theta, dtheta, target_torque, target_pitch
        params: Mapping[str, float],
    ) -> np.ndarray:
        omega, x, dx, theta, dtheta = state
        # omega, x, dx, theta, dtheta, target_torque, target_pitch = state
        wind_speed = 10
        target_torque = 1000
        target_pitch = 0.1
        return np.array(
            [
                # cp normally function of lambda, theta
                ((Ma(params["rho"], params["rotor_radius"], theta, wind_speed, omega, dx) - target_torque/(params["gearbox_ratio"])) / params["inertia"]), # omega
                dx, # x
                (Fa(params["rho"], params["rotor_radius"], theta, wind_speed, omega, dx) - params["cTe"] * dx - params["kTe"]*(x - params["xT0"])) / params["mTe"], # dx
                dtheta, # theta
                -2*params["damping_pitch"] * dtheta - params["freq_pitch"]**2 * (theta - target_pitch), # dtheta (3) in Schuler et al.
                # 0, # target_torque
                # 0, # target_pitch
            ],
            dtype=float,
        )

   
    left = Mode(name="left", flow=flow_)

   

    if initial_state is None:
        initial_state = np.array([122.0, 0, 0, 0.143, 0], dtype=float)
        # initial_state = np.array([120, 0, 0, 0.1, 0, 1000, 0.1], dtype=float)

    return HybridSystem(
        modes={"left": left},
        transitions=[],
        initial_mode="left",
        initial_state=initial_state,
        params={
            "damping_pitch": damping_pitch,
            "freq_pitch": freq_pitch,
            "rho": rho,
            "rotor_radius": rotor_radius,
            "gearbox_ratio": gearbox_ratio,
            "cTe": cTe,
            "mTe": mTe,
            "kTe": kTe,
            "xT0": xT0,
            "inertia": inertia,
        },
    )
