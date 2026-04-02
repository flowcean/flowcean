"""Time-varying guard benchmark."""

from collections.abc import Mapping

import numpy as np

from flowcean.ode import Guard, HybridSystem, Mode, Transition

J_H     = 115926   # kgm^2                     Hub Inertia About Shaft Axis
J_B     = 11776047 # kgm^2                     Second Mass Moment of Inertia (w.r.t. Root)
J_G     = 534.116   # kgm^2                     Generator Inertia About High-Speed Shaft
xt0 = 0.7452
omega0 = 1.2671
theta0 = 0.143
mas=[]
fas=[]
ts_ratios = []

class polymodel:
    def __init__(self, modelterms, coefficients, parametervar, parameterstd, r2, adjustedr2, rmse):
        self.modelterms = modelterms
        self.coefficients = coefficients
        self.parametervar = parametervar
        self.parameterstd = parameterstd
        self.r2 = r2
        self.adjustedr2 = adjustedr2
        self.rmse = rmse
### from Schuler et al.
cP_modelrm = polymodel(
    modelterms = np.array([[3,2,2,1,1,1,0,0,0,0],[0,1,0,2,1,0,3,2,1,0]]).T,
    coefficients = [-4.645506812711324e-04,-0.074780487295004,-0.002458175859067,-0.372740535667247,0.071114752590880,0.175932463831508,3.898087853290869,-3.514363457408013,1.599390296654579,-0.404129012120541],
    parametervar = [7.501388136040788e-10,6.679452782066278e-08,1.539190796429710e-07,7.258526274162107e-06,7.617687742264648e-06,3.419686114249399e-06,5.452288307592214e-04,4.709605455918789e-04,7.816513374422656e-05,8.319146166316219e-06],
    parameterstd = [2.738866213607519e-05,2.584463732008302e-04,3.923252217777631e-04,0.002694165227703,0.002760015895292,0.001849239333956,0.023350135561902,0.021701625413592,0.008841104780751,0.002884293009789],
    r2 = 0.9885,
    adjustedr2 = 0.9885,
    rmse = 0.97
)
cT_modelrm = polymodel(
    modelterms = np.array([[3,2,2,1,1,1,0,0,0,0],[0,1,0,2,1,0,3,2,1,0]]).T,
    coefficients = [0.001682233589763,-0.005151180495985,-0.047219724810382,1.164758256848169,-1.525761648555954,0.553206918519747,6.914278352507001,-10.967542962838312,6.000636278248273,-0.980172788618940],
    parametervar = [7.673725840212874e-09,6.832907254320090e-07,1.574552332631929e-06,7.425284443593678e-05,7.792697326196755e-05,3.498250353724942e-05,0.005577549770187,0.004817804442174,7.996090799980241e-04,8.510271132119102e-05],
    parameterstd = [8.759980502382910e-05,8.266140123612768e-04,0.001254811672177,0.008617009019140,0.008827625573277,0.005914600877257,0.074682995187575,0.069410405863778,0.028277359848437,0.009225113079046],
    r2 = 0.9924,
    adjustedr2 = 0.9924,
    rmse = 0.0311
)
###

def turbine(
    damping_pitch: float = 0.7,
    freq_pitch: float = 6.2832,
    rho: float = 1.225,
    rotor_radius: float = 63.0,
    gearbox_ratio: float = 1/97,
    cTe: float = 19325.0,
    mTe: float = 480650.0,
    kTe: float = 1942400.0 ,
    xT0: float = xt0 ,
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
    
    def polyvaln(model, indepvars):
        n, p = indepvars.shape # shape(1,2)
        if n == 1 and model.modelterms.shape[1] == 1: # shape(10,2)
            indepvars = indepvars.T
            n, p = indepvars.shape
        if model.modelterms.shape[1] != p:
            raise ValueError("Number of independent variables does not match model terms.")

        nt = model.modelterms.shape[0] 
        ypred = np.zeros((n,1))
        for i in range (nt):
            t = np.ones((n,1))
            for j in range(p):
                t = t * indepvars[:,j]**model.modelterms[i,j]
            ypred = ypred + model.coefficients[i] * t
        return ypred

    def tipspeed_ratio(omega, rotor_radius, wind_speed, dx):
        ts_ratio = omega * rotor_radius / (wind_speed - dx)
        return ts_ratio

    def Ma(rho, rotor_radius, theta, wind_speed, omega, dx):
        # calculate aerodynamic torque
        tipspeedratio = tipspeed_ratio(omega, rotor_radius, wind_speed, dx)
        ma = 1/2 * rho * np.pi * rotor_radius**3 * cp(tipspeedratio, theta) / tipspeedratio * (wind_speed - dx)**2
        return ma
    
    def Fa(rho, rotor_radius, theta, wind_speed, omega, dx):
        # calculate aerodynamic force
        tipspeedratio = tipspeed_ratio(omega, rotor_radius, wind_speed, dx)
        fa = 1/2 * rho * np.pi * rotor_radius**2 * c_thrust(tipspeedratio, theta) * (wind_speed - dx)**2
        return fa
    
    def cp(tipspeed_ratio_val, theta):
        a = polyvaln(cP_modelrm, np.array([[tipspeed_ratio_val, theta]])) #shape (1,2)
        return np.squeeze(a)
    
    def c_thrust(tipspeed_ratio_val, theta):
        a = polyvaln(cT_modelrm, np.array([[tipspeed_ratio_val, theta]])) #shape (1,2)
        return np.squeeze(a)

    def flow_(
        time: float,
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
        initial_state = np.array([omega0, xt0, 0, theta0, 0], dtype=float)
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
