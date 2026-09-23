"""Bounded polynomial approximation of NREL 5-MW rotor aerodynamics."""

import math

import numpy as np

# Total-degree-5 Chebyshev least-squares fit to 575 nodes of ROSCO's
# NREL-5MW performance table (source data licensed Apache-2.0):
# https://github.com/NatLabRockies/ROSCO/blob/974290ec39f7322a9ae83ffa989444bfbea8728b/Examples/Test_Cases/NREL-5MW/Cp_Ct_Cq.NREL5MW.txt
# Order (i, j) is increasing i, then increasing j, for i + j <= 5.
_TERMS = tuple((i, j) for i in range(6) for j in range(6 - i))
_CQ = np.array(
    [
        -0.030423281972146714,
        -0.09850746326681188,
        -0.0298386791504949,
        0.005784526743650399,
        0.0008342624352055414,
        -0.0007755765677714364,
        -0.09044680586298613,
        -0.10576976753348379,
        -0.024413277415392787,
        0.007199796896706968,
        0.0008577703831423318,
        -0.016804956593943765,
        0.00017330954961731614,
        -0.0047047455867383625,
        0.0004721825399604061,
        0.005559404088342448,
        -0.007350929461842459,
        4.7337326076580556e-05,
        -0.0030214581024476782,
        0.0008109387702317535,
        0.0007891774821619475,
    ]
)
_CT = np.array(
    [
        -0.09095537823535974,
        -1.0136650966375296,
        -0.02812596983248772,
        0.050204230667137716,
        0.008301486464804681,
        -0.002675510155512334,
        -0.45704483267590784,
        -1.1116994007541132,
        0.0018543393158432526,
        0.06804654000038583,
        0.014271774389670245,
        -0.1881466270807809,
        -0.09762697557352738,
        0.00745269570763879,
        0.018968582333213992,
        0.022345174180857935,
        0.0011192185651778832,
        -0.0019571722902429565,
        -0.003791117180479281,
        -0.011406756095895382,
        -0.0023675888874557607,
    ]
)


def aerodynamic_coefficients(
    tip_speed_ratio: float, pitch_rad: float
) -> tuple[float, float]:
    """Return (Cq, Ct) within lambda 2.5..14.5 and pitch -2..20 degrees.

    Normalized coordinates are x=(2*lambda-17)/12 and
    y=(2*pitch_degrees-18)/22. Signs are retained; negative Cq brakes the rotor.
    """
    if not math.isfinite(tip_speed_ratio) or not math.isfinite(pitch_rad):
        raise ValueError("tip-speed ratio and pitch must be finite")
    pitch_deg = math.degrees(pitch_rad)
    # Degree/radian conversion can land one or two ulps beyond an endpoint.
    if (
        not 2.5 - 8 * math.ulp(2.5)
        <= tip_speed_ratio
        <= 14.5 + 8 * math.ulp(14.5)
    ):
        raise ValueError(
            "tip-speed ratio outside aerodynamic domain [2.5, 14.5]"
        )
    if not -2.0 - 8 * math.ulp(2.0) <= pitch_deg <= 20.0 + 8 * math.ulp(20.0):
        raise ValueError("pitch outside aerodynamic domain [-2, 20] degrees")
    x = min(1.0, max(-1.0, (2.0 * tip_speed_ratio - 17.0) / 12.0))
    y = min(1.0, max(-1.0, (2.0 * pitch_deg - 18.0) / 22.0))
    tx = np.polynomial.chebyshev.chebvander(x, 5).ravel()
    ty = np.polynomial.chebyshev.chebvander(y, 5).ravel()
    basis = np.array([tx[i] * ty[j] for i, j in _TERMS])
    return float(_CQ @ basis), float(_CT @ basis)
