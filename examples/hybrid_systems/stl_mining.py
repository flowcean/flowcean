from dynamic_hybrid_system import Bound

STL_BOILER_TRANSITIONS = {
    ("cooling", "heating"): [
        {
            "x0": Bound(left=None, right=20.002998748217117),
            "t": Bound(left=1.5098182821386332, right=None),
        },
    ],
    ("heating", "cooling"): [
        {"t": Bound(left=1.008977239421488, right=None)},
        {"x0": Bound(left=20.00015945738306, right=None)},
    ],
}

STL_BOILERNOTIME_TRANSITIONS = {
    ("heating", "cooling"): [
        {"x0": Bound(left=22.001578094079846, right=None)},
    ],
    ("cooling", "heating"): [
        {"x0": Bound(left=None, right=18.00151967428954)},
    ],
}

STL_TANK_TRANSITIONS = {
    ("flow_2", "all_leak"): [
        {
            "x2": Bound(left=None, right=14.4512353337818),
            "t": Bound(left=1.0099984961565491, right=None),
        },
    ],
    ("flow_2", "flow_1"): [
        {
            "x2": Bound(left=12.99367224062909, right=None),
            "t": Bound(left=1.0051409446468613, right=None),
            "x1": Bound(left=None, right=2.3973562848676453),
        },
    ],
    ("all_leak", "flow_1"): [
        {"x1": Bound(left=None, right=5.102071903064735)},
    ],
    ("flow_1", "flow_2"): [
        {
            "x1": Bound(left=12.666840121840305, right=None),
            "x2": Bound(left=None, right=2.01028058475264),
            "t": Bound(left=1.0058590395229354, right=None),
        },
    ],
    ("all_leak", "flow_2"): [
        {"x2": Bound(left=None, right=5.066374375075659)},
    ],
    ("flow_2", "flow_0"): [
        {
            "x2": Bound(left=11.81077443019694, right=None),
            "t": Bound(left=1.0004827758384582, right=None),
            "x0": Bound(left=None, right=4.960610593460949),
        },
    ],
    ("flow_1", "flow_0"): [
        {
            "x0": Bound(left=1.7675438438934794, right=None),
            "t": Bound(left=1.0099607370292447, right=None),
            "x1": Bound(left=None, right=14.701386103378294),
        },
    ],
    ("all_leak", "flow_0"): [
        {"x0": Bound(left=None, right=5.042491415035765)},
    ],
    ("flow_0", "flow_2"): [
        {
            "x0": Bound(left=12.279978715429017, right=None),
            "x2": Bound(left=None, right=4.777410193758175),
            "t": Bound(left=1.0033787886337568, right=None),
        },
    ],
    ("flow_0", "flow_1"): [
        {
            "x0": Bound(left=11.56109109257665, right=None),
            "t": Bound(left=1.002330274867684, right=None),
            "x1": Bound(left=None, right=4.970656345527218),
        },
    ],
}
