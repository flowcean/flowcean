from dynamic_hybrid_system import Bound

STL_BOILER_TRANSITIONS = {
    ("cooling", "heating"): [
        {
            "t": Bound(left=1.509, right=None),
            "x0": Bound(left=None, right=20.026),
        },
    ],
    ("heating", "cooling"): [
        {"t": Bound(left=1.01, right=None)},
        {"x0": Bound(left=20.0, right=None)},
    ],
}

STL_BOILERNOTIME_TRANSITIONS = {
    ("heating", "cooling"): [{"x0": Bound(left=22.002, right=None)}],
    ("cooling", "heating"): [{"x0": Bound(left=None, right=18.002)}],
}

STL_TANK_TRANSITIONS = {
    ("flow_1", "flow_0"): [
        {
            "x0": Bound(left=None, right=4.78),
            "t": Bound(left=0.5, right=None),
            "x1": Bound(left=9.778, right=None),
        },
    ],
    ("flow_0", "flow_2"): [
        {
            "x2": Bound(left=None, right=4.027),
            "x0": Bound(left=9.774, right=None),
            "t": Bound(left=0.501, right=None),
        },
    ],
    ("flow_1", "flow_2"): [
        {
            "x2": Bound(left=None, right=4.017),
            "t": Bound(left=0.501, right=None),
            "x1": Bound(left=9.786, right=None),
        },
    ],
    ("flow_2", "flow_0"): [
        {
            "x0": Bound(left=None, right=4.973),
            "x2": Bound(left=9.746, right=None),
            "t": Bound(left=0.5, right=None),
        },
    ],
    ("flow_0", "all_leak"): [
        {
            "x0": Bound(left=9.842, right=None),
            "t": Bound(left=0.501, right=None),
        },
    ],
    ("all_leak", "flow_1"): [{"x1": Bound(left=None, right=5.074)}],
    ("flow_2", "all_leak"): [
        {
            "x2": Bound(left=None, right=11.108),
            "t": Bound(left=0.5, right=None),
        },
    ],
    ("all_leak", "flow_2"): [{"x2": Bound(left=None, right=5.031)}],
    ("all_leak", "flow_0"): [{"x0": Bound(left=None, right=5.042)}],
    ("flow_2", "flow_1"): [
        {
            "x2": Bound(left=9.939, right=None),
            "t": Bound(left=0.401, right=None),
            "x1": Bound(left=None, right=4.557),
        },
    ],
    ("flow_0", "flow_1"): [
        {
            "x0": Bound(left=9.869, right=None),
            "t": Bound(left=0.5, right=None),
            "x1": Bound(left=None, right=4.504),
        },
    ],
}
