from dynamic_hybrid_system import Bound

STL_BOILER_TRANSITIONS = {
    ("cooling", "heating"): [
        {
            "t": Bound(left=1.5099300659909154, right=None),
            "x0": Bound(left=None, right=17.32331263330143),
        },
    ],
    ("heating", "cooling"): [
        {"t": Bound(left=1.006767609317006, right=None)},
        {"x0": Bound(left=19.81311121967754, right=None)},
    ],
}

STL_BOILERNOTIME_TRANSITIONS = {
    ("heating", "cooling"): [
        {"x0": Bound(left=22.035562247795905, right=None)},
    ],
    ("cooling", "heating"): [
        {"x0": Bound(left=None, right=18.032671234629937)},
    ],
}

STL_TANK_TRANSITIONS = {
    ("flow_0", "flow_1"): [
        {
            "x0": Bound(left=None, right=14.716993882672586),
            "x1": Bound(left=0.1791352, right=None),
            "t": Bound(left=1.0051228055434456, right=None),
        },
    ],
    ("flow_1", "flow_0"): [
        {
            "x0": Bound(left=None, right=5.187879880950898),
            "x1": Bound(left=12.611817976371984, right=None),
            "t": Bound(left=1.0024809934190542, right=None),
        },
    ],
    ("flow_2", "flow_0"): [
        {
            "x2": Bound(left=12.747403384931612, right=None),
            "x0": Bound(left=None, right=5.155243396722669),
            "t": Bound(left=1.0006941217427792, right=None),
        },
    ],
    ("flow_1", "flow_2"): [
        {
            "x2": Bound(left=None, right=2.1850785535350634),
            "x1": Bound(left=12.76578803588011, right=None),
            "t": Bound(left=1.004095769468805, right=None),
        },
    ],
    ("flow_0", "flow_2"): [
        {
            "x2": Bound(left=0.056188706, right=None),
            "x0": Bound(left=None, right=14.750400608146629),
            "t": Bound(left=1.0039150378703288, right=None),
        },
    ],
    ("flow_2", "flow_1"): [
        {
            "x2": Bound(left=12.85799192792036, right=None),
            "x1": Bound(left=None, right=2.3230546487890473),
            "t": Bound(left=1.0063595676876442, right=None),
        },
    ],
    ("flow_2", "all_leak"): [
        {
            "x2": Bound(left=None, right=14.421619206475045),
            "t": Bound(left=1.0099885804885567, right=None),
        },
    ],
    ("all_leak", "flow_1"): [
        {"x1": Bound(left=None, right=5.146489460073912)},
    ],
    ("all_leak", "flow_2"): [
        {"x2": Bound(left=None, right=5.066502412140265)},
    ],
    ("all_leak", "flow_0"): [{"x0": Bound(left=None, right=5.0419000820698)}],
}
