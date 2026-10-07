"""Behavioral contract for native trajectories and explicit sampling."""

from dataclasses import FrozenInstanceError

import numpy as np
import polars as pl
import pytest

from flowcean.hybrid import (
    AmbiguousTransitionError,
    ContinuousSegment,
    CrossingDirection,
    Event,
    EventSurface,
    HybridSimulationError,
    HybridSystem,
    HybridTrajectory,
    InvalidEventSurfaceValueError,
    Location,
    SimulationProgressError,
    SurfaceEntryError,
    SurfaceEntryPolicy,
    TrajectoryPoint,
    Transition,
    simulate,
)


def constant_system(value=1.0):
    location = Location(lambda: value, label="steady")
    return HybridSystem([location], [], location, np.array([2.0]))


def chain_system(at=0.5):
    unused = Location(lambda: 100.0, label="same")
    a = Location(lambda: 1.0, label="same")
    b = Location(lambda: 2.0, label="same")
    c = Location(lambda: 3.0, label="same")
    transitions = [
        Transition(
            a,
            b,
            lambda t: t - at,
            lambda state: state + 10,
            entry_policy=SurfaceEntryPolicy.TRIGGER,
        ),
        Transition(
            b,
            c,
            lambda: 0.0,
            lambda state: state + 20,
            entry_policy=SurfaceEntryPolicy.TRIGGER,
        ),
    ]
    return HybridSystem([unused, a, b, c], transitions, a, np.array([0.0]))


def test_execution_has_positive_segments_and_individual_events():
    system = chain_system()
    trajectory = simulate(system, (0, 1))
    assert isinstance(trajectory, HybridTrajectory)
    assert trajectory.system is system
    assert [type(item) for item in trajectory.execution] == [
        ContinuousSegment,
        Event,
        Event,
        ContinuousSegment,
    ]
    assert [event.microstep for event in trajectory.events] == [0, 1]
    assert trajectory.events[0].transition.source is system.locations[1]
    assert trajectory.events[-1].transition.target is system.locations[3]
    assert [segment.t_span for segment in trajectory.segments] == [
        (0, 0.5),
        (0.5, 1),
    ]
    assert trajectory.segments[0].location_time(0.5) == 0.5
    assert trajectory.segments[1].location_time(0.5) == 0
    within = trajectory.evaluate(0.25)
    assert isinstance(within, TrajectoryPoint)
    assert within.location is system.locations[1]
    assert within.location_time == 0.25
    np.testing.assert_allclose(within.state, [0.25])
    before = trajectory.segments[0].evaluate(0.5)
    after = trajectory.evaluate(0.5)
    assert before.location is system.locations[1]
    assert before.location_time == 0.5
    np.testing.assert_allclose(before.state, [0.5])
    assert after.location is system.locations[3]
    assert after.location_time == 0
    np.testing.assert_allclose(after.state, [30.5])
    np.testing.assert_allclose(
        trajectory.segments[1].evaluate(0.5).state, after.state
    )
    assert (
        trajectory.evaluate(np.nextafter(0.5, 0)).location is before.location
    )
    assert trajectory.evaluate(np.nextafter(0.5, 1)).location is after.location
    grid = [0, np.nextafter(0.5, 0), 0.5, 0.5, np.nextafter(0.5, 1), 1]
    frame = trajectory.sample(grid, include_location_label=True)
    assert frame["t"].to_list() == grid
    assert frame["location_id"].to_list() == [1, 1, 3, 3, 3, 3]
    assert frame["location_label"].to_list() == ["same"] * 6
    np.testing.assert_allclose(frame["x0"], [0, 0.5, 30.5, 30.5, 30.5, 32])
    assert frame["location_time"][2] == 0


@pytest.mark.parametrize("time", [0.0, 1.0])
def test_initial_and_final_chains_are_right_continuous(time):
    trajectory = simulate(chain_system(time), (0, 1), initial_location_time=7)
    assert len(trajectory.events) == 2
    assert len(trajectory.segments) == 1
    point = trajectory.evaluate(time)
    np.testing.assert_allclose(point.state, [30 + time])
    assert point.location is trajectory.system.locations[-1]
    assert point.location_time == 0
    frame = trajectory.sample([time], include_derivatives=True)
    assert frame["x0"][0] == pytest.approx(30 + time)
    assert frame["dx0"][0] == 3
    assert frame["location_id"][0] == 3
    assert frame["location_time"][0] == 0
    assert trajectory.events[0].location_time_before == 7 + time
    assert trajectory.events[1].location_time_before == 0
    assert trajectory.initial_location_time == 7
    assert trajectory.initial_state[0] == 0


@pytest.mark.parametrize("chain", [False, True])
def test_zero_span_never_calls_solver(monkeypatch, chain):
    def fail(*args, **kwargs):
        pytest.fail("solver must not run")

    monkeypatch.setattr("flowcean.hybrid.simulator.solve_ivp", fail)
    system = chain_system(0) if chain else constant_system()
    trajectory = simulate(
        system, (0, 0), x0=(value for value in [5]), initial_location_time=4
    )
    assert trajectory.segments == ()
    assert trajectory.initial_state[0] == 5
    assert trajectory.initial_location is system.initial_location
    point = trajectory.evaluate(0)
    np.testing.assert_allclose(point.state, [35 if chain else 5])
    assert point.location is (
        system.locations[-1] if chain else system.initial_location
    )
    assert point.location_time == (0 if chain else 4)
    assert trajectory.sample(dt=1)["x0"].to_list() == [35 if chain else 5]
    assert trajectory.sample([0, 0])["location_time"].to_list() == (
        [0, 0] if chain else [4, 4]
    )
    assert trajectory.sample([]).height == 0


@pytest.mark.parametrize(
    "span", [(1, 0), (np.nan, 1), (0, np.inf), (-np.inf, 1)]
)
def test_bad_intervals_rejected_before_callbacks(span):
    location = Location(lambda: pytest.fail("flow called"))
    transition = Transition(
        location, location, lambda: pytest.fail("surface called")
    )
    system = HybridSystem([location], [transition], location, np.array([0.0]))
    with pytest.raises(ValueError, match="t_span"):
        simulate(system, span)


@pytest.mark.parametrize("time", [-0.1, 1.1, np.nan, np.inf, -np.inf])
def test_evaluate_rejects_invalid_scalar_times(time):
    trajectory = simulate(constant_system(), (0, 1))
    segment = trajectory.segments[0]
    for evaluate in (
        trajectory.evaluate,
        segment.evaluate,
        segment.location_time,
    ):
        with pytest.raises(ValueError, match="time"):
            evaluate(time)


def test_segment_evaluation_does_not_extrapolate_or_evaluate_state_for_age():
    trajectory = simulate(chain_system(), (0, 1))
    for segment, outside in zip(
        trajectory.segments, [0.75, 0.25], strict=True
    ):
        for evaluate in (segment.evaluate, segment.location_time):
            with pytest.raises(ValueError, match="time"):
                evaluate(outside)
    segment = ContinuousSegment(
        trajectory.segments[0].location,
        (0, 1),
        lambda _: pytest.fail("clock query must not evaluate dense solution"),
        trajectory.segments[0]._clock,
        np.array([0, 1]),
    )
    assert segment.location_time(0.25) == 0.25


@pytest.mark.parametrize("time", [0, 0.25, 0.5, 1])
def test_evaluated_points_are_detached_readonly_values(time):
    trajectory = simulate(chain_system(), (0, 1))
    segment = trajectory.segments[0 if time < 0.5 else 1]
    for evaluate in (trajectory.evaluate, segment.evaluate):
        point = evaluate(time)
        expected = point.state.copy()
        with pytest.raises(ValueError, match="read-only"):
            point.state[0] = 99
        with pytest.raises(ValueError, match="flag"):
            point.state.setflags(write=True)
        with pytest.raises(FrozenInstanceError):
            point.location_time = 99  # pyright: ignore[reportAttributeAccessIssue]
        with pytest.raises(FrozenInstanceError):
            point.location = trajectory.system.locations[0]  # pyright: ignore[reportAttributeAccessIssue]
        point.state.setflags(align=False)
        fresh = evaluate(time)
        assert fresh.state.flags.aligned
        np.testing.assert_allclose(fresh.state, expected)
    if time == 0.5:
        assert trajectory.events[-1].state_after.flags.aligned


def test_bound_parameters_source_age_reset_and_target_derivative():
    calls = []

    def flow(parameters, location_time):
        calls.append((parameters["gain"], location_time))
        return parameters["gain"] + location_time

    a = Location(flow, parameters={"gain": 2.0})
    b = Location(flow, parameters={"gain": 4.0})
    unused = Location(flow, parameters={"gain": 9.0})

    def reset(parameters, location_time):
        assert parameters["gain"] == 2
        assert location_time == 5.5
        return np.array([parameters["gain"] + location_time])

    system = HybridSystem(
        [a, b, unused],
        [Transition(a, b, lambda t: t - 0.5, reset)],
        a,
        np.array([0.0]),
        parameters={"gain": 1.0, "extra": 3.0},
    )
    trajectory = simulate(system, (0, 1), initial_location_time=5)
    assert isinstance(system.parameters, dict)
    assert isinstance(a.parameters, dict)
    assert isinstance(b.parameters, dict)
    assert isinstance(unused.parameters, dict)
    system.parameters["extra"] = 100
    a.parameters["gain"] = 200
    b.parameters["gain"] = 400
    unused.parameters["gain"] = 900
    assert trajectory.parameters[a]["extra"] == 3
    assert trajectory.parameters[unused]["gain"] == 9
    with pytest.raises(TypeError):
        trajectory.parameters[a]["gain"] = 0  # pyright: ignore[reportIndexIssue]
    with pytest.raises(TypeError):
        trajectory.parameters[a] = {}  # pyright: ignore[reportIndexIssue]
    calls.clear()
    trajectory.sample(dt=0.1)
    trajectory.sample([0, 0.5, 1])
    for time in [0, 0.25, 0.5, 1]:
        trajectory.evaluate(time)
    for segment in trajectory.segments:
        segment.evaluate(segment.t_span[0])
    assert calls == []
    frame = trajectory.sample([0, 0.5, 1], include_derivatives=True)
    np.testing.assert_allclose(frame["dx0"], [7, 4, 4.5])
    assert calls == [(2, 5), (4, 0), (4, 0.5)]
    assert frame["x0"][1] == 7.5


def test_snapshots_are_readonly_and_detached():
    initial = np.array([0.0])
    reset_buffer = np.array([8.0])
    a = Location(lambda: 0.0)
    b = Location(lambda: 0.0)
    trajectory = simulate(
        HybridSystem(
            [a, b],
            [Transition(a, b, lambda t: t - 0.5, lambda: reset_buffer)],
            a,
            initial,
        ),
        (0, 1),
        x0=initial,
    )
    initial[:] = -10
    reset_buffer[:] = -20
    for snapshot in (
        trajectory.initial_state,
        trajectory.events[0].state_before,
        trajectory.events[0].state_after,
    ):
        with pytest.raises(ValueError, match="read-only"):
            snapshot[0] = 99
        with pytest.raises(ValueError, match="flag to True"):
            snapshot.setflags(write=True)
        copy = snapshot.copy()
        copy[:] = 99
    with pytest.raises(FrozenInstanceError):
        trajectory.initial_location_time = 10  # pyright: ignore[reportAttributeAccessIssue]
    frame = trajectory.sample([0, 0.5, 1])
    np.testing.assert_allclose(frame["x0"], [0, 8, 8])
    writable = frame["x0"].to_numpy(writable=True)
    writable[:] = 100
    np.testing.assert_allclose(trajectory.sample([0, 0.5, 1])["x0"], [0, 8, 8])


def test_all_output_groups_and_callback_free_defaults():
    calls = []

    def stream(t):
        calls.append(t)
        return np.array([t, 2 * t])

    trajectory = simulate(constant_system(), (0, 1), input_stream=stream)
    frame = trajectory.sample(dt=0.4)
    assert calls == []
    assert frame.columns == ["t", "x0", "location_id", "location_time"]
    np.testing.assert_allclose(frame["t"], [0, 0.4, 0.8, 1])
    frame = trajectory.sample(
        (t for t in [0, 0.5, 0.5, 1]),
        include_state=False,
        include_location_id=False,
        include_location_time=False,
        include_location_label=True,
        include_inputs=True,
        include_derivatives=True,
    )
    assert frame.columns == ["t", "location_label", "u0", "u1", "dx0"]
    assert calls == [0, 0.5, 0.5, 1]
    assert frame["u1"].to_list() == [0, 1, 1, 2]
    assert frame["dx0"].to_list() == [1] * 4
    assert trajectory.sample(
        [],
        include_state=False,
        include_location_id=False,
        include_location_time=False,
    ).columns == ["t"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"times": [], "dt": 1},
        {"times": [[0]]},
        {"times": np.empty((0, 2))},
        {"times": [np.nan]},
        {"times": [np.inf]},
        {"times": [0.5, 0]},
        {"times": [-0.1]},
        {"times": [1.1]},
        {"dt": 0},
        {"dt": -1},
        {"dt": np.inf},
        {"dt": np.nan},
    ],
)
def test_invalid_sample_grids(kwargs):
    with pytest.raises(ValueError, match=r"times|dt"):
        simulate(constant_system(), (0, 1)).sample(**kwargs)


def test_nonadvancing_dt_and_short_final_interval():
    trajectory = simulate(constant_system(0), (1e16, 1e16 + 4))
    with pytest.raises(ValueError, match="advance"):
        trajectory.sample(dt=0.1)
    assert simulate(constant_system(), (2, 2.1)).sample(dt=1)[
        "t"
    ].to_list() == [2, 2.1]


def test_input_errors_and_derivative_stream():
    trajectory = simulate(constant_system(), (0, 1))
    for times in ([], [0]):
        with pytest.raises(ValueError, match="requires an input_stream"):
            trajectory.sample(times, include_inputs=True)
    calls = []

    def stream(t):
        calls.append(t)
        return np.array([1.0] if t == 0 else [1.0, 2.0])

    trajectory = simulate(constant_system(), (0, 1), input_stream=stream)
    with pytest.raises(ValueError, match="undeclared input width"):
        trajectory.sample([], include_inputs=True)
    assert calls == []
    with pytest.raises(ValueError, match="dimension changed"):
        trajectory.sample([0, 1], include_inputs=True)
    location = Location(lambda t, input_stream: input_stream(t)[0])
    trajectory = simulate(
        HybridSystem([location], [], location, np.array([0.0])),
        (0, 1),
        input_stream=lambda t: np.array([t]),
    )
    np.testing.assert_allclose(
        trajectory.sample([0, 0.5, 1], include_derivatives=True)["dx0"],
        [0, 0.5, 1],
    )


def test_shared_unhashable_and_positional_callback_forms():
    class Unhashable:
        __hash__ = None  # pyright: ignore[reportAssignmentType]

        def __call__(self, **kwargs):
            return kwargs["location_time"]

    callback = Unhashable()
    location = Location(callback)
    unused = Location(callback)
    trajectory = simulate(
        HybridSystem([location, unused], [], location, np.array([0.0])), (0, 1)
    )
    np.testing.assert_allclose(
        trajectory.sample([0, 0.5, 1], include_derivatives=True)["dx0"],
        [0, 0.5, 1],
    )
    assert trajectory.sample([1])["x0"][0] == pytest.approx(0.5)
    for callback in (
        lambda t, x, p, u: 1.0,
        lambda t, state, parameters, input_stream, /: 1.0,
    ):
        location = Location(callback)
        trajectory = simulate(
            HybridSystem([location], [], location, np.array([0.0])), (0, 1)
        )
        assert trajectory.sample([1], include_derivatives=True)["dx0"][0] == 1

    class Uninspectable:
        @property
        def __signature__(self):
            raise ValueError("not available")

        def __call__(self, t, x, p, u):
            return 2.0

    location = Location(Uninspectable())
    trajectory = simulate(
        HybridSystem([location], [], location, np.array([0.0])), (0, 1)
    )
    assert trajectory.sample([1])["x0"][0] == pytest.approx(2)


@pytest.mark.parametrize(
    ("invalid", "expected"),
    [(False, SurfaceEntryError), (True, InvalidEventSurfaceValueError)],
)
def test_atomic_entry_error_precedence(invalid, expected):
    seen = []
    a, b = Location(lambda: 1), Location(lambda: 1)

    def surface(value):
        def evaluate():
            seen.append(value)
            return value

        return evaluate

    transitions = [
        Transition(a, b, surface(0), entry_policy=policy)
        for policy in (
            SurfaceEntryPolicy.TRIGGER,
            SurfaceEntryPolicy.TRIGGER,
            SurfaceEntryPolicy.ERROR,
        )
    ]
    if invalid:
        transitions.append(Transition(a, b, surface(np.nan)))
    with pytest.raises(expected):
        simulate(HybridSystem([a, b], transitions, a, np.array([0.0])), (0, 1))
    assert len(seen) == len(transitions)


def test_ambiguity_jump_limit_and_no_progress_preserved():
    a, b = Location(lambda: 1), Location(lambda: 1)
    transition = Transition(
        a, b, lambda: 0, entry_policy=SurfaceEntryPolicy.TRIGGER
    )
    with pytest.raises(AmbiguousTransitionError):
        simulate(
            HybridSystem([a, b], [transition, transition], a, np.array([0.0])),
            (0, 1),
        )
    with pytest.raises(HybridSimulationError, match="Maximum"):
        simulate(chain_system(0), (0, 1), max_jumps=1)
    transition = Transition(
        a,
        b,
        EventSurface(
            lambda state: state[0], direction=CrossingDirection.RISING
        ),
        entry_policy=SurfaceEntryPolicy.CONTINUE,
    )
    with pytest.raises(SimulationProgressError):
        simulate(
            HybridSystem([a, b], [transition], a, np.array([0.0])), (0, 1)
        )


def test_actual_initial_overrides_survive_default_state_edits():
    system = chain_system(0)
    override = system.locations[-1]
    trajectory = simulate(
        system,
        (0, 1),
        x0=(x for x in [7]),
        location0=override,
        initial_location_time=2,
    )
    system.initial_state[:] = 500
    assert trajectory.initial_location is override
    assert trajectory.initial_state[0] == 7
    frame = trajectory.sample([0, 1])
    np.testing.assert_allclose(frame["x0"], [7, 10])
    assert frame["location_id"].to_list() == [3, 3]
    assert frame["location_time"].to_list() == [2, 3]


def test_scalar_derivative_is_invalid_for_multistate_system():
    location = Location(lambda: 1.0)
    with pytest.raises(ValueError, match="state dimension"):
        simulate(
            HybridSystem([location], [], location, np.array([0.0, 0.0])),
            (0, 1),
        )


def test_bindings_freeze_unvisited_parameters_before_first_callback():
    target = Location(
        lambda parameters: parameters["rate"], parameters={"rate": 3.0}
    )

    def reset():
        assert isinstance(target.parameters, dict)
        target.parameters["rate"] = 999
        return np.array([2.0])

    source = Location(lambda: 0.0)
    system = HybridSystem(
        [source, target],
        [
            Transition(
                source,
                target,
                lambda: 0.0,
                reset,
                entry_policy=SurfaceEntryPolicy.TRIGGER,
            )
        ],
        source,
        np.array([0.0]),
    )
    trajectory = simulate(system, (0, 1))
    assert trajectory.parameters[target]["rate"] == 3
    frame = trajectory.sample([0, 1], include_derivatives=True)
    np.testing.assert_allclose(frame["x0"], [2, 5])
    assert frame["dx0"].to_list() == [3, 3]


def test_plotting_preserves_location_identity_and_never_connects_jumps():
    import matplotlib.pyplot as plt

    from flowcean.hybrid import (
        plot_locations,
        plot_state_space,
        plot_trajectory,
    )

    system = chain_system()
    trajectory = simulate(system, (0, 1))
    fig, axes = plt.subplots(1, 3)
    try:
        plot_trajectory(
            trajectory, ax=axes[0], show_locations=False, show_events=False
        )
        assert len(axes[0].lines) == 2
        assert [line.get_xdata().tolist()[0] for line in axes[0].lines] == [
            0,
            0.5,
        ]
        assert axes[0].lines[0].get_ydata()[-1] == pytest.approx(0.5)
        assert axes[0].lines[1].get_ydata()[0] == pytest.approx(30.5)
        plot_locations(
            trajectory,
            ax=axes[1],
            location_colors={
                system.locations[1]: "red",
                system.locations[3]: "blue",
            },
        )
        assert len(axes[1].patches) == 2
        assert axes[1].get_legend_handles_labels()[1] == ["same", "same"]
        plot_state_space(
            trajectory, x_dim=0, y_dim=0, ax=axes[2], show_event_points=True
        )
        assert axes[2].get_legend_handles_labels()[1] == [
            "same",
            "same",
            "same",
        ]
    finally:
        plt.close(fig)


def test_dataframe_source_consumers(monkeypatch):
    from flowcean.core import Model
    from flowcean.hybrid.hydra import (
        HyDRAModel,
        HyDRATraceSchema,
        compare_state_traces,
    )

    class DerivativeModel(Model):
        def __init__(self, rate):
            self.rate = rate

        def _predict(self, input_features):
            return input_features.lazy().select(
                (pl.col("drive") * self.rate).alias("velocity")
            )

    schema = HyDRATraceSchema(
        time="clock",
        state=("position",),
        derivative=("velocity",),
        inputs=("drive",),
    )
    model = HyDRAModel(
        [DerivativeModel(1), DerivativeModel(-1)],
        input_features=list(schema.input_features),
        output_features=["velocity"],
        trace_schema=schema,
    )
    selection_times = []

    def select(frame):
        time = frame["clock"][0]
        selection_times.append(time)
        return int(time >= 0.5)

    monkeypatch.setattr(model, "_select_flow_id", select)
    frame = model.simulate(
        (0, 1),
        [0],
        sample_times=[0, 0.5, 1],
        input_stream=lambda t: np.array([2.0]),
    )
    assert frame.columns == ["t", "x0", "flow_id", "flow_time"]
    assert selection_times == [0, 0.5, 1]
    with_inputs = model.simulate(
        (0, 1),
        [0],
        sample_dt=0.5,
        input_stream=lambda t: np.array([2.0]),
        include_inputs=True,
    )
    assert with_inputs["u0"].to_list() == [2, 2, 2]
    np.testing.assert_allclose(frame["x0"], [0, 1, 0], atol=1e-14)
    assert frame["flow_id"].to_list() == [0, 1, 1]
    assert frame["flow_time"].to_list() == [0, 0, 0.5]
    assert compare_state_traces(frame, frame).max_error == 0
    renamed = frame.rename({"x0": "position"})
    assert (
        compare_state_traces(renamed, renamed, state_columns=["position"]).mae
        == 0
    )
    with pytest.raises(ValueError, match="Time grids must match"):
        compare_state_traces(frame, frame.with_columns(pl.col("t") + 1))
    with pytest.raises(ValueError, match="State columns must match"):
        compare_state_traces(frame, frame.with_columns(pl.lit(0).alias("x1")))
