---
icon: lucide/workflow
---

# Hybrid Systems

## From Observed Behavior to a Hybrid Model

The plot below shows the temperature in a room. The room warms up, cools down, and warms up again.

<figure class="hybrid-figure">

--8<-- "docs/assets/hybrid_systems/thermostat-time-series.svg"

<figcaption>Example temperature data, simulated with the model introduced below.</figcaption>

</figure>

At 22 degrees, the room might be getting warmer or cooler. Temperature alone doesn't tell us which: whether the heating is on or off also matters.

A useful way to model this behavior is to describe the warming and cooling separately, reuse those descriptions whenever they recur, and specify what causes the switches between them.

We represent the temperature as the **continuous state**. A **[location][flowcean.hybrid.Location]** records the situation the system is currently in. Each location selects a **[flow][flowcean.hybrid.Flow]**, an equation describing the continuous state's rate of change. A **[transition][flowcean.hybrid.Transition]** describes a discrete step, such as switching the active location.

These components form a **hybrid automaton**: a model combining continuous evolution with discrete changes. In the diagram, $T$ denotes temperature and $\dot T$ its rate of change.

<figure class="hybrid-figure hybrid-automaton">

--8<-- "docs/assets/hybrid_systems/thermostat-schematic.svg"

<figcaption>Ellipses contain the locations and their flows. The incoming arrow marks the initial heating location.</figcaption>

</figure>

The automaton describes the rules of the model. The time series shows one behavior those rules can produce.

## Describe Continuous Behavior

For the room, we track temperature. A model of a moving object could track both position and velocity. The continuous state collects the values the model needs to describe how things change.

A flow tells us how quickly each state value changes. For temperature, that could be a rate in degrees per minute. These rates are called **derivatives**. The flow supplies the rates; Flowcean uses them to calculate how the state evolves over time.

Each location selects a flow, but several locations can share the same one. For example, a heater in standby and a heater disabled by a fault both leave the room cooling. The cooling equation can be the same, while the conditions for restarting the heater differ. Locations distinguish these situations even when their continuous behavior is identical.

We also distinguish two kinds of information supplied to the model:

- **Parameters** configure it, such as the heater's power.
- **Inputs** describe externally supplied signals, such as the desired room temperature, which can change during a run.

Inputs and parameters can affect both continuous motion and the conditions for switching locations.

## Describe Discrete Changes

### Switching Between Locations

A transition describes a discrete change. In our example, one transition switches the heating off and another switches it back on.

The conditions beside the arrows are called **guards**. The guard $T\ge23$ marks when to switch from heating to cooling.

For simulation, Flowcean detects the boundary of that condition using a function that becomes zero there:

$$
g(T)=T-23.
$$

Below 23, this function is negative; at 23, it is zero; above 23, it is positive. The boundary where the function is zero is the **[event surface][flowcean.hybrid.EventSurface]**.

The **[crossing direction][flowcean.hybrid.CrossingDirection]** distinguishes rising crossings, from negative toward positive, from falling crossings, from positive toward negative. This refers to the motion under the flow before the transition. For heating, choose a rising crossing at 23. For the return transition, use $g(T)=T-21$ with a falling crossing.

### Changing the State Instantly

Switching the heater off changes how quickly the room cools, but its temperature stays continuous.

Other transitions change a state value instantly. When a bouncing ball hits the ground, its velocity changes from downward to upward. A **[reset][flowcean.hybrid.Reset]** describes that change.

A reset can happen without changing location: the ball can remain in a single "flight" location, with each impact represented by a transition back to it. See the [bouncing-ball example](../examples/hybrid_systems.md#bouncing-ball).

### Switching After Some Time

Some changes depend on how long the system has been doing something. For example, a pump might run for five minutes before switching off.

**Location residence time** measures the time since entering the current location. Calling that time $\tau$, the boundary $\tau-5=0$ describes the five-minute timer.

Every transition starts a new visit and resets residence time to zero, including a transition back to the same location. The [timed-switch example](../examples/hybrid_systems.md#time-forced-switch) uses this kind of condition.

### Delaying a Transition After Detection

A crossing can schedule a switch for later. For example, an inlet valve may remain open for a configured closing delay after the water reaches its threshold:

```python
close_valve = Transition(
    source=filling,
    target=closed,
    event_surface=upper_boundary,
    delay=closing_delay,
)
```

`delay` accepts either a fixed, finite, nonnegative scalar duration in the model's time units or a pure, deterministic callback returning one. It defaults to zero, which takes the transition immediately. A positive delay keeps the source location, flow, and residence clock active until execution. The reset, if present, uses the state, time, inputs, parameters, and source residence time at execution.

For a delay that depends on conditions when the crossing is detected, use a callback:

```python
def closing_delay(t, parameters, input_stream):
    return parameters["closing_time"] * (1 + abs(input_stream(t)[0]))


close_valve = Transition(
    source=filling,
    target=closed,
    event_surface=upper_boundary,
    delay=closing_delay,
)
```

Delay callbacks use the same [callback arguments and signatures][flowcean.hybrid.DelayFunction] as flows and resets: any named subset of `t`, `state`, `parameters`, `input_stream`, and `location_time`, or the four-positional form. They receive the source context at detection, including location-local parameter overrides and the current visit's residence time. Each accepted detection samples the duration once and freezes that occurrence's deadline at `detection_time + delay`. Fixed delays are validated at construction; invalid callback results raise `ValueError` when detected, not when constructing the transition.

Choose how repeated crossings affect a transition with its keyword-only `scheduling_policy`, using [TransitionSchedulingPolicy][flowcean.hybrid.TransitionSchedulingPolicy]:

- `FIRST_DETECTION` (the default): the first detection fixes the occurrence for the entire source visit. Further crossings do not reevaluate the delay, even if another transition currently has an earlier deadline. A losing candidate is still remembered.
- `LATEST_DETECTION`: every direction-qualified crossing replaces this transition's pending occurrence. Its new deadline can be earlier or later than the previous one.
- `EACH_DETECTION`: every direction-qualified crossing samples an independent occurrence. The earliest deadline wins. Equal deadlines belonging to this same transition coalesce, retaining the earliest `detection_time`.

For example, add `scheduling_policy=TransitionSchedulingPolicy.LATEST_DETECTION` to `close_valve` to restart scheduling at every accepted crossing. Changes in state or inputs between crossings do not by themselves resample a delay. The earliest deadline among all outgoing transitions executes; another candidate can become the winner if a latest detection postpones the current winner. Every exit cancels all pending occurrences and crossing history, including a self-transition that starts a fresh visit.

At an exactly equal numerical time, all direction-qualified detections are processed **before execution**. Thus a latest detection at its own deadline can postpone the transition, even beyond the simulation endpoint. After processing those detections, distinct transitions with equal earliest deadlines raise `AmbiguousTransitionError` when execution is due. There is no declaration-order priority or tolerance-based grouping of nearby times.

Crossings are found from integration-step endpoint signs. Exact numerical equality does not guarantee that mathematically simultaneous roots will be reported together. Multiple crossings inside one step can be missed, and tangency detection depends on the step endpoints. Set `max_step` to resolve the time scale of your surfaces; it does not remove these numerical limitations.

On entry, `SurfaceEntryPolicy.TRIGGER` detects the transition immediately and starts its delay, evaluating a delay callback with the entry context. Remaining on that zero surface does not repeatedly detect it. Each run starts with no pending occurrences, including when an initial residence time is supplied. A deadline at the simulation endpoint executes unless same-time detection replaces it; later deadlines do not. Only executed transitions count toward `max_jumps`.

The [delayed valve example](../examples/hybrid_systems.md#delayed-valve-closure) shows how continued filling during the delay raises the final water height.

## Construct and Simulate a Model

Flowcean stores the continuous state in a NumPy array. Our room has one state value, so `state[0]` is its temperature. Use an array of corresponding rates of change for the flow output.

Start by defining the flows and assigning them to locations:

```python
import numpy as np

from flowcean.hybrid import (
    CrossingDirection,
    EventSurface,
    Flow,
    HybridSystem,
    Location,
    Transition,
    simulate,
)

heating_flow = Flow(
    fn=lambda state: np.array([5.0 - 0.3 * (state[0] - 20.0)]),
)
cooling_flow = Flow(
    fn=lambda state: np.array([-0.3 * (state[0] - 20.0)]),
)
heating = Location(flow=heating_flow, label="heating")
cooling = Location(flow=cooling_flow, label="cooling")
```

Next, define the switching boundaries and connect the locations in a [`HybridSystem`][flowcean.hybrid.HybridSystem]. The initial conditions specify both the temperature and whether heating is active:

```python
upper_boundary = EventSurface(
    fn=lambda state: state[0] - 23.0,
    direction=CrossingDirection.RISING,
)
lower_boundary = EventSurface(
    fn=lambda state: state[0] - 21.0,
    direction=CrossingDirection.FALLING,
)

system = HybridSystem(
    locations=[heating, cooling],
    transitions=[
        Transition(heating, cooling, upper_boundary),
        Transition(cooling, heating, lower_boundary),
    ],
    initial_location=heating,
    initial_state=np.array([20.0]),
)
```

Now choose the time interval and run [`simulate`][flowcean.hybrid.simulate]:

```python
trajectory = simulate(system, t_span=(0.0, 10.0))
```

Use the same time unit throughout the model. If the flows describe temperature change per minute, this interval represents ten minutes.

The functions above request only `state`. Flowcean supplies [callback arguments][flowcean.hybrid.FlowFunction] by name; functions can also request `t`, `parameters`, `input_stream`, and `location_time` when they need them.

### Choose the Starting Situation

Choose the initial state and location together. Starting at 24 degrees in the heating location does not automatically select cooling just because $T\ge23$: Flowcean detects boundary crossings, rather than checking whether an inequality is already true.

If the system has already spent time in its initial location, set `initial_location_time` to preserve that elapsed time.

### Decide What Happens on a Boundary

A run can start exactly on an event surface, or arrive there after a transition. Set the transition's `entry_policy`, using [`SurfaceEntryPolicy`][flowcean.hybrid.SurfaceEntryPolicy], to choose what happens:

| Policy            | Behavior                                          |
| ----------------- | ------------------------------------------------- |
| `ERROR` (default) | Reject the entry.                                 |
| `TRIGGER`         | Detect immediately; execute now or after `delay`. |
| `CONTINUE`        | Begin continuous motion from the boundary.        |

`CONTINUE` leaves the surface active. Choose a crossing direction that allows the initial departure. For a bouncing ball leaving the ground upward, the impact surface should detect **falling** crossings. Otherwise, the solver can rediscover the event at the starting time and report a failure to make progress.

### Let Immediate Transitions Settle

An immediate transition can lead to another at the same time. Such a chain must eventually allow continuous motion to resume.

On entry, at most one outgoing zero-delay transition may request an immediate jump. Multiple requests are an error. More generally, make competing transitions unambiguous rather than treating numerical detection order as a priority rule.

`max_jumps` limits the total number of transitions, including ordinary switches and immediate chains. Increase it for longer runs with many legitimate switches; increasing it does not resolve a loop of immediate transitions.

### Choose Numerical Accuracy

The solver chooses its own integration steps. `rtol` and `atol` set error tolerances, while `max_step` limits the step size.

An event surface can cross zero and return within one step, so crossings can be missed. For rapidly changing conditions, reduce `max_step` and check whether the results and switching times remain stable as you refine the settings.

Reading more values from the completed trajectory later does not improve its numerical accuracy.

### Keep Model Functions Repeatable

For the same arguments, callbacks must return the same results. The solver can evaluate them repeatedly and revisit earlier times. Look up input values using the requested time, rather than advancing a counter or consuming the next reading on each call.

Parameter values are captured at the start of a run, with location-specific settings overriding system settings. Input sources are not copied: keep them available and unchanged if you want to evaluate inputs or derivatives from that run later.

## Understand and Use the Result

### Inspect the Run

`simulate` returns a **[trajectory][flowcean.hybrid.HybridTrajectory]**, a record of one run. It contains:

- **[Continuous segments][flowcean.hybrid.ContinuousSegment]**, describing motion in one location over a time interval.
- **[Events][flowcean.hybrid.Event]**, recording when transitions were taken, including the state before and after each change.

A transition belongs to the model; an event records one occurrence of it. The same location can appear in several segments as the run returns to it.

To plot the run, use [`plot_trajectory`][flowcean.hybrid.plot_trajectory]:

```python
from flowcean.hybrid import plot_trajectory

plot_trajectory(trajectory, show=True)
```

To read the temperature and active location at a particular time, use [`evaluate`][flowcean.hybrid.HybridTrajectory.evaluate]:

```python
point = trajectory.evaluate(2.0)
print(point.state[0], point.location.label)
```

You can also inspect the switches directly:

```python
for event in trajectory.events:
    print(event.detection_time, event.time, event.transition.target.label)
```

For a delayed transition, `event.detection_time` records the detection that scheduled the executed occurrence, while `event.time` records execution. This is the first detection for `FIRST_DETECTION`, the most recent for `LATEST_DETECTION`, or the detection with the winning deadline for `EACH_DETECTION` (earliest detection on a same-transition tie). Detection can occur on entry. The two times are equal for zero-delay transitions. The trajectory records executed transitions, not every detection.

Sampling during the delay still reports the source location and its continuing residence time. Event markers in plots mark execution.

### Choose When to Observe

A **sampled trace** is a table of observations from the trajectory. Choose the observation times to suit your analysis with [`sample`][flowcean.hybrid.HybridTrajectory.sample]:

```python
samples = trajectory.sample(dt=0.1)
```

This returns a Polars DataFrame with a row every 0.1 time units. Alternatively, supply `times=[0.0, 2.0, 5.0, 10.0]` instead of `dt` to request particular times.

The table includes time, temperature in `x0`, the active location's `location_id`, and its `location_time`. Location IDs follow the order in `system.locations`: here, 0 means heating and 1 means cooling.

### Interpret Values at Switches

At an exact event time, evaluation and sampling return the state **after** the transition. If several transitions happen at that time, they return the result after the whole chain.

To inspect an individual reset, use the event's `state_before` and `state_after`.

A coarse observation grid can miss a short visit to a location. Keep the trajectory when investigating switching behavior: its event records remain available independently of the observation times you choose.

## Move from Simulation to Learning

### Prepare Learning Data

Use **[HyDRA][flowcean.hybrid.hydra.HyDRALearner]** to discover flow models from recorded or simulated observations.

For simulated data, request the rates of change alongside the state:

```python
learning_data = trajectory.sample(
    dt=0.02,
    include_derivatives=True,
)
```

Here, `x0` contains temperature and `dx0` contains the rate returned by the active flow. Choose enough observations to capture the brief behaviors you want to learn.

If the run uses external inputs, include them with `include_inputs=True`. For measured recordings, derivatives must instead be supplied or estimated, taking care not to treat reset jumps as continuous rates of change.

### Discover Shared Flows

HyDRA groups observations that share a continuous behavior and fits a model for each group. Here the target is the temperature derivative `dx0`. You can also learn another scalar target, such as circuit current from voltage.

Pass each independent trace as a Polars DataFrame and choose the input columns and one output column. The factory creates a fresh batch learner and regressor for each fit; later fits must not change previously returned models or their prediction-affecting state:

```python
from pysr import PySRRegressor
from flowcean.pysr import PySRLearner
from flowcean.hybrid.hydra import HyDRALearner

learner = HyDRALearner(
    regressor_factory=lambda: PySRLearner(PySRRegressor(niterations=10)),
    threshold=0.01,
)
frames = [learning_data]  # Add other independent sampled traces here.
result = learner.learn(
    frames,
    input_features=["t", "x0"],
    output_features=["dx0"],
)
```

HyDRA fits a candidate on growing windows within the first unassigned segment, groups matching observations across traces with unassigned observations, and refits the flow on those accepted observations. Grouping predicts and validates each such trace in full to preserve batch context. `start_width` sets the initial window size and `step_width` sets how many observations to add.

Fitting and residual calculations use Polars Float64 for selected observations and predictions; Decimal values and large integers may lose precision. Original frames remain unchanged. Acceptance uses the strict comparison `error < threshold`.

`learn` returns only when every supplied observation is assigned. If it cannot identify a segment, it raises [`HyDRAIdentificationError`][flowcean.hybrid.hydra.HyDRAIdentificationError]. To diagnose a failure, wrap the learning call above and inspect `.segment` for the failing trace and half-open row bounds:

```python
from flowcean.hybrid.hydra import HyDRAIdentificationError

try:
    result = learner.learn(
        frames, input_features=["t", "x0"], output_features=["dx0"]
    )
except HyDRAIdentificationError as exc:
    print(exc.segment.trace_index, exc.segment.start, exc.segment.stop)
    raise
```

To watch discovery, pass `callback=PlotCallback(learning_data, columns=["x0"], time_column="t")` to the learner, importing [`PlotCallback`][flowcean.hybrid.hydra.PlotCallback] from `flowcean.hybrid.hydra`. Omit `time_column` to plot observation indices. For custom observers, subclass [`HyDRACallback`][flowcean.hybrid.hydra.HyDRACallback] and override the hooks you need: candidate hooks receive a segment and scalar fit, grouping receives accepted segments and the considered observation count, and `finish(result)` receives the fully assigned result only on success.

### Inspect Models and Assignments

[`LearnedFlows`][flowcean.hybrid.hydra.LearnedFlows] contains `result.flows`. Each [`LearnedFlow`][flowcean.hybrid.hydra.LearnedFlow] has a fitted `model` and accepted `segments`. Its position in `result.flows` is the shared flow ID. Several locations can share a flow.

```python
for flow_id, flow in enumerate(result.flows):
    print(flow_id, flow.model, flow.segments)

labeled_traces = result.to_labeled_frames(frames)
flow_models = [flow.model for flow in result.flows]
```

Each [`TraceSegment`][flowcean.hybrid.hydra.TraceSegment] identifies a trace by `trace_index` and a half-open row range `[start, stop)`. Segments record candidate acceptance before the final refit and together cover every supplied observation. `to_labeled_frames(frames)` preserves original columns and adds a non-null Int64 `flow_id` to each row. For NumPy assignments, `to_flow_ids()` returns fresh, fully assigned int64 arrays in trace order.

### Train a Selector and Predict

A **[selector][flowcean.hybrid.hydra.HybridDecisionTreeModel]** chooses a fitted flow model for each observation. Give it features that distinguish the behaviors: in the room example, current and previous temperature provide information about whether the room is heating or cooling.

```python
from flowcean.hybrid.hydra import (
    HybridDecisionTreeLearner,
    SelectorFeatureConfig,
)

selector = HybridDecisionTreeLearner(
    SelectorFeatureConfig(state_features=("x0",), state_history=1),
).learn_from_traces(
    labeled_traces,
    flow_models_by_id=dict(enumerate(flow_models)),
)
```

`state_history=1` adds the previous observation's temperature. History starts afresh at each trace boundary. Include external input features in the selector when they help distinguish the flows.

Use [`HyDRAModel`][flowcean.hybrid.hydra.HyDRAModel] to route batch predictions through the selector:

```python
from flowcean.hybrid.hydra import HyDRAModel

model = HyDRAModel(
    flow_models,
    input_features=result.input_features,
    output_features=result.output_features,
    selector=selector,
)
prediction = model.predict_with_diagnostics(
    learning_data.select(result.input_features)
)
```

Supply all flow inputs and selector features in the prediction frame. `prediction.outputs` contains the predicted derivatives, and `prediction.row_indices` maps them to the input rows. When routing multiple flows with one previous observation, prediction begins at the second row. A single discovered flow can predict directly with `selector=None`.

The [identification walkthrough](../examples/simulated_hybrid_system.md) runs this workflow with two traces and an external target-temperature signal.

### Roll Out Derivative Models

For a learned derivative model, [`HyDRAModel.simulate`][flowcean.hybrid.hydra.HyDRAModel.simulate] integrates the state over a requested time grid. Supply a [`HyDRATraceSchema`][flowcean.hybrid.hydra.HyDRATraceSchema] with corresponding state and derivative columns, plus an input stream for any external inputs.

Multi-flow rollout requires a selector using current time, state, and external input features. The history-based selector above is suitable for batch prediction; rollout with that selector is currently unsupported.

Simulation chooses a flow at each grid point and integrates it until the next point, so choose the grid to resolve the switching behavior you need. The returned table contains state observations, `flow_id` for the selected model, and `flow_time` for the time since that model became active.
