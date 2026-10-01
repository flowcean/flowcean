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

A crossing can schedule a switch for later. For example, an inlet valve may
remain open for one time unit after the water reaches its threshold:

```python
close_valve = Transition(
    source=filling,
    target=closed,
    event_surface=upper_boundary,
    delay=1.0,
)
```

`delay` is a fixed, finite, nonnegative duration in the model's time units.
It defaults to zero, which takes the transition immediately. A positive delay
keeps the source location, flow, and residence clock active until execution.
The reset, if present, uses the state, time, inputs, parameters, and source
residence time at execution.

The first crossing schedules one occurrence per transition. Crossing back or
crossing again neither cancels nor restarts it. Other outgoing transitions
remain available, and several different transitions may be pending. Leaving
the source visit cancels all its remaining pending occurrences, including when
a self-transition starts a new visit.

Multiple simultaneous detections may schedule different deadlines. Multiple
requests to **execute** at the same time are an error, including a pending
deadline coinciding with an immediate transition. Times indistinguishable at
root-finding precision are treated as simultaneous.

On entry, `SurfaceEntryPolicy.TRIGGER` detects the transition immediately and
starts its delay. Each run starts with no pending occurrences, including when
an initial residence time is supplied. A deadline at the simulation endpoint
executes; later deadlines do not. Only executed transitions count toward
`max_jumps`.

The [delayed valve example](../examples/hybrid_systems.md#delayed-valve-closure)
compares immediate closure with closure one time unit after detection.

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

For a delayed transition, `event.detection_time` records the earlier crossing
(or detection on entry), while `event.time` records execution. They are equal
for zero-delay transitions. The trajectory records executed transitions.

Detection may split continuous segments without starting a new location visit.
Sampling during the delay still reports the source location and its continuing
residence time. Event markers in plots mark execution.

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

Instead of writing every flow equation yourself, you can use **[HyDRA][flowcean.hybrid.hydra.HyDRALearner]** to learn flow models from data.

For simulated data, request the rates of change alongside the state:

```python
learning_data = trajectory.sample(
    dt=0.02,
    include_derivatives=True,
)
```

Here, `x0` contains temperature and `dx0` contains the rate returned by the active flow. Choose enough observations to capture the brief behaviors you want to learn.

If the run uses external inputs, include them with `include_inputs=True`. For measured recordings, derivatives must instead be supplied or estimated, taking care not to treat reset jumps as continuous rates of change.

### Understand What Is Learned

HyDRA fits continuous flow models. A **[selector][flowcean.hybrid.hydra.HybridDecisionTreeModel]** chooses which fitted model to use from the information supplied to it.

This produces flow models and a selection rule, rather than a reconstruction of the original automaton's locations, transitions, and resets. Different locations can share the same continuous behavior, so learned flow IDs need not correspond to native location IDs.

The selector also needs enough information to distinguish the behaviors. Our room illustrates why: temperature alone cannot tell whether heating is on or off.

### Simulate a Learned Model

[`HyDRAModel.simulate`][flowcean.hybrid.hydra.HyDRAModel.simulate] chooses a flow at each requested grid point and integrates that flow until the next point. Consequently, changing this grid can change the simulated behavior. This differs from sampling an existing native trajectory, where observation times do not affect the run.

Its returned table uses `flow_id` for the selected model and `flow_time` for the time since that model became active.

Two current limits matter when choosing this workflow:

- Trace-based HyDRA learning supports one state/derivative pair.
- Built-in learned-model simulation does not support selectors that require history.

The [identification walkthrough](../examples/simulated_hybrid_system.md) demonstrates fitting flows, training a selector, and comparing a learned run with its reference.
