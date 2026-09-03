# Hybrid Systems

Hybrid systems combine continuous evolution with discrete changes in behavior. Flowcean represents the active discrete mode as a location, evolves a continuous state according to that location's dynamics, and changes locations when event surfaces trigger transitions.

Use `flowcean.hybrid` for hybrid-system definition, simulation, traces, plotting, benchmarks, and identification:

```python
from flowcean.hybrid import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    Location,
    Reset,
    SurfaceEntryPolicy,
    Transition,
    simulate,
)
```

## Model and Continuous Evolution

A `HybridSystem` contains locations, transitions, an initial location, an initial continuous state, and global parameters. Exactly one location is active at each point in the simulation. A same-time transition chain may visit several locations at one physical time, ordered by microsteps. Between transitions, the active location's `ContinuousDynamics` callback returns the derivative of the continuous state.

The continuous state is a one-dimensional NumPy array. A system with continuous dynamics but no discrete switching is represented by one location and no transitions.

Callbacks can use the physical time, continuous state, effective parameters, and input stream:

```python
def flow(t, state, parameters, input_stream):
    control = input_stream(t)
    return parameters["gain"] * state + control
```

Callbacks may declare only the arguments they need when they retain the canonical names `t`, `state`, `parameters`, and `input_stream`. For example, a flow that depends only on state and parameters can use `def flow(state, parameters): ...`. System parameters apply globally, while parameters declared on the active location override global values with the same name. An input stream is a callable that returns a one-dimensional input array for a requested physical time.

## Transitions and Resets

A transition connects one source location to one target location. Its event surface is a scalar function whose zero crossing can trigger the transition. `CrossingDirection.RISING`, `FALLING`, and `EITHER` restrict which crossing directions are accepted during continuous evolution.

An event surface describes a numerical zero crossing, not a Boolean region. For example, a falling surface does not fire merely because its value is already negative. Whenever simulation enters a location—including the initial location—the simulator evaluates every outgoing surface exactly once before deciding what to do. A value equal to `0.0` (including `-0.0`) is on the surface. Arbitrarily small nonzero values and positive or negative infinity retain their sign; NaN is invalid.

Each transition's `entry_policy` determines what exact zero means on entry:

- `SurfaceEntryPolicy.ERROR` (the default) raises `SurfaceEntryError`, requiring the model to make its intent explicit.
- `SurfaceEntryPolicy.TRIGGER` performs the transition immediately at the same physical time.
- `SurfaceEntryPolicy.CONTINUE` leaves the transition inactive for entry handling and passes the zero-valued surface unchanged to continuous integration. Use this when a trajectory naturally enters a boundary and departs in the direction opposite to the transition.

Entry decisions are atomic. NaN takes precedence over every policy. Any zero `ERROR` surface is reported before trigger selection. More than one zero `TRIGGER` surface raises `AmbiguousTransitionError`; exactly one performs a jump, applies its reset, enters the target, and repeats entry evaluation there. A `CONTINUE` surface does nothing during entry handling.

For each settled continuous segment, the simulator:

1. Integrates the active location's dynamics until the earliest detected outgoing event or the end of the time span.
2. Records the state immediately before the transition.
3. Applies the optional reset using the source location's effective parameters.
4. Enters the target location and resolves its entry policy before integrating again.

Without a reset, the continuous state is unchanged by a transition. A transition whose source and target are the same location therefore requires a reset. A reset normally returns a one-dimensional state with the same dimension as the state before the transition. A scalar is also accepted for a single-state system.

!!! warning "Simultaneous entry transitions"

    Multiple exact-zero `TRIGGER` surfaces on one location entry are ambiguous and stop simulation. Design the entry state or policies so that at most one requests an immediate jump.

## Physical Time and Microsteps

`Event.time` is physical simulation time. Immediate transitions caused by a reset do not advance physical time. Their zero-based `microstep` values preserve their order within the same-time transition chain. An initial-entry trigger has microstep 0. A continuously detected crossing also has microstep 0, and triggers on successive target entries use microsteps 1, 2, and so on.

Suppose a transition from A to B resets the state onto an event surface in B, which immediately causes a transition from B to C:

| Record | Physical time | Microstep | Location change | Recorded state |
| --- | ---: | ---: | --- | --- |
| First event | 1.0 | 0 | A -> B | `state_before` in A and `state_after` in B |
| Second event | 1.0 | 1 | B -> C | `state_before` in B and `state_after` in C |
| Trace row | 1.0 | - | C | Final state after the complete chain |

Every transition in the chain counts toward `max_jumps`. Simulation raises an error if that limit is exceeded.

After a continuous crossing, integration restarts at the exact event time with the post-jump state; the simulator does not offset time to move away from the root. A location must be settled before this restart. If the ODE solver nevertheless returns an event at or before the segment start, simulation raises `SimulationProgressError` rather than applying the transition. This usually indicates stateful callbacks, a discontinuous event surface, or insufficient floating-point time resolution; use deterministic callbacks and continuous surfaces.

## Trace Boundary Semantics

Flat traces are right-continuous at transitions. A trace row at an event time contains the final state and active location after the complete immediate transition chain. This rule also applies to transitions at the initial or final time.

Each `Event` preserves the individual jump through independent `state_before` and `state_after` snapshots. The event sequence therefore retains intermediate states even though the flat trace contains only the final post-chain value at that physical time.

A `Trace` contains aligned simulation records:

| Field | Contents |
| --- | --- |
| `t` | Physical sample times |
| `x` | Continuous state at each sample time |
| `location` | Active location label at each sample time |
| `events` | Ordered transition events with pre-reset and post-reset states |
| `u` | Captured inputs, when requested |
| `dx` | Captured state derivatives, when requested |

## Sampling

The sampling options determine which physical times appear in the flat trace:

- Without `sample_times` or `sample_dt`, the trace uses the adaptive time points returned by the ODE solver. Detected event times are included.
- `sample_times` returns exactly the requested, non-descending times within `t_span`.
- `sample_dt` creates a fixed grid from the start of `t_span` and includes the final endpoint, even when the interval is not an exact multiple of the step.
- `sample_times` and `sample_dt` cannot be used together.

A fixed sampling grid is not expanded with off-grid transition times. Those transitions remain available through `Trace.events`. If a requested sample coincides with a transition, the sample follows the right-continuous boundary rule.

When an input stream is supplied, inputs are captured by default unless `capture_inputs=False` is selected. Setting `capture_derivatives=True` reevaluates the active location's dynamics at each returned sample. At a transition boundary, the derivative therefore uses the final target location and post-transition state. Derivative capture assumes that flow callbacks are pure under repeated evaluation.

## Simulation

Construct a `HybridSystem`, then call `simulate` with a physical time span and optional sampling configuration:

```python
trace = simulate(
    system,
    t_span=(0.0, 10.0),
    sample_dt=0.05,
    capture_derivatives=True,
)
```

The simulator also accepts input streams, initial-state and initial-location overrides, solver tolerances, and an event limit. Use `trace_to_polars` when identification or evaluation needs tabular state, input, and derivative columns.

See the [`flowcean.hybrid` API](../reference/flowcean/hybrid/index.md) for model and trace types and the [simulator API](../reference/flowcean/hybrid/simulator.md) for complete function signatures.

## Benchmarks and Identification

Reusable systems are available from `flowcean.hybrid.benchmarks`:

```python
from flowcean.hybrid import simulate
from flowcean.hybrid.benchmarks import registry

spec = registry()["Thermostat"]
trace = simulate(
    spec.factory(),
    t_span=spec.t_span,
    input_stream=spec.input_stream,
)
```

HyDRA interfaces such as `HyDRALearner`, `HyDRATraceSchema`, and `HybridDecisionTreeLearner` are exported from `flowcean.hybrid` for identifying mode dynamics and selectors from sampled traces.

Start with the [minimal hybrid system](../examples/hs_simple.md), browse the [hybrid systems gallery](../examples/hybrid_systems.md), and then run the [simulated hybrid system identification](../examples/simulated_hybrid_system.md) workflow.
