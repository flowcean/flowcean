# Hybrid Systems

Hybrid systems combine discrete modes with continuous evolution. Flowcean represents each mode as a location with its own dynamics and connects locations with event-triggered transitions.

Use `flowcean.hybrid` for hybrid-system definition, simulation, traces, plotting, benchmarks, and identification:

```python
from flowcean.hybrid import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    Location,
    Transition,
    simulate,
)
```

## Modeling Concepts

- A **location** is a discrete mode. Its continuous dynamics apply while the system remains in that location.
- **Continuous dynamics** return the derivative of the continuous state.
- An **event surface** is a scalar function whose zero crossing can trigger a transition.
- A **crossing direction** restricts an event to rising, falling, or either zero crossings.
- A **transition** connects a source location to a target location.
- A **reset** optionally changes the continuous state when a transition fires.
- A **trace** records sampled times, states, active location labels, and events. It can also capture inputs and derivatives.
- A **selector** assigns observations to learned modes. HyDRA combines selectors with learned mode dynamics.

A system with continuous dynamics but no discrete switching is represented as a `HybridSystem` with one location and no transitions. Multi-location systems add event surfaces, transitions, and optional resets.

## Simulation

Construct a `HybridSystem`, then call `simulate` with a time span and either a fixed sample interval or explicit sample times:

```python
trace = simulate(
    system,
    t_span=(0.0, 10.0),
    sample_dt=0.05,
    capture_derivatives=True,
)
```

The simulator accepts optional input streams, initial-state and initial-location overrides, solver tolerances, and event limits. Use `trace_to_polars` when identification or evaluation needs tabular state, input, and derivative columns.

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
