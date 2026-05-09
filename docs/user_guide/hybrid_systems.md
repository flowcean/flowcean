# Hybrid Systems

Hybrid systems combine discrete modes with continuous evolution. Flowcean models this by representing each mode as a location with its own dynamics and connecting locations with guarded transitions.

Common terms:

- A **location** is a discrete mode of the system. Each location has continuous dynamics that apply while the system remains in that location.
- **Continuous dynamics** describe how the continuous state changes over time inside one location. In code, this is usually a function returning the derivative of the state.
- An **event surface** is a scalar function whose zero crossing can enable a transition. For example, `temperature - limit` crosses zero when a threshold is reached.
- A **crossing direction** restricts an event surface to rising crossings, falling crossings, any crossing, or no direction filter, depending on the system definition.
- A **transition** connects a source location to a target location and is triggered by an event surface.
- A **reset** changes the continuous state when a transition is taken. If no reset is configured, the state is carried across unchanged.
- A **trace** is the recorded result of a simulation or measurement. It usually contains sampled time points, continuous states, active locations, events, and optionally derivatives.
- A **derivative** is the instantaneous rate of change of a continuous state variable. Learned dynamics often predict derivatives from state and input features.
- A **selector** assigns samples or states to discrete modes. HyDRA uses a selector to decide which learned mode should explain a sample.

The current hybrid-system simulation API is imported from `flowcean.ode`:

```python
from flowcean.ode import HybridSystem, Location, Transition, simulate
```

The `flowcean.ode` package name comes from the package history: the simulation support started with ordinary differential equation environments and later grew to include object-based hybrid-system definitions. The name is still the public import path for these APIs.

For runnable examples, start with the [minimal hybrid system](../examples/hs_simple.md), browse the [hybrid systems gallery](../examples/hybrid_systems.md), and then try the [simulated hybrid system identification](../examples/simulated_hybrid_system.md) example.
