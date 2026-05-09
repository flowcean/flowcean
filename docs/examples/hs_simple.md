# Minimal Hybrid System

This example builds and simulates a small two-location thermostat with the object-based hybrid-system API. It is intended as the shortest complete example of defining locations, flows, event surfaces, and transitions directly in Python.

Run it from the repository root:

```bash
uv run --directory ./examples/hs-simple python main.py
```

The reusable `build_thermostat()` function returns a `HybridSystem` assembled from these objects:

- `Location` represents a discrete mode of the system, here `heating` and `cooling`.
- `ContinuousDynamics` stores the derivative function for a location.
- `EventSurface` defines when a transition is enabled.
- `CrossingDirection` restricts an event to rising or falling crossings.
- `Transition` connects a source location to a target location.
- `simulate` runs the hybrid system over a time interval and returns a trace.
- `plot_trace` renders the simulated state, locations, and events.

The thermostat starts in the `heating` location at temperature `19.0`. It switches to `cooling` when the `too_hot` event surface is crossed upward at `21.0`, and switches back to `heating` when the `too_cold` event surface is crossed downward at `19.0`.

The script prints the number of recorded events and writes a plot artifact to `examples/hs-simple/outputs/hs-simple.png`.
