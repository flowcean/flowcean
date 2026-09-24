---
icon: lucide/workflow
---

# Minimal Hybrid System

This example constructs a thermostat from locations, continuous dynamics, event surfaces, and transitions, then simulates it.

Run it from the repository root:

```bash
uv run --directory ./examples/hs-simple python main.py
```

The [`build_thermostat()` function](https://github.com/flowcean/flowcean/blob/main/examples/hs-simple/main.py) assembles a `HybridSystem` from these objects:

- `Location` represents a discrete mode of the system, here `heating` and `cooling`.
- `ContinuousDynamics` stores the derivative function for a location.
- `EventSurface` defines a function whose zero crossing can trigger a transition.
- `CrossingDirection` restricts an event to rising or falling crossings.
- `Transition` connects a source location to a target location.

Temperature rises at a constant rate while heating and falls at a constant rate while cooling. Crossing the upper or lower temperature boundary selects the other location without resetting the temperature. The `HybridSystem` constructor also sets the initial state and location.

The script passes this model to `simulate` and displays its trace with `plot_trace`. It prints the number of recorded events and opens a Matplotlib plot.
