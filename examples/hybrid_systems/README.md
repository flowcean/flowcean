# Hybrid Systems Gallery

This example simulates the registered hybrid-system benchmarks, prints a concise summary for each system, and renders the results as a gallery.

Run it from the repository root:

```bash
uv run --directory ./examples/hybrid_systems python run.py
```

The command writes `examples/hybrid_systems/outputs/benchmarks.png`.

To export every registered benchmark's automaton without simulation:

```bash
uv run --directory ./examples/hybrid_systems python export_graphs.py
```

This writes one DOT file per benchmark under `examples/hybrid_systems/outputs/automata/`. Add `--svg` to also render SVG files with Graphviz's `dot` executable. See the [automaton diagram guide](../../docs/user_guide/hybrid_systems.md#automaton-diagrams) for the API and label options.

## Wind turbine

To run just the wind turbine:

```bash
uv run --directory ./examples/hybrid_systems python wind_turbine.py
```

The rotor starts already turning. Over 120 seconds, wind speed rises smoothly from 7 to 15 m/s and returns to 7 m/s. The example prints the controller's mode changes and writes `examples/hybrid_systems/outputs/wind_turbine.png`, showing wind speed, rotor speed, blade pitch, tower displacement, and generator mechanical power. Shaded areas identify the active controller mode, with consistent colors and one shared legend across the panels.

The power panel compares the simulated generator shaft power with the dashed 5.30 MW rated reference. Power is derived using `wind_turbine_power`; it is not an additional state. Electrical conversion losses are not modeled. Rated power is held in the `rated_power` mode, not imposed as a hard cap in every mode.

The model covers running operation, not startup from rest or shutdown. See the `wind_turbine` factory docstring for its equations and operating limits.

## Reusing individual benchmarks

The canonical benchmark registry is available from `flowcean.hybrid.benchmarks`:

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
