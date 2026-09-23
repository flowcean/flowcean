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

Runs a 120-second wind cycle from 7 to 15 m/s and back, prints mode changes, and saves `examples/hybrid_systems/outputs/wind_turbine.png`.

The plot shows wind speed, rotor speed, blade pitch, tower displacement, and generator power. Background colors mark the controller mode. This is mechanical shaft power, not electrical output. The dashed 5.30 MW line is the rated reference, not a hard cap in every mode.

The rotor starts already turning; startup and shutdown are not modeled. See the [wind-turbine API](https://flowcean.me/reference/hybrid/benchmarks/#flowcean.hybrid.benchmarks.wind_turbine.wind_turbine) for equations and operating limits.

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
