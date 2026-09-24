---
icon: lucide/gallery-horizontal-end
---

# Hybrid Systems Gallery

This example simulates hybrid-system benchmarks, prints a concise summary for each system, and renders the results as a gallery.

Run it from the repository root:

```bash
uv run --directory ./examples/hybrid_systems python run.py
```

The command writes `examples/hybrid_systems/outputs/benchmarks.png`. Its terminal summary reports each benchmark's tags, observed locations, state dimension, sample count, event count, and description.

Import a benchmark factory and supply a time span and input stream:

```python
import numpy as np

from flowcean.hybrid import simulate
from flowcean.hybrid.benchmarks import thermostat

trace = simulate(
    thermostat(),
    t_span=(0.0, 10.0),
    input_stream=lambda t: np.array([22.0 + 0.8 * np.sin(0.7 * t)]),
)
```

## Wind Turbine

Run the standalone turbine example:

```bash
uv run --directory ./examples/hybrid_systems python wind_turbine.py
```

The command prints mode changes and saves `examples/hybrid_systems/outputs/wind_turbine.png`. The plot shows wind speed, rotor speed, blade pitch, tower displacement, and generator mechanical power. Background colors show the controller mode; the dashed power line marks rated power.

See the [wind-turbine API](../reference/hybrid.md#flowcean.hybrid.benchmarks.wind_turbine.wind_turbine) for equations and operating limits.

## Export Automaton Diagrams

Export every benchmark's automaton without simulating the systems:

```bash
uv run --directory ./examples/hybrid_systems python export_graphs.py
```

This writes one DOT file per benchmark under `examples/hybrid_systems/outputs/automata/`, using lowercase benchmark names with spaces replaced by underscores. Add `--svg` to also render SVG files; this requires Graphviz's `dot` executable on `PATH`.

See [Automaton Diagrams](../user_guide/hybrid_systems.md#automaton-diagrams) for the export API and label options, and the [Hybrid Systems guide](../user_guide/hybrid_systems.md) for modeling concepts and simulator options.
