---
icon: lucide/gallery-horizontal-end
---

# Hybrid Systems Gallery

This example simulates 14 reusable hybrid-system models with the horizons and driving signals in `examples/hybrid_systems/scenarios.py`, prints a concise summary for each, and renders a gallery. Those settings belong to the example, not the `flowcean.hybrid.benchmarks` API.

Run it from the repository root:

```bash
uv run --directory ./examples/hybrid_systems python run.py
```

The command writes `examples/hybrid_systems/outputs/benchmarks.png`. Its terminal summary reports each scenario's tags, observed locations, state dimension, sample count, event count, and description. To reuse a factory independently, choose your own time span and input stream:

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

For a standalone turbine simulation, see the [wind-turbine example](https://github.com/flowcean/flowcean/blob/main/examples/hybrid_systems/README.md#wind-turbine).

## Export Automaton Diagrams

Export every gallery scenario's automaton without simulating the systems:

```bash
uv run --directory ./examples/hybrid_systems python export_graphs.py
```

This writes one DOT file per scenario under `examples/hybrid_systems/outputs/automata/`, using lowercase names with spaces replaced by underscores. Add `--svg` to also render SVG files; this requires Graphviz's `dot` executable on `PATH`.

See [Automaton Diagrams](../user_guide/hybrid_systems.md#automaton-diagrams) for the export API and label options, and the [Hybrid Systems guide](../user_guide/hybrid_systems.md) for modeling concepts and simulator options.
