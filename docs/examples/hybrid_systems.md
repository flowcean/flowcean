---
icon: lucide/gallery-horizontal-end
---

# Hybrid Systems Gallery

This example simulates the registered hybrid-system benchmarks, prints a concise summary for each system, and renders the results as a gallery.

Run it from the repository root:

```bash
uv run --directory ./examples/hybrid_systems python run.py
```

The command writes `examples/hybrid_systems/outputs/benchmarks.png`. Its terminal summary reports each benchmark's tags, observed locations, state dimension, sample count, event count, and description.

The benchmark registry is part of `flowcean.hybrid.benchmarks`, so individual systems can be reused without importing the example package:

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

For a standalone turbine simulation, see the [wind-turbine example](https://github.com/flowcean/flowcean/blob/main/examples/hybrid_systems/README.md#wind-turbine).

## Export Automaton Diagrams

Export every registered benchmark's automaton without simulating the systems:

```bash
uv run --directory ./examples/hybrid_systems python export_graphs.py
```

This writes one DOT file per benchmark under `examples/hybrid_systems/outputs/automata/`, using lowercase benchmark names with spaces replaced by underscores. Add `--svg` to also render SVG files; this requires Graphviz's `dot` executable on `PATH`.

See [Automaton Diagrams](../user_guide/hybrid_systems.md#automaton-diagrams) for the export API and label options, and the [Hybrid Systems guide](../user_guide/hybrid_systems.md) for modeling concepts and simulator options.
