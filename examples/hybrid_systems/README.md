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
