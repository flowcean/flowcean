# Hybrid Systems Gallery

This example renders the registered hybrid-system benchmark gallery and prints a short summary for each benchmark.

Run it from the repository root:

```bash
uv run --directory ./examples/hybrid_systems python run.py
```

The rendered gallery is written to `examples/hybrid_systems/outputs/benchmarks.png`.

The command prints the benchmark name, tags, observed location count, state dimension, sampled step count, event count, and description for every registered benchmark.

You can also import the registry and simulate individual benchmarks directly:

```python
from examples.hybrid_systems.benchmarks import registry
from flowcean.ode import simulate

spec = registry()["Thermostat"]
trace = simulate(
    spec.factory(),
    t_span=spec.t_span,
    input_stream=spec.input_stream,
)
```
