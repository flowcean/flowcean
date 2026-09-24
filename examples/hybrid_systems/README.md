# Hybrid Systems Gallery

This example simulates 14 reusable benchmark models with the horizons and driving signals defined in `scenarios.py`, prints a concise summary, and renders a gallery. These scenario settings belong to the example, not the reusable model package.

Run it from the repository root:

```bash
uv run --directory ./examples/hybrid_systems python run.py
```

The command writes `examples/hybrid_systems/outputs/benchmarks.png`.

To export every gallery scenario's automaton without simulation:

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

The rotor starts already turning; startup and shutdown are not modeled. See the [wind-turbine API](https://flowcean.me/reference/hybrid/#flowcean.hybrid.benchmarks.wind_turbine.wind_turbine) for equations and operating limits.

## Reusing individual benchmarks

Import a model factory from `flowcean.hybrid.benchmarks` and choose the time span and inputs for your own simulation. For example, the thermostat requires a finite input vector `[target]`:

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

The other driven models require `[force]` for the impact oscillator, `[threshold]` for the time-varying event surface, `[reference, reference_rate]` for the PID plant, and positive `[wind]` in m/s for the wind turbine.
