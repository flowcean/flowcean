---
icon: lucide/chart-no-axes-combined
---

# Evaluation

## Choose What to Score

Evaluate selected observations with NumPy, scikit-learn, or Flowcean's [existing metrics](../reference/sklearn.md). Producing predictions is separate from scoring: a learned model need not support execution to be useful, and evaluation does not prescribe how it was fitted.

Your experiment decides which observations correspond, which signals to score, and how to handle missing values. Align independent time grids before comparing values when needed. Exclude measured seeds or unavailable warmup rows rather than counting them as predictions.

```python
import numpy as np
import polars as pl

reference = pl.DataFrame({"temperature": [20.0, 21.0, 22.0]})
prediction = pl.DataFrame({"temperature": [20.0, 20.8, 21.6]})
signals = ["temperature"]
seed_rows = 1
observed = reference.select(signals).slice(seed_rows).to_numpy()
predicted = prediction.select(signals).slice(seed_rows).to_numpy()
error = predicted - observed
mae = np.mean(np.abs(error), axis=0)
rmse = np.sqrt(np.mean(error**2, axis=0))
max_error = np.max(np.abs(error), axis=0)
print(signals, mae, rmse, max_error)
print(f"Scored rows: {len(error)}/{reference.height}")
```

Each result follows the selected signal order and retains that signal's units. Avoid pooling outputs with different physical units. For custom figures, use ordinary Matplotlib; the optional [`plot_style()` context](../user_guide/hybrid_systems.md#inspect-the-run) supplies Flowcean's visual defaults.

See [FaMoS](../examples/famos.md), [MMSR](../examples/mmsr.md), and [Dainarx](../examples/dainarx.md) for explicit seed-excluded held-out scoring, or the [HyDRA example](../examples/simulated_hybrid_system.md) for a saved reference/prediction plot.
