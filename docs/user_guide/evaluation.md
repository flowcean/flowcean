---
icon: lucide/chart-no-axes-combined
---

# Evaluation

## Compare Sampled Traces

Use `compare_traces` to compare recorded observations with predictions from any method. It takes two eager Polars frames, not models. Producing predictions is a separate, optional operation: a learned model need not support execution to be useful, and comparison does not prescribe how it was fitted.

```python
import polars as pl

from flowcean.hybrid import compare_traces, plot_comparison

reference = pl.DataFrame(
    {"time": [0, 1, 2], "temperature": [20.0, 21.0, 22.0]}
)
prediction = pl.DataFrame(
    {"time": [0, 1, 2], "temperature": [None, 20.8, 21.6]}
)
comparison = compare_traces(
    reference,
    prediction,
    columns=["temperature"],
    time_column="time",
    exclude_initial=1,
)
print(comparison.scores)
print(comparison.total_rows, comparison.excluded_rows, comparison.scored_rows)
ax = plot_comparison(comparison, "temperature")
ax.set_ylabel("Temperature (deg C)")
```

Supply the same row-ordered time grid in both frames; comparison does not join, sort, interpolate, or truncate. Duplicate timestamps correspond by row position. If different grids need alignment, choose and apply that policy before comparing. The [API reference](../reference/hybrid.md#flowcean.hybrid.compare_traces) specifies the numeric time tolerance.

Declare the leading rows to exclude when prediction uses measured seeds or needs warmup. Missing warmup predictions may stay missing; all scored observations must be finite. Coverage counts include the excluded rows. The plot marks the scoring boundary and gives the excluded row count in its legend without treating every excluded value as a measured seed.

## Inspect Each Quantity

`scores` contains MAE, RMSE, and maximum absolute error for each named signal in the declared order. These errors have the signal's original units; outputs with different units are not pooled. Plot one quantity per axes, with reference and prediction distinguished by solid and dashed lines. Create multiple axes yourself for multiple quantities. The helper returns the axes without showing or saving the figure and leaves missing values as gaps.

See [FaMoS](../examples/famos.md), [MMSR](../examples/mmsr.md), and [Dainarx](../examples/dainarx.md) for seed-excluded held-out scoring, or the [HyDRA example](../examples/simulated_hybrid_system.md) for a saved comparison plot. Sampled comparisons do not reconstruct reset events; use `plot_trajectory` on a native `HybridTrajectory` when its continuous segments and event structure matter.
