# Learning complete simulated system behavior

This example asks how well a decision tree can map scenario parameters to a
system's complete simulated state trajectory. It studies four Flowcean hybrid
benchmarks in a fixed order: Thermostat, Bouncing Ball, Hybrid Oscillator, and
Tank Valves.

For each system, the experiment draws 256 fitting scenarios and 256 independent
assessment scenarios from the documented parameter domains. A master
`numpy.random.SeedSequence(1)` supplies one ordered child seed per system and
split. Every simulation uses 128 times over the benchmark registry horizon.
Each split is simulated once and then reused while
`sklearn.tree.DecisionTreeRegressor` varies `max_leaf_nodes` from 2 through 16,
plus an unbounded tree.

The target is the complete `trace.x`, flattened in time-major, state-minor
order. Each time-state coordinate is standardized with its fitting-split mean
and population standard deviation. Coordinates whose fitting scale is at most
`1e-12` are excluded. Aggregate RMSE therefore gives equal weight to every
retained time-state coordinate and assessment scenario. Per-state RMSE values
are also reported using each state's retained time coordinates.

Run the standalone workspace example with:

```console
uv run --directory examples/system_behavior_learning run.py
```

Use `--output-dir DIRECTORY` to change the default
`examples/system_behavior_learning/outputs` destination. The command writes:

- `report.json`: exact domains and configuration, seed assignments, target
  metadata, all capacity results, aggregate and per-state metrics, and named
  feature importances.
- `performance.png`: the four assessment RMSE-ratio curves. The distinct `X`
  marker denotes the unbounded tree.

A ratio of `1.0` is the natural reference: it is the assessment RMSE of a
historical-mean predictor fitted only on fitting targets. Values below one mean
that the tree has lower RMSE than that reference on this fixed assessment
split; values above one mean higher RMSE. The report also gives normalized
squared-error reduction, defined as one minus the squared RMSE ratio. Positive
values indicate less squared error than the historical-mean predictor.

The reported impurity-based feature importances are model diagnostics. They do
not establish causal influence or stability. Likewise, the lowest-observed and
saturation summaries describe this one fixed capacity sweep; the assessment
split is not used as a model-selection procedure.
