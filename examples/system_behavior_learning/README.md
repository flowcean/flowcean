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

## Optional path-box representatives

A companion analysis asks a narrower scientific question for all four systems:
how closely does one geometric representative of each fitted tree leaf realize
the leaf's fitting prototype? It uses the same deterministic fitting and
assessment scenarios as the main experiment and fixes every tree at
`make_tree(8)`. Eight leaves are an illustrative fixed capacity, not a selected
or recommended capacity.

Each root-to-leaf path is intersected with the system's declared parameter
domain to form a box, including repeated splits on the same feature. The
representative is the arithmetic midpoint of that box. Its simulated,
standardized trajectory is compared with the leaf prototype, defined as the
mean standardized trajectory of the fitting scenarios assigned to the leaf and
checked against the tree's prediction.

There is a numerical subtlety at every split. Scikit-learn converts inputs to
`float32` before `apply` and `predict`, even though callers can provide
`float64` values and the model exposes a `float64` threshold. The analysis
therefore reconstructs each threshold's effective preimage boundary in the
original `float64` domain and records which branch owns exact equality under
round-to-nearest-even conversion. Those effective boundaries, rather than the
stored model thresholds, define box endpoints, volumes, and midpoints. Every
midpoint is checked with `tree.apply`.

Distances are RMS values over equally weighted retained standardized trajectory
coordinates. Within each leaf, the midpoint distance is shown beside the
assessment members' distances to the fitting prototype. A midpoint's empirical
CDF fraction is the count of assessment residuals less than or equal to its
error, divided by the leaf's assessment occupancy. The two descriptive
aggregates weight leaf midpoint errors and leaf assessment means by exact
relative path-box volume under the independent-uniform declared domain.

Run the optional analysis with:

```console
uv run --directory examples/system_behavior_learning \
  run_path_representatives.py --output-dir outputs
```

It writes (and overwrites) exactly six analysis artifacts in the requested
directory:

- `path_representatives.json`: shared compact configuration followed by one
  record per system with domains, state and target-coordinate metadata,
  effective predicates, boxes, occupancies, residual summaries, midpoint
  empirical CDF fractions, and volume-weighted aggregates.
- `path_representatives.png`: a combined 2-by-2 view of the four systems. Each
  panel shows held residual boxplots, midpoint diamonds, and the two weighted
  aggregate lines in standardized-coordinate RMS units. Panel scales are
  independent.
- `path_prototypes_thermostat.png`, `path_prototypes_bouncing_ball.png`,
  `path_prototypes_hybrid_oscillator.png`, and
  `path_prototypes_tank_valves.png`: one eight-row prototype grid per system,
  with state dimensions in columns and common physical-coordinate limits down
  each column. Faint gray lines are all fitting trajectories assigned to the
  leaf, the thick blue line is their leaf prototype, and the dashed orange line
  is the path-midpoint trajectory. Only fitting trajectories appear as faint
  traces because only fitting members form the prototypes.

All target coordinates happen to be retained for this fixed example. The
prototype figures invert the fitted standardization into physical coordinates
without resimulation and check that the complete coordinate transform can be
inverted; they fail rather than fill excluded coordinates with fitted means.
A prototype is still an average, not necessarily a trajectory produced by any
scenario.

This is a descriptive view of one deterministic split per system and one fixed
tree capacity. A box midpoint need not resemble an observed scenario, and
path-box volume need not resemble empirical occupancy. The results do not
establish causal importance, uncertainty, robustness across seeds, or behavior
at other capacities.
