# System-behavior suite comparison

This example compares compact scenario suites for five Flowcean hybrid-system
benchmarks: Thermostat, Bouncing Ball, Hybrid Oscillator, Tank Valves, and a
PID-Controlled Plant. The full configuration uses 30 independent
fitting/assessment replicates per
system, one shared 4096-scenario reference sample per system, capacities
2, 4, 8, 16, 32, and 64, and 128 samples of each complete state trajectory.
Tank Valves scenarios vary pump inflow, both effective outlet areas, open-valve
transfer gain, thresholds, and both initial levels; tank cross-sectional areas
and gravity remain fixed. The [model](../../src/flowcean/hybrid/benchmarks/tank_valves.py)
represents atmospheric tanks on a common bottom datum, an always-on pump, and a
one-way gravity transfer valve, with no modeled overflow. Outlet and valve areas
are the discharge coefficient times geometric area in square meters. Inflow is
in cubic meters per second, and levels are in meters. Its square-root flows use
the standard gravity-orifice basis described in [Section 2.1.2 of this
reference](https://www.mdpi.com/2076-3417/13/9/5414), not an apparatus
calibration. Tank solver tolerances and numerical height allowances are recorded
in the settings. Defaults are `rtol=1e-6`, `atol=1e-8`, and a `1e-6 m` boundary
allowance: levels below `-1e-6 m` or above the `1.5 m` wall height plus that
allowance are rejected. Accepted numerical residuals are retained without clipping.
PID-Controlled Plant scenarios span 11 controller, plant, setpoint, actuator, and initial-state
coordinates; one actuator-limit coordinate defines symmetric lower and upper
saturation limits.

Run the development configuration from the repository root:

```console
uv run --frozen --directory examples/system_behavior_learning python run.py
```

For the full experiment, add `--full`:

```console
uv run --frozen --directory examples/system_behavior_learning python run.py --full
```

Add `--workers N` to either command to override the number of simulation
processes; `--workers 1` runs serially. There is no automatic CPU manager.

Numerical runtimes use a global thread cap of 8 by default, independently of
`--workers`. Override it with `--numerical-threads N`; use
`--numerical-threads 1` for serial numerical kernels. The cap applies to the
parent process and simulation workers for the entire run, and prior runtime and
environment limits are restored after the API returns.

`run.py` selects one immutable preset from `settings.py`: `DEVELOPMENT` or
`FULL`. Each includes statistical analysis settings in `statistics`, so
simulation and bootstrap seeds are selected together. Development uses 2
histories and 3 geometric repetitions; full
uses 30 histories and 31 repetitions. Development is reduced-repetition, not a
tiny smoke run: both retain all five systems, sample sizes, six capacities, and
10,000 bootstrap draws. The two configurations have separate seeds.

The full run is expensive; use development for iteration. Both modes allow
uncommitted source changes. Reproducing numerical results requires the same
source, settings, seeds, and numerical environment; run names, timestamps, and
elapsed times vary between runs.

Both physical configurations predeclare replicate 0 and capacity 8 for the
explanatory prototype figures, so the displayed tree is not selected after
inspecting outcomes. This does not designate a primary comparison capacity.

The settings module is also the single source of truth for each system's ordered
scenario ranges, fixed numeric parameters, simulation horizon, and state names;
these settings are serialized directly into `settings.json`. Systems and
histories run in their original order. By default, one run-scoped pool
simulates scenario chunks across worker processes, including single-system
runs; `--workers 1` uses no pool. Each system has a replicate progress bar and
a temporary sub-bar. During setup and data preparation, the sub-bar counts completed
scenario simulations. During evaluation, it counts capacities and shows whether
it is fitting a tree, realizing midpoints, selecting medoids, or evaluating a
specific geometric suite. Each invocation creates a fresh
`outputs/experiment-<UTC-microsecond-timestamp>/` directory by default:

- `settings.json`: the full configuration, including nested statistical settings.
- `environment.json`: Python/platform/package versions, numerical-thread
  environment variables, and source Git HEAD plus dirty flag (including
  untracked files). Missing Git information is recorded as null. HEAD plus a
  dirty flag does **not** reconstruct uncommitted source; retain that source
  separately when needed.
- `uv.lock`: a copy of the repository lockfile, when available.
- `data/raw/`: recorded numerical experiment evidence.
- `data/records.json`: ordered scalar records and prototype metadata.
- `data/prototypes/prototype_<index>.npz`: selected prototype arrays, when present.
- `results/`: the following tables, figures, and inference outputs.

Under `results/`:

- `tree_metrics.csv`
- `leaf_metrics.csv`
- `leaf_boxes.csv`
- `unbounded_tree_metrics.csv`
- `unbounded_leaf_metrics.csv`
- `unbounded_leaf_boxes.csv`
- `suite_metrics.csv`
- `suite_status.csv`
- `execution_status.csv`
- `prototype_trajectories.csv` (when selected prototype evidence is complete)
- `summary.csv`
- `prediction_quality.png`
- `representative_realization.png`
- `reference_coverage.png`
- `coverage_ratios.png`
- `inference/<system>/`: `method_scores.csv`, `paired_effects.csv`,
  `capacity_effects.csv`, `bootstrap.npz`, and `metadata.json`
- one `trajectory_prototypes_<system>.png` per system with complete selected evidence

`data/raw/<system>/shared.npz` stores the reference scenarios, physical reference
trajectories, and sample times generated once per system. Each
`data/raw/<system>/replicate_<index>.npz` stores that replicate's fitting and
assessment scenarios and physical trajectories together with the fitting-only
transform means, scales, and retained-coordinate indices. The replicate
archive also stores each bounded and unbounded fitted tree's exact node-state
array, values, depth, node count, fitted dimensions, maximum feature count, and
integer random state. Unbounded midpoint batches are stored under
`unbounded_midpoints_<field>`. The shared archive stores each geometric suite's
scenarios and physical trajectories once per method, realized budget, and
repetition. Replicate archives store behavior and input-only midpoint scenarios
and trajectories, plus archive-PAM scenarios, trajectories, and fitting-row
indices, for each requested capacity. Structurally infeasible input-only suites
are omitted. Standardized trajectories are reconstructed from the physical
arrays and transform rather than stored twice.

Replicate archives also store `reference_assignments` (int64) and
`reference_distances` (float64) for every evaluated suite. These vectors follow
shared reference-scenario row order. Assignments are zero-based suite-row
indices; equal nearest distances select the first suite row. Distances are
coordinate-RMS over retained standardized trajectory coordinates. Local keys
use `suite_capacity_<capacity>__<method>__reference_<field>`; geometric keys use
`suite_<method>__budget_<realized>__repetition_<repetition>__reference_<field>`
(with three-digit numeric fields). Geometric vectors are replicate-specific
because the fitting transform differs, and are stored only once per realized
budget, method, and repetition even when requested capacities share that budget.
Invalid or dependency-blocked suites have no reference vectors.

Run destinations are never reused or overwritten. Direct
`run_and_write(settings, systems)` calls use `settings.output_dir` as the exact
new run destination. Settings and environment metadata are saved before
simulation. Raw batches remain on disk if execution is interrupted; scalar
records are saved only after experiment generation returns. Reporting and
configured inference follow, with inference loading one system at a time.
Exceptions propagate and partial files remain for inspection; there is no
automatic cleanup or resume. The final completion message appears only after
reporting and inference finish, and does not imply every scientific comparison
is available. Existing runs are left untouched.

Outputs stay under the configured worktree output folder by default; move
retained runs before deleting that worktree.

To report a saved run again, use Python from this example directory:

```python
from pathlib import Path

from report import report_saved_run

report_saved_run(
    Path("outputs/experiment-..."),
    Path("outputs/experiment-.../restyled"),
    numerical_threads=8,  # Optional; this is the default.
)
```

The destination must be new and outside `data/`; omitting it selects the run's
`results/` directory, which fails if it already exists. Recorded data, saved
settings and environment, and existing results remain unchanged. No simulations,
tree refitting, prototype reselection, or distance recomputation occur. This is
reanalysis with the current analysis code, including bootstrap inference when
configured, not just aesthetic restyling. For descriptive tables and diagnostic
plots only, use `write_reports(load_records(run_dir / "data"), destination)`
with `write_reports` from `report` and `load_records` from `record_io`.

For development, run the focused tests without starting the full experiment:

```console
uv run pytest examples/system_behavior_learning/tests -q
```

## Design and information regimes

Every replicate draws independent uniform fitting and assessment scenarios.
Target coordinates are standardized using fitting trajectories only; constant
coordinates are omitted. A behavior tree maps scenarios to complete
standardized trajectories. A requested capacity is an upper bound: the
behavior tree can realize fewer leaves when no useful split remains. Its
realized leaf count is recorded as `suite_size` and becomes the common budget
for every method in that comparison. Matching cardinality does not equalize
information access or simulation cost.

Each replicate also fits one otherwise matching tree without a leaf limit. Its
prediction, assessment-weighted realization, fitting support, occupancy, and
path geometry are written to the three `unbounded_*` tables. This diagnostic is
kept separate from capacity summaries, plots, and reference-coverage suite
comparisons.

The methods intentionally have different information regimes:

- **Behavior midpoint** uses a behavior-fitted tree and simulates the verified
  midpoint of each float32-aware root-to-leaf input box.
- **Archive PAM** uses pairwise trajectory distances in the fitting archive.
- **Input-only midpoint** partitions normalized fitting inputs without seeing
  trajectories.
- **Scrambled Sobol, IID random, and scrambled strength-1 Latin hypercube** use
  only the declared input bounds. Their planned repetitions use fixed derived seeds
  and are shared across historical replicates and requested capacities for the
  same system and realized suite size.

No method sees assessment or reference trajectories during construction.

## Execution policy

Every original member of a planned batch is attempted exactly once. There is
no survivor-only scoring, retry, replacement, or adaptive repair. Fitting needs
the complete fitting batch and a defined fitting-only transform. Assessment
failure blocks held prediction and assessment-weighted realization, but not
fitting, tree geometry, successful midpoint diagnostics, or reference coverage.
Reference failure blocks coverage and assignments, not tree diagnostics or suite
execution. Aggregate midpoint realization needs both complete midpoint and
assessment batches; successful per-leaf midpoint errors remain available when
another midpoint fails. Unbounded failures do not block bounded comparisons.

A matched suite is eligible only with exactly B generated, exact-unique scenario
coordinates and B successful trajectories, where B is the behavior tree's
realized leaf count. Distinct scenario coordinates with identical physical
traces remain eligible: trace uniqueness is diagnostic, not an exclusion rule.
Archive PAM reuses its selected fitting trajectories and stores their indices;
it performs no replay simulations. Geometric batches, including failures, are
cached once per system, method, realized budget, and repetition.

`suite_status.csv` has one finalized row per planned capacity/method/repetition:
`valid`, `structurally_infeasible`, `invalid_execution`, `duplicate_scenarios`,
`dependency_blocked`, or an explicit fit/generation failure. An input-only tree
with too few leaves retains its actual leaf count but has no simulations, raw
suite arrays, or coverage metric. If behavior fitting is blocked or fails,
`intended_size` is empty because B is unknown, not the requested capacity.

Counters distinguish evidence from new execution:

- `actual_size`: generated scenario rows (actual leaf count for structurally
  infeasible input trees).
- `attempted_count`: original simulator calls for the batch, including failures.
  Cached geometric statuses describe that original execution; summing their
  repeated uses does not count new simulations.
- `successful_count`: available valid physical trajectories, including archive
  reuse.
- `unique_scenario_count`: exact unique generated scenario coordinates.
- `unique_valid_trajectory_count`: exact unique valid physical traces; diagnostic
  only.
- `reused_count`: selected archive trajectories, B for PAM with
  `attempted_count=0`; zero for simulated suites.

`execution_status.csv` records reference/fitting/assessment and unbounded
midpoint batch yields, plus transform/fit failures and undefined-ratio reasons,
with system, replicate, capacity, and stage identity. NPZ evidence retains row
alignment: base batches use `<role>_valid` and `<role>_errors`, and suites use
`__valid` and `__errors`. Errors are Unicode arrays readable with
`allow_pickle=False`. Invalid trajectory rows contain NaN placeholders only;
these never enter transforms or scoring. A blocked transform has no transform
arrays; its absence is explained in execution status, not by a fake transform.

## Reading the outputs

Prediction error is coordinate-RMS over retained standardized trajectory
coordinates. The historical-mean ratio uses a fitting-only mean predictor;
values below one indicate lower assessment error for the tree. Representative
realization compares each behavior-box midpoint trajectory with its leaf
prototype and with held assessment members routed to that leaf. Both aggregate
distances are assessment-empirical means: every assessment scenario contributes
once, so leaf midpoint errors are weighted by assessment occupancy. The table
also reports the total path-box volume of leaves not reached by the assessment
sample. Missing evidence and zero-denominator ratios are empty fields, not
infinities or numerical placeholders. Header-only tables represent no valid
rows; plots annotate `no valid results` where appropriate.

`leaf_metrics.csv` preserves each leaf's depth, domain volume, fitting and
assessment occupancy, midpoint distance, held-member distance, and realization
ratio. Empty assessment leaves retain their midpoint distance and have empty
held-distance and ratio fields. `leaf_boxes.csv` records the lower and upper
bound, endpoint inclusivity, and midpoint for every leaf-parameter pair.

The prototype figure for each system shows all fitting traces routed to each
selected leaf in gray, their physical-coordinate mean/tree prototype in blue,
and the simulated path-box midpoint trace in dashed orange. Rows are leaves and
columns are state variables, with shared state-wise vertical limits. The
long-form `prototype_trajectories.csv` preserves every plotted value, including
the fitting-member index, role, time, state, and leaf, so these explanatory
figures can be restyled without rerunning the experiment.

Reference coverage reports the mean, 95th percentile, 99th percentile, and
maximum nearest-suite trajectory distance, plus suite-member use and normalized
input spacing. Smaller trajectory distances mean denser coverage of this fixed
reference sample; spacing describes input geometry, not behavioral
quality. For complete geometric groups, descriptive summaries and plots first
take the median over planned repetitions within each historical replicate,
then summarize across replicates. If any planned repetition is invalid and any
individual measurement is valid, the **entire system/capacity/method group** is
withheld across historical replicates, including complete banks. `summary.csv`
marks these groups `incomplete_planned_evidence` with blank `n` and statistics;
plots annotate the omitted summaries. Otherwise, `summary_status` is `available`
with usable replicate count `n`, or `no_valid_results` with `n=0` and empty
statistics for entirely failed groups. Failed planned denominators remain in
`suite_status.csv`. All individual valid suite measurements remain in
`suite_metrics.csv` regardless of bank completeness.

Matrix trajectory distances use the fast Gram identity on fitting-standardized
targets; input-spacing distances use normalized scenario coordinates. This
trades general numerical robustness for runtime: near-zero distances can show
roundoff, and large common offsets must be removed before using these helpers.
Zero or nonfinite coverage scores still withhold affected inference; no positive
epsilon is added to make a score usable.

### Paired coverage comparisons

The coverage score Q is the mean nearest-reference distance. Within each
history, compare behavior Q with the matched-budget comparator Q; for a
geometric method, first take the median of its planned repetition Qs. The
reported effect is `exp(median(log(Q_behavior) - log(Q_comparator)))` across
histories. A ratio below one favors behavior; 0.8 means 20% lower error under
this typical-history summary. It is not a ratio of separately summarized method
errors. `summary.csv` remains descriptive and is not this paired estimator.

`coverage_ratios.png` emphasizes Sobol, input-only midpoint, and archive PAM.
Random and LHS retain supporting paired estimates in the tables, without
uncertainty bands. The configured approximate 95% band covers all requested
capacities for **one system/comparator curve**, not all systems or comparators
jointly. It uses 10,000 bootstrap recalculations of whole histories, shared
reference rows, and independently resampled repetitions for each actual
method/budget bank. A bank's repetition choices are reused wherever that bank
occurs, while its reference distances remain history-specific. Suite members
are not independently resampled. The band uses the 95th percentile of the
largest absolute deviation from the observed log-effect over the capacity
curve, then converts its endpoints back to ratios.

Every paired point requires all planned histories and repetitions. Missing
required evidence withholds that point and the affected whole-curve band, not
unrelated comparisons. Zero or nonfinite Q entering a logarithm makes that
comparison unavailable; no small constant is added. Any undefined bootstrap
draw withholds the band without dropping or redrawing it, while valid observed
points remain. A zero log-effect (ratio one) or a zero-width band is distinct
from a zero error entering a logarithm. Ratios outside floating-point range
have separate presentation reasons; finite log-effects remain available.

Within each `results/inference/<system>/` directory:

- `method_scores.csv` retains every planned repetition's Q and availability
  reason, including valid zero scores.
- `paired_effects.csv` records history-level errors, paired effects, ratios,
  realized budgets and reasons. Budget zero means unknown, not an empty suite.
- `capacity_effects.csv` records point estimates, planned/valid history counts,
  band status/endpoints and attempted/invalid bootstrap counts. Unavailable
  numeric fields are empty, with reasons.
- `bootstrap.npz` preserves every attempted log-effect curve in draw order,
  including invalid entries and their masks/counts, plus capacity/budget axes
  and degeneracy diagnostics. Read it with `allow_pickle=False`.
- `metadata.json` identifies the system, configuration, and seed derivation.
  The run/report entry points also record the applied overall
  `numerical_threads` runtime cap here, not in `settings.json`. It is written
  last after that system's inference files are complete.

Prediction, realization, leaf/prototype diagnostics and raw method summaries
remain descriptive. Development bands demonstrate the analysis pipeline, not
confirmatory evidence. Coverage inference depends on its sampling and bootstrap
assumptions; neither it nor the other outputs establishes causal importance,
safety coverage or universal method superiority. The reference maximum is a
finite-sample observation, not a population worst-case guarantee.
