# HyDRA Simulation Design

## Goal

Add first-class support for simulating a learned HyDRA model and comparing its generated state trace with a ground-truth hybrid-system trace.

The first iteration treats HyDRA as hybrid-system identification from trace-shaped data. A trace contains time, state samples, derivative samples, and optional external inputs. HyDRA learns continuous dynamics per discovered mode plus a selector that chooses the active mode. During simulation, the model integrates the selected learned dynamics and returns a `flowcean.ode.Trace`.

## Scope

In scope:

- Introduce a single trace schema object that describes the columns of HyDRA training traces.
- Store that schema on `HyDRALearner` and pass it into learned `HyDRAModel`s.
- Add `HyDRAModel.predict_next_state(...)` to integrate learned dynamics over one step.
- Add `HyDRAModel.simulate(...)` to roll out the learned hybrid system over a requested time grid.
- Add `compare_state_traces(...)` to compare ground-truth and HyDRA-simulated `Trace.x` values on matching time grids.
- Support recursive rollout: each integrated state becomes the starting state for the next step.
- Support one-dimensional learned HyDRA models and multi-output manually constructed `HyDRAModel`s when derivative width matches state width.

Out of scope for this iteration:

- Reclassifying `HyDRALearner` away from `SupervisedLearner` in the core type hierarchy.
- Adding a trace-native `learn_from_traces(...)` API.
- Training multi-output HyDRA models. `HyDRALearner` currently rejects multi-output targets, so multi-dimensional simulation with learned models remains limited until the learner supports vector-valued dynamics.
- One-step replay against ground-truth states.
- Comparing predicted mode labels against ground-truth locations.
- Recovering guard events or explicit `flowcean.ode.Transition` objects.
- Selector configurations that require history or warmup rows during simulation.
- Changing the existing `HyDRAModel.predict(...)` contract.

## Trace Schema

Replace the earlier simulation-specific config idea with a trace schema:

```python
@dataclass(frozen=True)
class HyDRATraceSchema:
    time: str
    state: tuple[str, ...]
    derivative: tuple[str, ...]
    inputs: tuple[str, ...] = ()
```

This schema is the single source of truth for column semantics:

- `time`: the trace time column.
- `state`: continuous state columns, in vector order.
- `derivative`: derivative columns that HyDRA learns as dynamics outputs, in vector order.
- `inputs`: optional external input columns, in vector order.

This metadata is necessary because training and prediction are tabular while simulation is vector-based. SciPy supplies a numeric state vector, but HyDRA mode models expect named Polars columns. The schema maps between those representations.

## Learning API

Extend `HyDRALearner.__init__` with `trace_schema: HyDRATraceSchema | None = None`.

Keep the current `learn(inputs, outputs)` method. When `trace_schema` is provided, validate at the start of learning that:

- `inputs` contains exactly `schema.time`, all `schema.state`, and all `schema.inputs` columns.
- `outputs` contains exactly `schema.derivative` columns in schema order.
- All schema column groups are disjoint.
- `len(schema.state) == len(schema.derivative)` for simulatable models.

For the current learner implementation, `len(schema.derivative)` must be `1` because `HyDRALearner._initialize_traces` is single-output-only. This keeps current behavior explicit. Multi-dimensional learned dynamics remain a follow-up.

The produced `HyDRAModel` stores the same schema. Manually constructed `HyDRAModel`s may also provide a schema directly; this is how iteration 1 can support multi-output simulation tests without changing the learner.

Longer-term, HyDRA should probably grow a trace-native API like `learn_from_traces(trace_frames)`, because it behaves more like hybrid-system identification than ordinary supervised regression. That is out of scope here so existing Flowcean strategy integration remains stable.

## Model API

Extend `HyDRAModel.__init__` with `trace_schema: HyDRATraceSchema | None = None`.

When a schema is provided, validate during model construction that:

- `model.input_features` contains exactly `schema.time`, all `schema.state`, and all `schema.inputs` columns. Order may differ because callers often choose a feature order for training, but all required columns must be present and no extras are allowed.
- `model.output_features` equals `list(schema.derivative)` so derivative vector order is unambiguous.
- Schema column groups are disjoint.

Export `HyDRATraceSchema`, `StateTraceComparison`, and `compare_state_traces` from `flowcean.hydra`.

Add methods to `HyDRAModel`:

```python
def predict_next_state(
    self,
    state: Iterable[float],
    *,
    t: float,
    dt: float,
    input_stream: InputStream | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_step: float | None = None,
) -> np.ndarray:
    """Integrate learned dynamics from `t` to `t + dt`."""

def simulate(
    self,
    t_span: tuple[float, float],
    x0: Iterable[float],
    *,
    input_stream: InputStream | None = None,
    capture_inputs: bool | None = None,
    sample_times: Iterable[float] | None = None,
    sample_dt: float | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_step: float | None = None,
) -> Trace:
    """Roll out learned dynamics over the requested sample grid."""
```

`HyDRAModel.predict(...)` remains unchanged and continues to predict derivatives for a Polars feature frame.

## Simulation Flow

`predict_next_state(...)` validates cheap constraints before integration:

- `trace_schema` is present.
- `len(state) == len(schema.state)`.
- `len(model.output_features) == len(schema.state)`.
- Multi-mode selectors do not request any history features that require warmup rows.
- `input_stream(t)` returns a 1D vector with `len(schema.inputs)` values when external inputs are configured.

Then it integrates one interval:

1. Build a one-row Polars frame from interval-start `t`, interval-start state, and interval-start external inputs.
2. Select the active mode once with `predict_with_diagnostics(...)`.
3. Hold that mode fixed for the interval.
4. Build a SciPy RHS that calls the selected flow model directly at internal solver evaluation times.
5. Each RHS call maps current numeric state to `schema.state`, current time to `schema.time`, and `input_stream(t)` to `schema.inputs`.
6. The selected flow model predicts derivatives in `schema.derivative` order.
7. Integrate with `solve_ivp(t_span=(t, t + dt), t_eval=[t + dt])` and return the single final state sample.

The per-interval mode selection is intentional. It keeps selector calls out of every adaptive solver substep, avoids making SciPy chase discontinuous selector changes inside one interval, and aligns the rollout with sampled traces used for HyDRA training. A future stateful simulation runtime can revisit this if guard/event recovery or continuous selector switching becomes necessary.

`simulate(...)` prepares a sample grid from either explicit `sample_times` or `sample_dt`. Exactly one grid source is required for the first iteration. It recursively integrates each adjacent interval. The state at the end of one interval becomes the starting state for the next interval.

The returned `Trace` contains:

- `t`: the requested sample grid.
- `x`: simulated continuous states.
- `location`: the selected mode label at each sample. For per-interval rollout, the label for the next sample is the mode selected at the start of the interval that produced it. Single-mode models use a stable label such as `"mode_0"`.
- `events`: an empty tuple because guard/event recovery is out of scope.
- `u`: sampled inputs when `capture_inputs` resolves to true.
- `dx`: omitted in the first iteration.

## Trace Comparison

Add comparison helpers in `flowcean.hydra.simulation`:

```python
@dataclass(frozen=True)
class StateTraceComparison:
    absolute_error: np.ndarray
    mae: float
    rmse: float
    max_error: float


def compare_state_traces(reference: Trace, predicted: Trace) -> StateTraceComparison:
    """Compare state samples from two traces on the same time grid."""
```

`compare_state_traces` uses `np.allclose(reference.t, predicted.t, rtol=0.0, atol=1e-12)` for time-grid compatibility. State shape checks remain exact.

## Error Handling

Raise `ValueError` when:

- Simulation is requested without `trace_schema`.
- `len(x0)` or `len(state)` does not match `len(schema.state)`.
- Derivative output width does not match the state dimension.
- Schema columns do not match model input/output features when schema is attached.
- `input_stream(t)` returns a non-1D vector or a vector whose length differs from `len(schema.inputs)`.
- The requested sample grid is missing, ambiguous, non-monotone, or outside `t_span`.
- Ground-truth and predicted traces have different time grids or state shapes during comparison.

Raise `RuntimeError` when SciPy integration fails.

Raise `NotImplementedError` before integration when the selector configuration requires unsupported history-dependent inference for simulation. For the first iteration this means rejecting selectors whose `feature_config.max_history > 0`, because a one-row interval-start selector frame cannot satisfy warmup requirements.

## Testing

Add focused tests under `tests/hydra/`:

- `HyDRATraceSchema` validates disjoint column groups and feature coverage.
- `HyDRALearner` accepts `trace_schema`, validates `learn(inputs, outputs)` column coverage, and passes the schema into `HyDRAModel`.
- A single-mode constant-derivative model integrates to the expected state.
- `HyDRAModel.simulate(...)` returns a `Trace` on the requested sample grid.
- Recursive rollout feeds integrated states back into later intervals.
- Simulation validates missing schema and state dimension mismatches.
- `compare_state_traces(...)` reports absolute error, MAE, RMSE, and max error.
- Trace comparison rejects mismatched time grids outside the documented tolerance.
- Multi-output manually constructed models can simulate multi-dimensional state when derivative width matches state width.
- Learned multi-dimensional simulation is rejected clearly while `HyDRALearner` remains single-output-only.
- A simple selector-backed model where the selector chooses one mode for the whole trace uses interval-start routing without requiring history.

## Follow-Up Work

- Add a trace-native `HyDRALearner.learn_from_traces(...)` API.
- Add multi-output HyDRA learning for vector-valued dynamics.
- Add one-step replay mode to measure local prediction error using ground-truth states.
- Compare predicted mode labels against ground-truth locations when label mappings are available.
- Add optional derivative capture during simulation.
- Support state, input, derivative, and mode history features with a stateful simulation runtime.
- Explore exporting a learned HyDRA model into an ODE-compatible hybrid-system abstraction if the `ode` module gains selector-style transitions.
- Add user-guide documentation once the API stabilizes.
