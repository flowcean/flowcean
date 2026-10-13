---
icon: lucide/workflow
---

# Dainarx Sampled-System Identification

Dainarx learns locations with sampled NARX dynamics, transition guards, and post-switch reset predictors from independent input/output traces. Unlike a native hybrid `Flow`, a NARX predictor computes the next observation directly; use its `rollout`, not the continuous-time hybrid simulator.

Run the self-contained example from the repository root:

```bash
just examples-dainarx
```

[`examples/dainarx/run.py`](https://github.com/flowcean/flowcean/blob/main/examples/dainarx/run.py) generates a noise-free two-location sampled system with input-dependent guards and output jumps. It trains on four independent traces and predicts a fifth. No external dataset or reference implementation is needed.

## Fit Independent Traces

Each trace is an ordered Polars frame. Select numeric output, input, and time columns explicitly; other columns are ignored and frames are not modified. Timestamps must be uniformly increasing with the same sampling period across traces. Do not concatenate independent traces: output histories must not cross their boundaries.

```python
from flowcean.hybrid.dainarx import fit_dainarx

fit = fit_dainarx(
    training,  # a list of independent Polars frames
    input_features=["u"],
    output_features=["x"],
    time_column="time",
    order=1,
    window=6,
    guard_kernel="linear",
    guard_c=1e4,
    guard_gamma="scale",
)
```

`fit.segments` records half-open row intervals, trace indices, and learned location IDs. `fit.model.locations` contains their fitted dynamics; `fit.model.transitions` contains output-and-input SVM guards and reset predictors. `fit_narx` is also available when all supplied traces are already known to share one dynamics model.

The linear basis includes all outputs at every lag, current inputs, and a bias. For nonlinear dynamics, supply numerical callables returning one component per output. For example, a scalar second-order cubic basis is:

```python
nonlinear_terms = [
    lambda history, current_input: history[-1] ** 3,
    lambda history, current_input: history[-2] ** 3,
]
```

History is always oldest first, with shape `(order, outputs)`. Inputs have shape `(inputs,)`; each term returns shape `(outputs,)`. With this cubic basis, one output, one input, and order two, there are six coefficient columns per output; `window=10` supplies eight regression targets. In general, `window - order` must exceed the coefficient count. Rank deficiency alone is not rejected.

## Predict Recursively

Supply exactly `order` seed output observations, the learned location at the last seed, and a timestamped known-input frame aligned with **all** rows, including seed rows. No later true outputs or labels are needed:

```python
predicted = fit.model.rollout(
    held_out.select("time", "u"),
    output_history=held_out.select("x").head(1).to_numpy(),
    initial_location=fit.segments[0].location_id,
)
```

Here all generated traces start in the same physical location, so the first training segment provides its learned ID. For other datasets you must supply the initial location yourself. The last seed must not lie inside an unfinished reset sequence.

The returned Polars frame contains timestamps, predicted outputs (including seeds), and `location_id`. Location IDs before the last seed are null because they were not supplied. Guards evaluate the current predicted output and current input. A switch affects the next row: its `order` reset predictors generate the first post-switch outputs before destination dynamics resume. These predictors are not physical instantaneous reset maps. Multiple enabled guards or a switch interrupting a reset sequence raise an error.

## Acceptance and Evidence

This noise-free implementation follows [Yu et al.'s Dainarx method](https://arxiv.org/html/2507.16426v3), with one explicit adaptation: ordinary least squares is retained, but fitting acceptance uses **maximum per-output RMSE** rather than the paper's unnormalized L2 residual. The threshold is `1e-6 * dt * max(abs(training_outputs))`, in output units.

Every whole segment is screened and every candidate cluster is refitted using all its segments, without crossing segment histories. RMSE acceptance is not subset-monotone; deterministic merge order does not imply unique recovery. Insufficient or rejected segment/reset evidence raises `ValueError` rather than silently dropping samples or inventing identity resets. A reset phase needs at least as many observed transition occurrences as coefficient columns. Only observed distinct-location transitions are learned.

See the [hybrid API reference](../reference/hybrid.md#flowcean.hybrid.dainarx) for signatures and fitted-model attributes.
