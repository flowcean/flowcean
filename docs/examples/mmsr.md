---
icon: lucide/network
---

# Learning With MMSR

Multi-modal symbolic regression (MMSR) learns a known number of scalar symbolic input-output mappings and source-conditioned transition guards. Flowcean implements the clustered symbolic regression (CSR) and subsequent transition modeling (TM) stages of [Ly and Lipson (2012)](https://www.jmlr.org/papers/v13/ly12a.html), using the public PySR backend instead of Eureqa. These are separately usable library operations, not a continuous-time hybrid-system learner or a universal rollout model.

## Run the Example

From a repository checkout:

```bash
uv sync --extra pysr
just examples-mmsr
```

PySR requires Julia and may provision its runtime on first use. Ordinary `flowcean.hybrid` imports do not load PySR or Julia. The dedicated recipe runs serial, one-thread searches and can take several minutes; it is intentionally outside the aggregate example recipe and CI.

[`examples/mmsr/run.py`](https://github.com/flowcean/flowcean/blob/main/examples/mmsr/run.py) contains the fixed search budgets, operators, and separate initialization/search seeds. CSR and TM use the same predeclared per-search budget, with a continuing search-seed stream. The script prints the selected expressions, original tree complexities, variances, stopping diagnostics, guards, FaMoS tree rules and scales, and both recursive held-out errors. Errors propagate rather than being hidden behind a fallback predictor.

## Prepare Aligned Observations

`fit_mmsr_csr` accepts independent eager Polars frames, ordered `input_features`, one `output_feature`, a known `n_locations`, and a zero-argument factory returning a fresh configured `PySRRegressor`. Regressors can be measured states, not just exogenous controls. The library does not infer timestamps, sampling periods, derivatives, history lengths, or latent continuous states.

The example uses the unchanged native [thermostat](hybrid_systems.md#thermostat) with a fixed target. It samples temperature without location labels, forms each `(temperature_n, temperature_n+1)` pair within its own trace, and drops only the final unavailable target. Every switch-straddling interval remains. Constant target is omitted as a regressor. The learned functions are sampled next-temperature mappings, not vector fields, arbitrary-setpoint response, or physical reset maps.

## Inspect CSR and Fit Transitions

CSR couples fresh weighted symbolic searches with soft Gaussian memberships and unbiased weighted residual variances. It selects each update from the candidate frontier by complete-mixture AIC, not PySR's default best equation. Search budgets and operators belong to the factory; method-owned settings and numerical policies are documented in the [API reference](../reference/hybrid.md#flowcean.hybrid.mmsr).

The returned `MMSRCSR` exposes `expressions`, `complexities`, `variances`, `memberships`, and `diagnostics`. `predict(frame)` evaluates all location mappings. `membership(frame)` needs both regressors and observed targets; it is not an input-only classifier. Each training membership matrix retains the original trace's rows and has shape `(n_rows, n_locations)`.

Pass the original ordered traces and CSR artifact to `fit_mmsr_transitions`, supplying a fresh search factory. TM uses only within-trace adjacency and learns every ordered nonself edge from soft memberships. Each source's departure/stay-balanced weights are shared across its outgoing edges. A returned `MMSRTransition` exposes its source and destination locations, raw symbolic `expression`, original `complexity`, weighted sigmoid MSE, AIC, and `guard(frame)`, which evaluates `expression >= 0`.

Missing source departure/stay evidence or an unusable edge raises an error; the CSR artifact remains independently available. The edge collection does not define implicit self edges, competing-guard arbitration, or probability propagation.

## Compare With FaMoS Trees

The same example fits CSR once and holds its sampled dynamics fixed while comparing symbolic TM guards with [FaMoS next-location trees](famos.md). TM learns from soft memberships; the trees learn from explicit hard labels obtained by taking each stored training membership row's argmax, with ties resolved to the lowest positional location ID. These are different training targets, not interchangeable objectives.

Tree training uses the original sampled frames with one nullable integer `location_id` column. Each mapping row receives its stored membership label; the final raw sample remains null because no next-temperature target was observed for membership inference. Adjacent pairs with either label null are excluded without bridging gaps or traces. Every switch-straddling mapping interval remains. Only current `temperature` is a tree feature, not `next_temperature`; there are no exogenous input features in this fixed-target comparison. Here FaMoS `output_features` denotes current physical observations and `input_features` denotes controls, whereas MMSR `input_features` denotes regression features, which can include observations.

Both recursive evaluations use the exact same CSR artifact, initial observed temperature, and first-training-row argmax location. The trees return a next location directly, including self targets. This isolates transition selection on fixed dynamics, not the effect of refitting or tuning either learner.

## Interpret Timing and Results

Flowcean chooses source-time transition training: regressors at row n and location n predict location n+1. The paper's formal mapping convention elsewhere is inconsistent with this TM indexing, so this is an explicit implementation contract rather than a claim of paper-exact execution semantics.

The two-location example has exactly one outgoing guard per source. Each feedback loop computes both next temperature and next location from the old predicted temperature and old location, then advances both. The symbolic selector takes its outgoing edge when the guard is active and otherwise stays. All initial temperatures start in heating by construction. The predictor's initial location comes from the first training row's largest membership, not a held-out posterior or simulator location. Only the held-out initial temperature seeds prediction; future measurements are used solely for scoring. Both loops [score temperature predictions](../user_guide/evaluation.md) with NumPy after excluding that one seed row, reporting the errors and number of scored rows.

Finite fresh searches can worsen the incumbent and need not recover distinct functions. `error_stable` means five complete sweeps with less than 2% relative error change, not successful discovery; even collapsed mappings can be stable. `no_usable_candidate` preserves the last valid mixture and is not convergence. Invalid initialization or committed mixtures raise errors. Inspect the expressions and predictive behavior as well as memberships and losses.

Noise and overlapping behaviors can make membership inference ambiguous. A small soft transition loss alone does not establish correct guards, and missed switches can accumulate during recursive prediction. Rational expressions may have singularities outside the fitted domain; nonfinite evaluations raise rather than being silently repaired. No perfect-recovery or accuracy threshold is imposed by this example.
