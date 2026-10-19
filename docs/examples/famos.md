# Learning With FaMoS-DTW

Run from the repository root:

```bash
just examples-famos
```

This example learns sampled affine dynamics and next-location trees from the native [thermostat](hybrid_systems.md#thermostat), without simulator location labels. It uses unchanged thermostat physics and continuous event times, sampled on a uniform grid. All runs start in heating. Training and held-out initial temperatures differ; the target temperature is constant.

Edit the experiment settings in the [source](https://github.com/flowcean/flowcean/blob/main/examples/famos/run.py). It calls three independent stages:

1. `discover_famos_locations`: geometric segmentation and exact per-signal DTW clustering, with partial labels for derivative warmup or insufficient evidence.
2. `fit_famos_dynamics`: output-only affine ARX fits within discovered segments.
3. `fit_famos_transitions`: next-location decision trees, one per source location.

`FamosModel` composes the fitted artifacts for recursive prediction. Initialization uses one held-out temperature and a learned training location; subsequent measured temperatures are used only for scoring. Both the next output and next location use the current location, output and input. See the [API reference](../reference/hybrid.md#flowcean.hybrid.famos) for history shapes, scales and timing.

The script reports discovered locations, unassigned rows, fit ranks and training errors, then held-out recursive RMSE and maximum absolute error. These are diagnostics, not accuracy acceptance tests. A constant target is collinear with the affine bias, so rank deficiency is expected. This experiment does not identify arbitrary target response: the native target changes switching thresholds, not flows. Segmentation errors, sampled approximations to continuous events, and tree extrapolation can all affect recursive predictions.

The method follows the [FaMoS paper](https://doi.org/10.1145/3641513.3650131) and [official artifact](https://doi.org/10.5281/zenodo.10657936). Optional LMI refinement is not included. Output-only ARX fitting and source-conditioned trees are explicit adaptations, not a MATLAB wrapper.
