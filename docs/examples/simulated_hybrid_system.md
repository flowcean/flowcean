# Simulated Hybrid System Identification

This example runs a full HyDRA identification loop on simulated one-dimensional hybrid-system traces. The reference system starts in one affine mode and switches to another when the state crosses a threshold. HyDRA learns mode dynamics, trains a selector for mode assignment, simulates the learned model, and compares that learned trajectory with the reference trace.

Run it from the repository root:

```bash
uv run --directory ./examples/simulated_hybrid_system python run.py
```

The learner uses PySR for symbolic regression. PySR requires Julia, and the first run can take longer while Julia packages are resolved and compiled.

The script performs these steps:

1. Build a two-location `HybridSystem` with affine continuous dynamics.
2. Simulate the system with derivative capture enabled.
3. Convert the trace to a Polars frame with `trace_to_polars`.
4. Train a `HyDRALearner` with PySR regressors for mode dynamics.
5. Train a `HybridDecisionTreeLearner` selector over the state feature `x`.
6. Simulate the learned `HyDRAModel` on the reference time grid.
7. Print selector diagnostics and state-trace comparison metrics.
8. Save selector and comparison plot artifacts.

The example passes `HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))` to the learner. This records which learned input column is time, which column is state, and which output column is the derivative, so `HyDRAModel.simulate()` can reconstruct a `Trace` after training.

Options:

- `--output-dir PATH`: write selector and comparison artifacts to `PATH` instead of `outputs`.
- `--verbose`: show HyDRA learner progress logs.

Expected printed output includes a summary dictionary containing `rows`, `locations`, `modes`, `input_features`, and `output_features`. If a selector is learned, the script also prints `selector_summary`, `selector_mode_summary`, `selector_leaf_summary`, `selector_tree`, and `selector_svg` diagnostics. The comparison block starts with `learned trace comparison` and reports `mae`, `rmse`, and `max_error`.

By default, artifacts are written to `examples/simulated_hybrid_system/outputs/selector_tree.svg` and `examples/simulated_hybrid_system/outputs/learned_vs_reference.png`. Use `--output-dir PATH` to choose another output directory.
