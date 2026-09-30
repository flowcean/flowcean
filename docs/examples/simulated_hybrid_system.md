---
icon: lucide/scan-search
---

# Simulated Hybrid System Identification

This example runs a HyDRA identification loop on simulated room-temperature traces. The reference is the thermostat benchmark, switching between heating and cooling around a time-varying target temperature. HyDRA learns flow models, trains a selector for flow assignment, simulates the learned model, and compares the learned rollout frame with the sampled reference frame.

The selector uses only current temperature, omitting the target signal and history. This limits its ability to reproduce the reference's switching behavior; the comparison measures the resulting approximation.

Run it from the repository root:

```bash
uv run --directory ./examples/simulated_hybrid_system python run.py
```

The learner uses PySR for symbolic regression. PySR requires Julia, and the first run can take longer while Julia packages are resolved and compiled.

The script performs these steps:

1. Create the two-location thermostat benchmark with heating and cooling flows.
2. Simulate it with a varying target temperature to retain a hybrid reference trajectory for plotting.
3. Sample that trajectory at `dt=0.02` with derivatives and rename `x0` and `dx0` to `x` and `dx` for learning.
4. Train a `HyDRALearner` with PySR regressors for flow models.
5. Train a `HybridDecisionTreeLearner` selector over the state feature `x`.
6. Simulate the learned `HyDRAModel` on the reference time grid.
7. Print selector diagnostics and state-trace comparison metrics.
8. Save selector and comparison plot artifacts.

The example passes `HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))` to the learner. This records which learned input column is time, which column is state, and which output column is the derivative. `HyDRAModel.simulate()` returns a grid-scheduled Polars frame with `flow_id` and `flow_time`, not native locations or a trajectory. The script renames its `x0` state to `x` before comparing it with the reference frame.

Expected printed output includes a summary dictionary containing `rows`, `locations`, `flow_count`, `input_features`, and `output_features`. If a selector is learned, the script also prints `selector_summary`, `selector_flow_summary`, `selector_tree`, and `selector_svg` diagnostics. The comparison block starts with `learned trace comparison` and reports `mae`, `rmse`, and `max_error`.

By default, artifacts are written to `examples/simulated_hybrid_system/outputs/selector_tree.svg` and `examples/simulated_hybrid_system/outputs/learned_vs_reference.png`.
