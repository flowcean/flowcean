# Simulated Hybrid System HyDRA Example

This example runs a full HyDRA loop on a known one-dimensional hybrid system:

1. Simulate a reference hybrid system with two affine modes.
2. Learn mode dynamics from sampled state and derivative columns.
3. Train a decision-tree selector for learned mode assignment.
4. Simulate the learned `HyDRAModel` on the reference time grid.
5. Compare the learned trajectory with the original reference trace.

The learner uses PySR, which requires a working Julia installation. The first run can take longer while Julia packages are resolved and compiled.

The example passes `HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))` to the learner so the trained `HyDRAModel` knows how to simulate a `Trace` from the learned time, state, and derivative columns.

Run the example from the repository root:

```sh
uv run --directory ./examples/simulated_hybrid_system python run.py
```

Options:

- `--output-dir PATH`: write selector and comparison artifacts to `PATH` instead of `outputs`.
- `--verbose`: show HyDRA learner progress logs.

Expected output includes a summary dictionary with the row count, locations, mode count, input features, and output features. When a selector is learned, the script prints `selector_summary`, `selector_mode_summary`, `selector_leaf_summary`, `selector_tree`, and `selector_svg` diagnostics. It also prints a `learned trace comparison` block with `mae`, `rmse`, and `max_error`, followed by `trace_comparison_plot`.

Artifacts are written under the selected output directory. By default, this creates `outputs/selector_tree.svg` and `outputs/learned_vs_reference.png` inside the example directory.
