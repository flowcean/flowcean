---
icon: lucide/scan-search
---

# Simulated Hybrid System Identification

Learn flow models from room-temperature observations, then train a selector to choose between them. This example uses a thermostat that heats and cools around a time-varying target temperature. It supplies two independently simulated traces to HyDRA, with temperature, its derivative, and the target signal sampled every `0.02` time units.

## Run the Example

From the repository root:

```bash
uv run --directory ./examples/simulated_hybrid_system python run.py
```

The example uses PySR and requires Julia. The first run can take time to install and compile dependencies. The script explicitly displays the reference trajectory; close that plot to continue with learning. Its interactive `PlotCallback` shows one trace's discovery progress: plain bands are finalized flows, hatching is provisional grouping, and a dotted outline marks the active window. For headless use, set `show=False` on the callback; this also disables explicit redraws, GUI event processing, and pauses, leaving drawing or saving to your application.

## Read the Results

HyDRA learns the temperature derivative `dx` from time `t`, temperature `x`, and target temperature `target`. Each entry in `result.flows` has a fitted `model` and accepted `segments` representing shared continuous behavior across the traces. Its position is the flow ID.

`HyDRALearner.learn` returns only after assigning every supplied observation. If a segment cannot be identified, it raises `HyDRAIdentificationError` with the failing trace index and half-open row bounds in `.segment`; the script lets this exception propagate.

`result.to_labeled_frames(frames)` adds non-null `flow_id` labels to the original observations. The example trains a decision-tree selector using current temperature, previous temperature, and current target temperature. The previous observation helps distinguish heating from cooling. Batch prediction then uses the selector to route observations to flow models; `batch_prediction_rows_after_warmup` reports how many rows have sufficient history for prediction.

The script prints selector summaries and the decision tree. Diagram branches label whether the displayed predicate is true or false; leaf colors identify flows within that fitted selector, not corresponding native locations. When SVG export is available, it writes `examples/simulated_hybrid_system/outputs/selector_tree.svg`.

For multiple discovered flows, the example finishes after batch prediction: learned-model rollout currently requires a selector using current features only. If a single flow is discovered, the script also integrates that model on the reference time grid. Shared [trace comparison](../user_guide/evaluation.md) excludes the supplied initial temperature, prints per-temperature error metrics and coverage, and plots reference and prediction in `examples/simulated_hybrid_system/outputs/learned_vs_reference.png`.

See [Move from Simulation to Learning](../user_guide/hybrid_systems.md#move-from-simulation-to-learning) for the discovery and prediction API workflow.
