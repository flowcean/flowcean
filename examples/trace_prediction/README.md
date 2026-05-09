# Trace Prediction

This workshop example trains a supervised model on a recorded thermostat trace. Flowcean loads the CSV as a `DataFrame`, keeps the temporal order with `TrainTestSplit(shuffle=False)`, converts each split into supervised samples with `SlidingWindow(window_size=4)`, trains with `learn_offline`, and reports metrics with `evaluate_offline`.

Run the example from the repository root:

```bash
uv run --directory ./examples/trace_prediction python run.py
```

Expected output includes a short data summary and model report:

```text
Trace prediction example
source trace rows: ...
transform: SlidingWindow(window_size=4)
training windows: ...
test windows: ...
evaluation:
prediction preview:
```

Regenerate the deterministic CSV data when changing the synthetic trace:

```bash
uv run --directory ./examples/trace_prediction python generate_data.py
```

Key Flowcean pieces:

- `DataFrame.from_csv` wraps `data/thermostat_trace.csv` as an offline environment.
- `TrainTestSplit(shuffle=False)` preserves the trace order for train and test data.
- `SlidingWindow(window_size=4)` creates three-step input histories and the next-temperature target.
- `learn_offline` trains the `RegressionTree` on the training windows.
- `evaluate_offline` computes the held-out regression metrics.
