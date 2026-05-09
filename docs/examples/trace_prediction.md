# Trace Prediction

This example turns a recorded thermostat trace into a supervised prediction problem. It loads the CSV with `DataFrame.from_csv`, preserves time order with `TrainTestSplit(shuffle=False)`, applies `SlidingWindow(window_size=4)`, trains a `RegressionTree` with `learn_offline`, and evaluates the held-out windows with `evaluate_offline`.

Run it from the repository root:

```bash
uv run --directory ./examples/trace_prediction python run.py
```

The output starts with the data flow and then prints the model results:

```text
Trace prediction example
source trace rows: ...
transform: SlidingWindow(window_size=4)
training windows: ...
test windows: ...
evaluation:
prediction preview:
```

Regenerate the deterministic trace data with:

```bash
uv run --directory ./examples/trace_prediction python generate_data.py
```

Reusable notebook snippet:

```python
from pathlib import Path

from flowcean.core import learn_offline
from flowcean.polars import DataFrame, SlidingWindow, TrainTestSplit
from flowcean.sklearn import RegressionTree

inputs = [
    f"{column}_{step}"
    for step in range(3)
    for column in ("temperature", "target", "heating")
]
outputs = ["temperature_3"]

trace = DataFrame.from_csv(Path("examples/trace_prediction/data/thermostat_trace.csv"))
train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(trace)
window = SlidingWindow(window_size=4)

model = learn_offline(
    train | window,
    RegressionTree(max_depth=5, random_state=42),
    inputs,
    outputs,
)
```

Key Flowcean pieces:

- `DataFrame` wraps the lazy Polars CSV data as an offline environment.
- `TrainTestSplit(shuffle=False)` creates train and test environments without disrupting trace order.
- `SlidingWindow(window_size=4)` builds supervised rows from sequential samples.
- `learn_offline` trains the learner on the training environment.
- `evaluate_offline` computes regression metrics on the test environment in the full script.
