---
icon: lucide/workflow
---

# Learning with Dainarx

This example learns a system that follows two different rules. An input command switches it from one location to the other. Dainarx receives the recorded inputs and outputs, but not the location labels. We then use the learned model to predict a new trace.

## Run the Example

From the repository root:

```bash
just examples-dainarx
```

The [example script](https://github.com/flowcean/flowcean/blob/main/examples/dainarx/run.py) generates its own noise-free data: four traces for learning and a fifth for prediction. Each trace contains time stamps in `time`, input values in `u`, and output values in `x`. Input commands `+2` and `-2` make the system switch locations, changing both the output and the rule it follows.

## Learn from the Traces

Dainarx looks for changes in the traces and groups the parts that follow the same rule. It also learns when to switch locations and how the output changes at a switch.

The script passes the four training traces as separate Polars frames. That way, the final output of one trace is not used to predict the first output of another:

```python
from flowcean.hybrid.dainarx import fit_dainarx

fit = fit_dainarx(
    training,
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

Here, `order=1` tells the model to predict each output from the previous output and the current input. With `window=6`, Dainarx examines six observations at a time to look for changes. The guard options control how it learns the conditions for switching locations.

## Predict a New Trace

The fifth trace was not used for learning. To start predicting it, we supply its first output, its starting location, and its input values:

```python
predicted = fit.model.rollout(
    held_out.select("time", "u"),
    output_history=held_out.select("x").head(1).to_numpy(),
    initial_location=fit.segments[0].location_id,
)
```

All traces in this example start in the same location, so we take its learned ID from the first training segment. For another dataset, you need to supply the starting location yourself.

After the first output, the model uses its own predictions to calculate each next output. It does not read the remaining recorded outputs. Those are kept for comparison. The returned frame contains the times, predicted outputs, and location IDs; it also includes the first output we supplied.

## Read the Results

The script prints the number of segments and learned locations, followed by the prediction error. You should see two locations. There are more segments: Dainarx splits each trace at the changes it detects, and several of those parts can belong to the same location.

The error is the root-mean-square error (RMSE) between the predicted and recorded outputs, excluding the first output used to start the prediction. This example has no noise and follows rules that the model can represent, so the error should be close to zero.

See the [API reference](../reference/hybrid.md#flowcean.hybrid.dainarx) for configuration options, including nonlinear terms. The method is described in the [Dainarx paper](https://arxiv.org/html/2507.16426v3).
