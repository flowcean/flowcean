# One-Tank Offline Learning

This example identifies a one-step predictor from a simulated water-tank trace. It uses the same system definition and simulation as the [incremental learning variant](one_tank_incremental.md), while training each model from a fixed dataset.

![The one-tank system](./images/one_tank.svg)

## System and Data

The water level $x$ follows

$$
\dot{x} = \frac{b V(t) - a \sqrt{x}}{A},
$$

where $A = 5$ is the tank area, $a = 0.5$ is the outflow rate, $b = 2$ is the inflow rate, and

$$
V(t) = \max\left(0, \sin\left(2 \pi t / 10\right)\right).
$$

[`examples/one_tank/system.py`](https://github.com/flowcean/flowcean/blob/main/examples/one_tank/system.py) is the single source of this simulation. `one_tank_system()` represents the dynamics as a one-location `HybridSystem` with no transitions, and `simulate_one_tank()` returns a deterministic trace sampled from 0 to 25 seconds at 0.1-second intervals.

Both learning variants consume the resulting `t` and `h` columns. A three-sample `SlidingWindow` creates `h_0`, `h_1`, and `h_2`; the learners predict `h_2` from the two preceding levels `h_0` and `h_1`.

## Offline Workflow

[`run_offline.py`](https://github.com/flowcean/flowcean/blob/main/examples/one_tank/run_offline.py) performs these steps:

1. Simulate the shared one-tank `HybridSystem`.
2. Apply a three-sample sliding window.
3. Shuffle and split the fixed data into 80 percent training and 20 percent test observations with seed 42.
4. Train a regression tree, a PyTorch multilayer perceptron, and a two-tree ensemble with `learn_offline`.
5. Evaluate each model with mean absolute error, mean squared error, and maximum error.

The script prints training times and evaluation reports. Reported values can vary across platforms and ML backend versions.

## Run

From the repository root:

```sh
uv run --directory ./examples/one_tank python run_offline.py
```

To run both the offline and incremental variants:

```sh
just examples-one_tank
```

See the [Hybrid Systems guide](../user_guide/hybrid_systems.md) for the simulation model and [Learning Strategies](../user_guide/learning_strategies.md) for offline learning concepts.
