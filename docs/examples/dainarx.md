---
icon: lucide/workflow
---

# Learning with Dainarx

This example applies Dainarx to the [Duffing oscillator benchmark](hybrid_systems.md#duffing-oscillator). A cosine force drives a nonlinear spring system whose damping and stiffness switch between locations. Switching also reduces its velocity. The gallery explains the physical model and shows its behavior.

The learner receives sampled position and force, but neither velocity nor the simulator's location labels. The aim is to explore identification on a standard benchmark, not to require a perfect fit.

## Run the Example

From the repository root:

```bash
just examples-dainarx
```

The [example script](https://github.com/flowcean/flowcean/blob/main/examples/dainarx/run.py) generates independent training traces with the native simulator and different initial states. It retains timestamps in `t`, the prescribed input in `force`, and the observed output in `position`. The script owns the experiment's initial conditions, sampling settings, and learner configuration.

## Read the Discovery Results

Dainarx first segments and groups the observations, fitting NARX predictors from position history, cubic features, and the current force. The script reports the discovered segments, learned locations, acceptance threshold, and ordinary location-fitting errors.

These are training results, not prediction errors on unseen trajectories. Successful dynamics discovery does not by itself provide an executable hybrid model: switching conditions and post-switch predictors still need to be learned.

## Interpret Incomplete Identification

With the current configuration, dynamics discovery completes, but transition fitting rejects a post-switch predictor because its training error exceeds the acceptance threshold. The script prints the reason and ends normally, explicitly reporting that no executable model or held-out prediction was produced.

This is a limitation of the current implementation and configuration. Completing the command is not a claim that identification succeeded.

If transition fitting succeeds, the script attempts a separate held-out rollout. It supplies initial position history, the starting location inferred from the first training segment, and the prescribed force. The chosen initial conditions start in the same physical location. Later recorded positions are used only to score predictions, not fed back into the learned model. Reported errors exclude the supplied initial history, and no near-zero error is required.

See the [API reference](../reference/hybrid.md#flowcean.hybrid.dainarx) for separate dynamics discovery, observation annotations, transition fitting, and rollout. The method is described in the [Dainarx paper](https://arxiv.org/html/2507.16426v3).
