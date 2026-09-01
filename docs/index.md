---
hide:
  - navigation
---

# Flowcean

Flowcean is a research-oriented Python toolkit for defining, simulating, identifying, evaluating, and reproducing models of cyber-physical systems.

Flowcean builds on research into automatic model generation for CPS. Its modular environments, transforms, learning strategies, metrics, adapters, and backend integrations support a broad range of data-driven modeling studies. First-class support for hybrid dynamical systems adds explicit system structure and simulation to this established foundation without requiring every CPS model to be hybrid.

## Capabilities

- Define and simulate hybrid systems with continuous dynamics, events, transitions, and resets.
- Identify hybrid dynamics and mode selectors with HyDRA.
- Learn models from datasets, incremental streams, active environments, simulations, or connected CPS data sources.
- Compose reusable environments, transforms, learners, models, metrics, adapters, and evaluation strategies across studies.

## Start Here

Install Flowcean from PyPI:

```sh
pip install flowcean
```

Then choose a path:

- Follow the [installation guide](getting_started/installation.md) for user and developer setups.
- Build the [minimal hybrid system](examples/hs_simple.md).
- Browse the [hybrid systems benchmark gallery](examples/hybrid_systems.md).
- Run the [simulated hybrid system identification](examples/simulated_hybrid_system.md) workflow.
- Explore the [user guide overview](user_guide/overview.md), [modules](user_guide/modules.md), and [learning strategies](user_guide/learning_strategies.md) for Flowcean's general model-learning toolkit.

## Citation

If you use Flowcean in research, please consider citing:

- Towards the Automatic Generation of Models for Prediction, Monitoring, and Testing of Cyber-Physical Systems, IEEE International Conference on Emerging Technologies and Factory Automation (ETFA), 2023.
- Flowcean - Model Learning for Cyber-Physical Systems, Italian Workshop on Artificial Intelligence and Applications for Business and Industries (AIABI) at AIxIA, 2024, ArXiv, [abs/2603.12015](https://arxiv.org/abs/2603.12015).

## Acknowledgement

This work has been funded by BMBF project AGenC no. 16IS22047A.

![BMBF](assets/BMBF-light.svg#only-light)
![BMBF](assets/BMBF-dark.svg#only-dark)

[Impressum](impressum.md)
