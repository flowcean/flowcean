# Overview

Flowcean is a research-oriented Python toolkit for defining, simulating, identifying, evaluating, and reproducing models of cyber-physical systems.

Flowcean originates from research into automatic model generation for CPS. Its modular architecture combines environments, transforms, learning strategies, metrics, adapters, and integrations with established model-learning tools. Hybrid dynamical systems are a first-class modeling approach within this broader toolkit, not a requirement for every Flowcean study.

## Modeling Approaches

Flowcean supports different study designs rather than prescribing a single pipeline.

### Hybrid-System Modeling

Use [`flowcean.hybrid`](hybrid_systems.md) to define locations, continuous dynamics, event surfaces, transitions, resets, parameters, and initial conditions. Systems can be simulated to produce traces, while HyDRA can identify hybrid mode dynamics and selectors from observed data.

### Data-Driven Model Learning

Compose [learning strategies](learning_strategies.md) with recorded datasets, incremental streams, active environments, simulations, or connected CPS data sources. Backends such as scikit-learn, PyTorch, River, PySR, and external learners provide concrete learning algorithms.

### Evaluation and Integration

Evaluate models with task-appropriate metrics or trajectory comparisons. Environments, transforms, learners, models, metrics, and adapters can be combined according to the needs of each study and reused across hybrid and non-hybrid applications.

## Main Interfaces

- [`flowcean.hybrid`](hybrid_systems.md) provides hybrid-system definitions, simulation, trace conversion, plotting, benchmarks, and HyDRA identification.
- [`flowcean.core`](../reference/flowcean/core/index.md) provides environments, learners, models, metrics, transforms, callbacks, and learning strategies.
- [`flowcean.polars`](../reference/flowcean/polars/index.md) provides dataframe environments, datasets, time-series support, and reusable transforms.
- Backend packages such as `flowcean.sklearn`, `flowcean.river`, `flowcean.torch`, and `flowcean.pysr` connect concrete learning algorithms.
- Adapters and `flowcean.grpc` connect Flowcean studies to CPS data sources and external learners.

See the [modules](modules.md) for the established conceptual architecture, the [API reference](../reference/flowcean/index.md) for implementation details, and the [examples](../examples/hs_simple.md) for runnable studies. The initial Flowcean concepts and research context are presented by Knitt et al. [^1].

[^1]: Knitt, Markus, Swantje Plambeck, Jan Christian Wieck, Julian Kohlisch, Stephan Balduin, Eric MSP Veith, Jakob Schyga, Johannes Hinckeldeyn, Goerschwin Fey, and Jochen Kreutzfeldt. "Towards the Automatic Generation of Models for Prediction, Monitoring, and Testing of Cyber-Physical Systems." In 2023 IEEE 28th International Conference on Emerging Technologies and Factory Automation (ETFA), 1-4, 2023. [doi:10.1109/ETFA54631.2023.10275706](https://doi.org/10.1109/ETFA54631.2023.10275706).
