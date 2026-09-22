# API Reference

Flowcean's public API is imported from its packages, not from the top-level `flowcean` module.
Choose a package below for its exported classes, functions, and signatures.
For concepts and workflows, start with the [user guide](../user_guide/overview.md).

## Foundations

- [Core](core.md): environments, learners, models, metrics, transforms, callbacks, and learning strategies.
- [Hybrid systems](hybrid.md): system definitions, simulation, benchmarks, and HyDRA identification.
- [Polars](polars.md): dataframe environments, time-series processing, and transforms.

## Learners and models

- [scikit-learn](sklearn.md): regression learners, model wrappers, and evaluation metrics.
- [River](river.md): incremental learning.
- [PyTorch](torch.md): neural networks and Lightning training.
- [PySR](pysr.md): symbolic regression.
- [XGBoost](xgboost.md): boosted classifiers and regressors.
- [PalaestrAI](palaestrai.md): Soft Actor-Critic learning for active environments.
- [AALpy](aalpy.md): passive Mealy and Moore automata learning with RPNI.
- [Ensemble](ensemble.md): combined and cluster-based learners and models.

See [installation](../getting_started/installation.md) for optional backend dependencies.

## Integrations and tools

- [Adapters](adapters.md): dataframe and OPC interfaces for model deployment.
- [ROS](ros.md): loading ROS bag data.
- [Mosaik](mosaik.md): active energy-system environments.
- [Testing](testing.md): model tests, input domains, and predicates.
- [Utilities](utilities.md): experiment initialization, random seeds, and prediction loops.
