---
icon: lucide/sigma
---

# PySR

`flowcean.pysr` provides symbolic regression through PySR.
Install the `pysr` extra as described in [installation](../getting_started/installation.md).

Configure a `PySRRegressor`, wrap it in `PySRLearner`, and call `learn(inputs, outputs)` with lazy Polars frames. The inputs contain the explanatory variables and the outputs contain one target column. The returned `PySRModel` predicts that target and exposes the learned equation through `flow_summary()`.

```python
from pysr import PySRRegressor
from flowcean.pysr import PySRLearner

learner = PySRLearner(PySRRegressor(niterations=10))
model = learner.learn(inputs, outputs)
predictions = model.predict(inputs).collect()
print(model.flow_summary())
```

For [HyDRA discovery](../user_guide/hybrid_systems.md#discover-shared-flows), supply a factory that creates a fresh learner and regressor for each fit:

```python
regressor_factory = lambda: PySRLearner(PySRRegressor(niterations=10))
```

::: flowcean.pysr
