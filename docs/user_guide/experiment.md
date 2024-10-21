# Experiment

An **Experiment** is a structure definition.
It defines

- which *Environment* to load (i.e. are we learning based on a data set, a simulation or a data stream?),
- which *learning strategy* to apply,
- which *learner* should be selected and how it is configured,
- which *transforms* should be applied to the data,
- if previously trained models should be loaded and from where,
- where to save the model that is being trained,
- and how to evaluate the performance of the models.

The definition is done via a Python script, which is usually called `run.py`.

The following flow chart shows the procedure inside a `run.py`.

``` mermaid
graph LR
  A(CLI initialize) --> B(Specify
  Environment);
  B --> C{Apply transform
  to environment?};
  C --> |Yes| D(Specify and
  apply transforms);
  D --> E(Load
  Environment);
  C --> |No| E;
  E --> F(Decide on
  Learning Strategy);
  F --> G(Apply
  Learning
  Strategy)
  G --> H(Evaluate
  Model)
```

More information on learning strategies can be found [here](https://www3.tuhh.de/agenc/user_guide/learning_strategies/).
How the evaluation of models is done in Flowcean, is explained [here](https://www3.tuhh.de/agenc/user_guide/evaluation/).

Below, is a basic code implementation of an environment definition.
In this case, the environment is a *DataSet* which is a type of [OfflineEnvironment](https://www3.tuhh.de/agenc/reference/flowcean/core/environment/offline/).
Its [learner](https://www3.tuhh.de/agenc/user_guide/model/) is a linear regression algorithm.
It uses an incremental [Learning Strategy](https://www3.tuhh.de/agenc/user_guide/learning_strategies/).
In this example, no model is saved or loaded.
The evaluation strategy is defined by the `evaluate_offline()` function.

```python
import logging

import flowcean.cli
import polars as pl
from flowcean.environments.dataset import Dataset
from flowcean.environments.train_test_split import TrainTestSplit
from flowcean.learners.linear_regression import LinearRegression
from flowcean.metrics import MeanAbsoluteError, MeanSquaredError
from flowcean.strategies.incremental import learn_incremental
from flowcean.strategies.offline import evaluate_offline

logger = logging.getLogger(__name__)

N = 1_000


def main() -> None:
    flowcean.cli.initialize_logging()

    data = Dataset(
        pl.DataFrame(
            {
                "x": pl.arange(0, N, eager=True).cast(pl.Float32) / N,
                "y": pl.arange(N, 0, -1, eager=True).cast(pl.Float32) / N,
            },
        ),
    )
    data.load()
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    learner = LinearRegression(
        input_size=1,
        output_size=1,
        learning_rate=0.01,
    )
    inputs = ["x"]
    outputs = ["y"]

    model = learn_incremental(
        train.as_stream(batch_size=1).load(),
        learner,
        inputs,
        outputs,
    )

    report = evaluate_offline(
        model,
        test,
        inputs,
        outputs,
        [MeanAbsoluteError(), MeanSquaredError()],
    )
    print(report)


if __name__ == "__main__":
    main()

```
