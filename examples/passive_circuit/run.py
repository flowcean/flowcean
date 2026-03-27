#!/usr/bin/env python

from pysr import PySRRegressor

import flowcean.cli
from flowcean.core import (
    evaluate_offline,
    learn_offline,
)
from flowcean.hydra import HyDRALearner
from flowcean.polars import DataFrame
from flowcean.pysr import PySRLearner
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError


def main() -> None:
    flowcean.cli.initialize()

    train = DataFrame.from_uri("file:./data/circuit_data.csv")
    regressor = PySRLearner(
        model=PySRRegressor(niterations=10, verbosity=0),
    )
    learner = HyDRALearner(
        regressor_factory=lambda: regressor,
        threshold=1e-5,
        start_width=400,
        step_width=200,
    )
    inputs = ["U1", "U2", "U3", "R"]
    outputs = ["I1"]

    model = learn_offline(
        train,
        learner,
        inputs,
        outputs,
    )

    test = DataFrame.from_uri("file:./data/circuit_eval_data.csv")

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
