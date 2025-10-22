#!/usr/bin/env python

from river import tree

import flowcean.cli
from flowcean.core import (
    evaluate_offline,
    learn_offline,
)
from flowcean.hydra import HyDRALearner
from flowcean.polars import DataFrame
from flowcean.river.learner import RiverLearner
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError


def main() -> None:
    flowcean.cli.initialize()

    train = DataFrame.from_uri("file:./data/circuit_data.csv")
    regressor = RiverLearner(
        model=tree.HoeffdingTreeRegressor(grace_period=50, max_depth=5),
    )
    learner = HyDRALearner(regressor_factory=lambda: regressor, threshold=0.1)
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
