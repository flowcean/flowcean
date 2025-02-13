#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
# ]
#
# [tool.uv.sources]
# flowcean = { path = "../../", editable = true }
# ///

import flowcean.cli
from flowcean.core import evaluate_offline, learn_offline
from flowcean.grpc import GrpcLearner
from flowcean.polars import DataFrame, TrainTestSplit
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError


def main() -> None:
    flowcean.cli.initialize_logging()

    data = DataFrame.from_uri(uri="file:./data/coffee_data.csv")
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    learner = GrpcLearner.run_docker(
        image="collaborating.tuhh.de:5005/w-6/agenc/agenc/java-automata-learner:latest",
    )
    inputs = ["^i.*$", "^o\\d$", "^o1[0-8]$"]
    outputs = ["o19"]

    model = learn_offline(
        train,
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
