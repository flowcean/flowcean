#!/usr/bin/env python

from pathlib import Path

from tqdm import tqdm

import flowcean.cli
from flowcean.core import (
    ChainedOfflineEnvironments,
    evaluate_offline,
    learn_offline,
)
from flowcean.grpc import GrpcPassiveAutomataLearner
from flowcean.polars import (
    DataFrame,
    Explode,
    Select,
    ToTimeSeries,
    TrainTestSplit,
    Unnest,
    collect,
)
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError


def main() -> None:
    flowcean.cli.initialize()

    data = ChainedOfflineEnvironments(
        [
            DataFrame.from_uri("file:" + path.as_posix()) | ToTimeSeries("t")
            for path in tqdm(
                list(Path("./data").glob("*.csv")),
                desc="Loading environments",
            )
        ],
    )
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(
        collect(data),
    )

    learner = GrpcPassiveAutomataLearner.run_docker(
        image="ghcr.io/flowcean/flowcean/java-automata-learner:latest",
        pull=False,
    )
    inputs = ["input"]
    outputs = ["output"]

    model = learn_offline(
        train,
        learner,
        inputs,
        outputs,
    )

    model.post_transform |= (
        Explode(["output"]) | Unnest(["output"]) | Select(["value"])
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
