#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
# ]
#
# [tool.uv.sources]
# flowcean = { path = "../../", editable = true }
# ///

from pathlib import Path

from tqdm import tqdm

import flowcean.cli
from flowcean.core import evaluate_offline, learn_offline
from flowcean.core.environment.chained import ChainedOfflineEnvironments
from flowcean.grpc import GrpcPassiveAutomataLearner
from flowcean.polars import (
    DataFrame,
    Explode,
    Select,
    ToTimeSeries,
    TrainTestSplit,
    Unnest,
)
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError


def main() -> None:
    flowcean.cli.initialize_logging()

    data = ChainedOfflineEnvironments(
        [
            DataFrame.from_uri(uri="file:" + path.as_posix()).with_transform(
                ToTimeSeries("t")
            )
            for path in tqdm(
                list(Path("./data").glob("*.csv")),
                desc="Loading environments",
            )
        ]
    )
    print(data.observe().head())
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(
        data.observe()
    )

    learner = GrpcPassiveAutomataLearner.with_address(address="localhost:8080")
    inputs = ["input"]
    outputs = ["output"]

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
        Explode(["output"]) | Unnest("output") | Select(["value"]),
    )
    print(report)


if __name__ == "__main__":
    main()
