#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
# ]
#
# [tool.uv.sources]
# flowcean = { path = "../../", editable = true }
# ///

import logging

import polars as pl

import flowcean.cli
from flowcean.core import evaluate_offline, learn_incremental
from flowcean.polars import (
    DataFrame,
    StreamingOfflineEnvironment,
    TrainTestSplit,
)
from flowcean.sklearn import MeanAbsoluteError, MeanSquaredError
from flowcean.torch import LinearRegression

logger = logging.getLogger(__name__)

N = 1_000


def main() -> None:
    flowcean.cli.initialize_logging()

    data = DataFrame(
        pl.DataFrame(
            {
                "x": pl.arange(0, N, eager=True).cast(pl.Float32) / N,
                "y": pl.arange(N, 0, -1, eager=True).cast(pl.Float32) / N,
            },
        ),
    )
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    learner = LinearRegression(
        input_size=1,
        output_size=1,
        learning_rate=0.01,
    )
    inputs = ["x"]
    outputs = ["y"]

    model = learn_incremental(
        StreamingOfflineEnvironment(train, batch_size=1),
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
