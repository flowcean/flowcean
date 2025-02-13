#!/usr/bin/env python
# /// script
# dependencies = [
#     "flowcean",
# ]
# ///

import logging
import time

import flowcean.cli
from flowcean.core import evaluate_offline, learn_offline
from flowcean.polars import (
    DataFrame,
    Flatten,
    Resample,
    Select,
    TrainTestSplit,
)
from flowcean.sklearn import (
    MeanAbsoluteError,
    MeanSquaredError,
    RegressionTree,
)

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize_logging()
    time_start = time.time()

    data = (
        DataFrame.from_parquet("./alp_sim_data.parquet")
        | Select(
            [
                "p_accumulator",
                "containerWeight",
                "p_initial",
                "activeValveCount",
                "T",
            ],
        )
        | Resample(1.0)
        | Flatten()
    )
    time_end = time.time()
    logger.info("took %.5f s to load data", time_end - time_start)

    train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(data)
    time_end = time.time()

    learner = RegressionTree()
    inputs = [
        "^p_accumulator_.*$",
        "activeValveCount",
        "p_initial",
        "T",
    ]
    outputs = ["containerWeight"]

    model = learn_offline(
        train_env,
        learner,
        inputs,
        outputs,
    )

    report = evaluate_offline(
        model,
        test_env,
        inputs,
        outputs,
        [MeanAbsoluteError(), MeanSquaredError()],
    )
    print(report)


if __name__ == "__main__":
    main()
