#!/usr/bin/env python

import logging
from datetime import UTC, datetime

import numpy as np
import torch
from system import simulate_one_tank

from flowcean.cli import initialize
from flowcean.core import evaluate_offline, learn_offline
from flowcean.ensemble import EnsembleLearner
from flowcean.polars import DataFrame, SlidingWindow, TrainTestSplit
from flowcean.sklearn import (
    MaxError,
    MeanAbsoluteError,
    MeanSquaredError,
    RegressionTree,
)
from flowcean.torch import LightningLearner, MultilayerPerceptron
from flowcean.utils.random import initialize_random

logger = logging.getLogger(__name__)


def main() -> None:
    initialize()

    initialize_random(seed=42)

    data = DataFrame(simulate_one_tank()) | SlidingWindow(window_size=3)

    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(data)

    inputs = ["h_0", "h_1"]
    outputs = ["h_2"]

    for learner in [
        RegressionTree(max_depth=4),
        LightningLearner(
            module=MultilayerPerceptron(
                learning_rate=1e-3,
                output_size=len(outputs),
                hidden_dimensions=[10, 10],
                activation_function=torch.nn.LeakyReLU,
            ),
            max_epochs=100,
        ),
        EnsembleLearner(
            RegressionTree(max_depth=4),
            RegressionTree(max_depth=4),
        ),
    ]:
        t_start = datetime.now(tz=UTC)
        model = learn_offline(
            train,
            learner,
            inputs,
            outputs,
        )
        delta_t = datetime.now(tz=UTC) - t_start
        print(f"Learning took {np.round(delta_t.microseconds / 1000, 1)} ms")

        report = evaluate_offline(
            model,
            test,
            inputs,
            outputs,
            [
                MeanAbsoluteError(),
                MeanSquaredError(),
                MaxError(),
            ],
        )
        print(report)


if __name__ == "__main__":
    main()
