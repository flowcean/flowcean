#!/usr/bin/env python

import logging
from datetime import datetime, timezone

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import polars as pl
import torch
from jaxtyping import PyTree

from flowcean.cli import initialize
from flowcean.core import Lambda, evaluate_offline, learn_offline
from flowcean.ode import OdeEnvironment
from flowcean.polars import SlidingWindow, TrainTestSplit
from flowcean.sklearn import (
    MeanAbsoluteError,
    MeanSquaredError,
    RegressionTree,
)
from flowcean.torch import LightningLearner, MultilayerPerceptron
from flowcean.utils.random import initialize_random

logger = logging.getLogger(__name__)


class OneTank(eqx.Module):
    """One tank system.

    This class represents a one tank system. The system is defined by a
    differential flow function $f$ that governs the evolution of the state $x$.

    This example is based on https://de.mathworks.com/help/slcontrol/ug/watertank-simulink-model.html.
    """

    area: float
    outflow_rate: float
    inflow_rate: float

    def __call__(
        self,
        t: float,
        x: PyTree,
        args: PyTree,
    ) -> PyTree:
        _ = args
        pump_voltage = jnp.maximum(0.0, jnp.sin(2.0 * jnp.pi * 1.0 / 10.0 * t))
        return (
            self.inflow_rate * pump_voltage - self.outflow_rate * jnp.sqrt(x)
        ) / self.area


def main() -> None:
    initialize()

    initialize_random(seed=42)

    system = OneTank(
        area=5.0,
        outflow_rate=0.5,
        inflow_rate=2.0,
    )

    def solution_to_dataframe(data: tuple[PyTree, PyTree]) -> pl.LazyFrame:
        ts, xs = data
        return pl.LazyFrame({"t": np.asarray(ts), "h": np.asarray(xs)})

    data = (
        OdeEnvironment(
            system,
            t0=0.0,
            x0=1.0,
            ts=jnp.arange(0.0, 10.0, 0.1),
        )
        | Lambda(solution_to_dataframe)
        | SlidingWindow(window_size=3)
    )

    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(data)

    inputs = ["h_0", "h_1"]
    outputs = ["h_2"]

    for learner in [
        RegressionTree(max_depth=5),
        LightningLearner(
            module=MultilayerPerceptron(
                learning_rate=1e-3,
                output_size=len(outputs),
                hidden_dimensions=[10, 10],
                activation_function=torch.nn.Tanh,
            ),
            max_epochs=10,
        ),
    ]:
        t_start = datetime.now(tz=timezone.utc)
        model = learn_offline(
            train,
            learner,
            inputs,
            outputs,
        )
        delta_t = datetime.now(tz=timezone.utc) - t_start
        print(f"Learning took {np.round(delta_t.microseconds / 1000, 1)} ms")

        report = evaluate_offline(
            model,
            test,
            inputs,
            outputs,
            [MeanAbsoluteError(), MeanSquaredError()],
        )
        report.pretty_print()


if __name__ == "__main__":
    main()
