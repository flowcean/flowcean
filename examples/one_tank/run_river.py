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
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing_extensions import Self, override

from flowcean.cli import initialize_logging
from flowcean.core import evaluate_offline, learn_offline
from flowcean.ode import (
    OdeEnvironment,
    OdeState,
    OdeSystem,
)
from flowcean.polars import SlidingWindow, TrainTestSplit, collect
from flowcean.sklearn import (
    MeanAbsoluteError,
    MeanSquaredError,
    RegressionTree,
)
from flowcean.torch import LightningLearner, MultilayerPerceptron

from river import tree
import logging
from typing import Any


from river.base import Regressor
from typing_extensions import override

from flowcean.core import Model, SupervisedLearner
import pickle
from typing import Any
from typing_extensions import override

from flowcean.core import Model
#from flowcean.river import RiverLearner

class RiverModel(Model):
    def __init__(self, model: Regressor, output_column: str) -> None:
        self.model = model
        self.output_column = output_column

    @override
    def predict(self, inputs: pl.DataFrame) -> pl.DataFrame:
        df = inputs.collect() if isinstance(inputs, pl.LazyFrame) else inputs
        predictions = [
            self.model.predict_one(row) for row in df.to_dicts()
        ]
        return pl.DataFrame({self.output_column: predictions})

    @override
    def save_state(self) -> bytes:
        return pickle.dumps({
            "model": self.model,
            "output_column": self.output_column,
        })

    @classmethod
    @override
    def load_from_state(cls, state: bytes) -> "RiverModel":
        data = pickle.loads(state)
        return cls(model=data["model"], output_column=data["output_column"])


class RiverLearner(SupervisedLearner):
    """Wrapper for River regressors."""

    def __init__(self, model: Regressor) -> None:
        self.model = model

    @override
    def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> Model:
        X = inputs.collect().to_dicts()
        y_col = outputs.columns[0]
        y = outputs.collect()[y_col].to_list()

        logger.info("Training river model")

        for xi, yi in zip(X, y):
            self.model.learn_one(xi, yi)

        return RiverModel(self.model, y_col)


logger = logging.getLogger(__name__)


@dataclass
class TankState(OdeState):
    water_level: float

    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array([self.water_level])

    @classmethod
    @override
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state[0])


class OneTank(OdeSystem[TankState]):
    """One tank system.

    This class represents a one tank system. The system is defined by a
    differential flow function $f$ that governs the evolution of the state $x$.

    This example is based on https://de.mathworks.com/help/slcontrol/ug/watertank-simulink-model.html.
    """

    def __init__(
        self,
        *,
        area: float,
        outflow_rate: float,
        inflow_rate: float,
        initial_t: float = 0.0,
        initial_state: TankState,
    ) -> None:
        """Initialize the one tank system.

        Args:
            area: Area of the tank.
            outflow_rate: Outflow rate.
            inflow_rate: Inflow rate.
            initial_t: Initial time (default: 0).
            initial_state: Initial state.
        """
        super().__init__(
            initial_t,
            initial_state,
        )
        self.area = area
        self.outflow_rate = outflow_rate
        self.inflow_rate = inflow_rate

    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        pump_voltage = np.max([0.0, np.sin(2.0 * np.pi * 1.0 / 10.0 * t)])
        tank = TankState.from_numpy(state)
        d_level = (
            self.inflow_rate * pump_voltage
            - self.outflow_rate * np.sqrt(tank.water_level)
        ) / self.area
        return np.array([d_level])


def main() -> None:
    initialize_logging()

    system = OneTank(
        area=5.0,
        outflow_rate=0.5,
        inflow_rate=2.0,
        initial_state=TankState(water_level=1.0),
    )

    data_incremental = OdeEnvironment(
        system,
        dt=0.1,
        map_to_dataframe=lambda ts, xs: pl.DataFrame(
            {
                "t": ts,
                "h": [x.water_level for x in xs],
            },
        ),
    )

    data = collect(data_incremental, 250).with_transform(
        SlidingWindow(window_size=3),
    )

    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(data)

    inputs = ["h_0", "h_1"]
    outputs = ["h_2"]

    for learner in [
        RegressionTree(max_depth=5),
        RiverLearner(model=tree.HoeffdingTreeRegressor(grace_period=50, max_depth=5)),
        LightningLearner(
            module=MultilayerPerceptron(
                learning_rate=1e-3,
                input_size=len(inputs),
                output_size=len(outputs),
                hidden_dimensions=[10, 10],
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
        print(report)


if __name__ == "__main__":
    main()
