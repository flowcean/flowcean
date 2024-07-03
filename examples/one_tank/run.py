import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Self, override

import numpy as np
import polars as pl
from numpy.typing import NDArray

import flowcean.cli
from flowcean.environments.ode_environment import (
    OdeEnvironment,
    OdeSystem,
    State,
)
from flowcean.environments.train_test_split import TrainTestSplit
from flowcean.learners.lightning import LightningLearner, MultilayerPerceptron
from flowcean.learners.regression_tree import RegressionTree
from flowcean.metrics import MeanAbsoluteError, MeanSquaredError
from flowcean.strategies.offline import evaluate_offline, learn_offline
from flowcean.transforms.sliding_window import SlidingWindow

logger = logging.getLogger(__name__)


@dataclass
class TankState(State):
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
        initial_t: float = 0,
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
        pump_voltage = np.max([0, np.sin(2 * np.pi * 1 / 10 * t)])
        tank = TankState.from_numpy(state)
        d_level = (
            self.inflow_rate * pump_voltage
            - self.outflow_rate * np.sqrt(tank.water_level)
        ) / self.area
        return np.array([d_level])


def main() -> None:
    flowcean.cli.initialize_logging()

    system = OneTank(
        area=5,
        outflow_rate=0.5,
        inflow_rate=2,
        initial_state=TankState(water_level=1),
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
    ).load()

    data = data_incremental.collect(250).with_transform(
        SlidingWindow(window_size=3),
    )
    data = data.load()

    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(data)

    inputs = ["h_0", "h_1"]
    outputs = ["h_2"]

    for learner in [
        RegressionTree(max_depth=5),
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
            [MeanAbsoluteError(), MeanSquaredError()],
        )
        print(report)


if __name__ == "__main__":
    main()
