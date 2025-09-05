import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import polars as pl
from numpy.typing import NDArray
from river import tree
from typing_extensions import Self, override

import flowcean.cli
from flowcean.core import evaluate_offline, learn_incremental
from flowcean.ode import (
    OdeEnvironment,
    OdeState,
    OdeSystem,
)
from flowcean.polars import (
    SlidingWindow,
    StreamingOfflineEnvironment,
    TrainTestSplit,
)
from flowcean.polars.environments.dataframe import collect
from flowcean.river import RiverLearner
from flowcean.sklearn import (
    MeanAbsoluteError,
    MeanSquaredError,
)

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
    flowcean.cli.initialize()

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

    inputs = ["h_0", "h_1"]
    outputs = ["h_2"]

    # Collect the data first
    data = collect(data_incremental, 250) | SlidingWindow(window_size=3)

    # Split the data into train and test sets
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    train = StreamingOfflineEnvironment(train, batch_size=1)

    learner = RiverLearner(
        model=tree.HoeffdingTreeRegressor(grace_period=50, max_depth=5),
    )

    t_start = datetime.now(tz=timezone.utc)
    model = learn_incremental(
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
    logger.info("Model learning successful.")


if __name__ == "__main__":
    main()
