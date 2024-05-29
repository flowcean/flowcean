#!/usr/bin/env python

import random
from collections.abc import Iterator
from itertools import islice
from typing import Self, override

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from flowcean.environments.hybrid_system import (
    DifferentialMode,
    HybridSystem,
    State,
)


class Temperature(State):
    temperature: float

    def __init__(self, temperature: float) -> None:
        self.temperature = temperature

    @override
    def as_numpy(self) -> NDArray[np.float64]:
        return np.array([self.temperature])

    @override
    @classmethod
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        return cls(state[0])


type TargetTemperature = float


class Heating(DifferentialMode[Temperature, TargetTemperature]):
    heating_rate: float = 0.5
    maximum_timeout: float = 1.0

    @override
    def differential_flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _ = t
        return np.array([self.heating_rate])

    @override
    def transition(
        self,
        i: TargetTemperature,
    ) -> DifferentialMode[Temperature, TargetTemperature]:
        if self.state.temperature > i or self.t > self.maximum_timeout:
            return Cooling(t=0.0, state=self.state)
        return self


class Cooling(DifferentialMode[Temperature, TargetTemperature]):
    cooling_rate: float = 0.1
    minimum_timeout: float = 1.0

    @override
    def differential_flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _ = t
        return np.array([-self.cooling_rate])

    @override
    def transition(
        self,
        i: TargetTemperature,
    ) -> DifferentialMode[Temperature, TargetTemperature]:
        if self.state.temperature < i and self.t > self.minimum_timeout:
            return Heating(t=0.0, state=self.state)
        return self


def randomly_changing_values(
    change_probability: float,
    minimum: float,
    maximum: float,
) -> Iterator[float]:
    value = random.uniform(minimum, maximum)
    while True:
        if random.random() < change_probability:
            value = random.uniform(minimum, maximum)
        yield value


def main() -> None:
    target_temperatures = islice(
        randomly_changing_values(
            change_probability=0.002,
            minimum=30.0,
            maximum=60.0,
        ),
        1000,
    )
    environment = HybridSystem(
        initial_mode=Heating(t=0.0, state=Temperature(30)),
        inputs=target_temperatures.__iter__(),
        sampling_time=0.1,
        map_to_dataframe=lambda times, inputs, modes: pl.DataFrame(
            {
                "time": times,
                "target": inputs,
                "temperature": [mode.temperature for mode in modes],
            }
        ),
    ).load()

    data = pl.concat(environment)
    print(data)
    plt.plot(data)
    plt.show()


if __name__ == "__main__":
    main()
