import copy
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Self, override

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


class State(ABC):
    @abstractmethod
    def as_numpy(self) -> NDArray[np.float64]:
        pass

    @classmethod
    @abstractmethod
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        pass


class DifferentialMode[X: State, Input](ABC):
    t: float

    def __init__(self, t: float) -> None:
        self.t = t

    def step(self, state: X, dt: float) -> X:
        solution = solve_ivp(
            self.differential_flow,
            t_span=[self.t, self.t + dt],
            y0=state.as_numpy(),
        )
        self.t = solution.t[-1]
        return state.from_numpy(solution.y[:, -1])

    @abstractmethod
    def differential_flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def transition(self, state: X, i: Input) -> "DifferentialMode[X, Input]":
        pass


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
        state: Temperature,
        i: TargetTemperature,
    ) -> DifferentialMode[Temperature, TargetTemperature]:
        if state.temperature > i or self.t > self.maximum_timeout:
            return Cooling(t=0.0)
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
        state: Temperature,
        i: TargetTemperature,
    ) -> DifferentialMode[Temperature, TargetTemperature]:
        if state.temperature < i and self.t > self.minimum_timeout:
            return Heating(t=0.0)
        return self


def simulate[X: State, Input](
    initial_state: X,
    initial_mode: DifferentialMode[X, Input],
    inputs: Iterator[Input],
    sampling_time: float,
) -> list[X]:
    mode = initial_mode
    state = initial_state
    states = []

    for i in inputs:
        state = mode.step(state, sampling_time)
        mode = mode.transition(state, i)
        states.append(copy.deepcopy(state))

    return states


def randomly_changing_values(
    n: int,
    change_probability: float,
    minimum: float,
    maximum: float,
) -> Iterator[float]:
    value = random.uniform(minimum, maximum)
    for _i in range(n):
        if random.random() < change_probability:
            value = random.uniform(minimum, maximum)
        yield value
