import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import override

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


class Mode(ABC):
    @abstractmethod
    def state(self) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def from_state(self, state: NDArray[np.float64]) -> None:
        pass

    @abstractmethod
    def differential_flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def transition(self) -> "Mode":
        pass


class Heating(Mode):
    temperature: float
    timeout: float
    heating_rate: float = 0.5
    maximum_timeout: float = 1.0
    target_temperature: float = 5.0

    def __init__(self, temperature: float, timeout: float) -> None:
        self.temperature = temperature
        self.timeout = timeout

    @override
    def state(self) -> NDArray[np.float64]:
        return np.array([self.temperature, self.timeout])

    @override
    def from_state(self, state: NDArray[np.float64]) -> None:
        self.temperature = state[0]
        self.timeout = state[1]

    @override
    def differential_flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _ = t
        return np.array([self.heating_rate, 1])

    @override
    def transition(self) -> Mode:
        if (
            self.temperature > self.target_temperature
            or self.timeout > self.maximum_timeout
        ):
            return Cooling(self.temperature, 0.0)
        return self


class Cooling(Mode):
    temperature: float
    timeout: float
    cooling_rate: float = 0.1
    minimum_timeout: float = 1.0
    target_temperature: float = 5.0

    def __init__(self, temperature: float, timeout: float) -> None:
        self.temperature = temperature
        self.timeout = timeout

    @override
    def state(self) -> NDArray[np.float64]:
        return np.array([self.temperature, self.timeout])

    @override
    def from_state(self, state: NDArray[np.float64]) -> None:
        self.temperature = state[0]
        self.timeout = state[1]

    @override
    def differential_flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        _ = t
        return np.array([-self.cooling_rate, 1])

    @override
    def transition(self) -> Mode:
        if (
            self.temperature < self.target_temperature
            and self.timeout > self.minimum_timeout
        ):
            return Heating(self.temperature, 0.0)
        return self


class Boiler:
    mode: Mode
    sampling_time: float
    t: float = 0.0

    def __init__(
        self,
        initial_mode: Mode,
        sampling_time: float,
    ) -> None:
        self.mode = initial_mode
        self.sampling_time = sampling_time

    def step(self, dt: float) -> None:
        time_span = [self.t, self.t + dt]
        solution = solve_ivp(
            self.mode.differential_flow,
            time_span,
            self.mode.state(),
        )
        self.mode.from_state(solution.y[:, -1])
        self.t += dt

        self.mode = self.mode.transition()

    def simulate(self, n: int) -> list[NDArray[np.float64]]:
        states = [self.mode.state()]

        for _ in range(n):
            self.step(self.sampling_time)
            states.append(self.mode.state())

        return states


def random_references(
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
