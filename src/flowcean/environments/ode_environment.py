from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from typing import Self, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from flowcean.core.environment import IncrementalEnvironment


class IntegrationError(Exception):
    """Error while integrating an ODE.

    This exception is raised when an error occurs while integrating an ordinary
    differential equation.
    """

    def __init__(self) -> None:
        super().__init__("failed to integrate ODE")


class State(ABC):
    """State of a differential equation.

    This class represents the state of a differential equation. It provides
    methods to convert the state to and from a numpy array for integration.
    """

    @abstractmethod
    def as_numpy(self) -> NDArray[np.float64]:
        """Convert the state to a numpy array.

        Returns:
            State as a numpy array.
        """

    @classmethod
    @abstractmethod
    def from_numpy(cls, state: NDArray[np.float64]) -> Self:
        """Create a state from a numpy array.

        Args:
            state: State as a numpy array.

        Returns:
            State instance.
        """


class OdeSystem[X: State](ABC):
    r"""System governed by a ordinary differential equation.

    This class represents a continuous system. The system is defined by a
    differential flow function $f$ that governs the evolution of the state $x$.

    $$
    \begin{aligned}
        \dot{x} &= f(t, x) \\
        y &= g(x)
    \end{aligned}
    $$
    """

    @abstractmethod
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Ordinary differential equation.

        Compute the derivative of the state $x$ at time $t$.

        Args:
            t: Time.
            state: State.

        Returns:
            Derivative of the state.
        """

    def step(
        self,
        t: float,
        state: X,
        dt: float,
    ) -> tuple[Sequence[float], Sequence[X]]:
        """Step the mode forward in time.

        Step the mode forward in time by integrating the differential equation
        for a time step of dt.

        Args:
            t: Current time.
            state: Current state.
            dt: Time step.

        Returns:
            Tuple of times and states of the integration.
        """
        y0 = state.as_numpy() if isinstance(state, State) else state
        solution = solve_ivp(
            self.flow,
            t_span=[t, t + dt],
            y0=y0,
        )
        if not solution.success:
            raise IntegrationError
        ts = cast(Sequence[float], solution.t[1:])
        states = [state.from_numpy(y) for y in solution.y.T[1:]]
        return ts, states


class OdeEnvironment[X: State](IncrementalEnvironment):
    def __init__(
        self,
        system: OdeSystem[X],
        initial_state: X,
        *,
        dt: float = 1,
        map_to_dataframe: Callable[
            [Sequence[X]],
            pl.DataFrame,
        ],
    ) -> None:
        self.system = system
        self.initial_state = initial_state
        self.dt = dt
        self.map_to_dataframe = map_to_dataframe

    def load(self) -> Self:
        return self

    def __iter__(self) -> Iterator[pl.DataFrame]:
        t = 0.0
        state = self.initial_state
        while True:
            ts, states = self.system.step(t, state, self.dt)
            yield pl.concat(
                [
                    pl.DataFrame(
                        {
                            "t": ts,
                        },
                    ),
                    self.map_to_dataframe(states),
                ],
                how="horizontal",
            )
            t = ts[-1]
            state = states[-1]
