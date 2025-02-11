from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Generic, TypeVar, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from typing_extensions import Self, override

from flowcean.core import IncrementalEnvironment


class IntegrationError(Exception):
    """Error while integrating an ODE.

    This exception is raised when an error occurs while integrating an ordinary
    differential equation.
    """

    def __init__(self) -> None:
        """Initialize the exception."""
        super().__init__("failed to integrate ODE")


class OdeState(ABC):
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


X = TypeVar("X", bound=OdeState)


class OdeSystem(ABC, Generic[X]):
    r"""System governed by an ordinary differential equation.

    This class represents a continuous system. The system is defined by a
    differential flow function $f$ that governs the evolution of the state $x$.

    $$
    \begin{aligned}
        \dot{x} &= f(t, x) \\
    \end{aligned}
    $$

    The system can be integrated to obtain the state at a future time.

    Attributes:
        t: Current time.
        state: Current state.
    """

    t: float
    state: X

    def __init__(self, t: float, state: X) -> None:
        """Initialize the system.

        Args:
            t: Initial time.
            state: Initial state.
        """
        self.t = t
        self.state = state

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

    def step(self, dt: float) -> tuple[Sequence[float], Sequence[X]]:
        """Step the mode forward in time.

        Step the mode forward in time by integrating the differential equation
        for a time step of dt.

        Args:
            dt: Time step.

        Returns:
            Tuple of times and states of the integration.
        """
        y0 = self.state.as_numpy()
        solution = solve_ivp(
            self.flow,
            t_span=[self.t, self.t + dt],
            y0=y0,
        )
        if not solution.success:
            raise IntegrationError

        ts = cast(Sequence[float], solution.t[1:])
        states = [self.state.from_numpy(y) for y in solution.y.T[1:]]

        self.t = ts[-1]
        self.state = states[-1]

        return ts, states


class OdeEnvironment(IncrementalEnvironment, Generic[X]):
    """Environment governed by an ordinary differential equation.

    This environment integrates an OdeSystem to generate a sequence of states.
    """

    ts: Sequence[float]
    states: Sequence[X]

    def __init__(
        self,
        system: OdeSystem[X],
        *,
        dt: float = 1.0,
        map_to_dataframe: Callable[
            [Sequence[float], Sequence[X]],
            pl.DataFrame,
        ],
    ) -> None:
        """Initialize the environment.

        Args:
            system: ODE system.
            dt: Time step.
            map_to_dataframe: Function to map states to a DataFrame.
        """
        super().__init__()
        self.system = system
        self.dt = dt
        self.map_to_dataframe = map_to_dataframe
        self.ts = [self.system.t]
        self.states = [self.system.state]

    @override
    def step(self) -> None:
        self.ts, self.states = self.system.step(self.dt)

    @override
    def _observe(self) -> pl.LazyFrame:
        return self.map_to_dataframe(self.ts, self.states).lazy()
