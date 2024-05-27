from collections.abc import Callable, Generator
from typing import Self, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from flowcean.core.environment import IncrementalEnvironment


class IntegrationError(Exception):
    def __init__(self) -> None:
        super().__init__("Error while integrating ode")


class ODEEnvironment(IncrementalEnvironment):
    r"""Environment to get samples by integrate a differential equation.

    This environment generates samples by integrating a differential equation
    of the form

    $$
    \begin{aligned}
        \dot{\vec{x}} &= f(t, \vec{x}) \\
        \vec{y} &= g(\vec{x})
    \end{aligned}
    $$

    where $f$ is the differential function, $\vec{x}$ the current state,
    $\dot{\vec{x}}$ it's derivative and $\vec{y}$ is the output calculated by
    the output function $g$.

    Args:
        f: Handle to the ODE function.
        x0: Initial state for the integration.
        g: Output function. Defaults to an invariant function.
        tstep: Time between two samples yielded by `get_next_data`.
        output_names: Names of the output features. Defaults to `y_0`, `y_1`
            ... `y_n`
        implicit_integration: Indicate if an implicit integrator should be
            used. If `True` the `BDF` integrator will be used. Otherwise `RK45`
            is used for integration.

    Example:
        Create an environment that integrates a simple pendulum with length
        $L=1$. Samples are taken every 10 ms with output columns named `phi`
        and `Dphi`.
        ```python
        environment = ODEEnvironment(
            lambda _, x, L=1, g=9.81: np.array(
                [x[1], -g/L * np.sin(x[0])]
            ),
            np.array([np.sin(np.deg2rad(30)), 0]),
            tstep=1e-2,
            output_names=["phi", "Dphi"],
        )
        ```
    """

    def __init__(
        self,
        f: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        x0: npt.NDArray[np.float64],
        *,
        g: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]
        | None = None,
        tstep: float = 1,
        output_names: list[str] | None = None,
        implicit_integration: bool = False,
    ) -> None:
        self.f = f
        self.x0 = x0
        self.tstep = tstep
        self.g = g if g is not None else lambda _, x: x
        self.implicit_integration = implicit_integration

        if output_names is None:
            self.output_names = [
                f"y_{i}" for i in range(len(self.g(0, self.x0)))
            ]
        else:
            self.output_names = output_names

    def load(self) -> Self:
        return self

    def get_next_data(self) -> Generator[pl.DataFrame, None, None]:
        current_state = self.x0.astype(np.float64)
        current_t = 0.0
        while True:
            yield pl.DataFrame(
                [self.g(current_t, current_state).tolist()],
                schema=self.output_names,
            )
            result = cast(
                OdeResult,
                solve_ivp(
                    self.f,
                    (current_t, current_t + self.tstep),
                    current_state,
                    method="RK45" if not self.implicit_integration else "BDF",
                ),
            )
            if not result.success:
                raise IntegrationError
            current_t += self.tstep
            current_state = result.y[:, -1]
