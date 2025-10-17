from collections.abc import Callable, Sequence

from diffrax import (
    AbstractSolver,
    AbstractStepSizeController,
    ConstantStepSize,
    ODETerm,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from jax import Array
from jaxtyping import PyTree
from typing_extensions import override

from flowcean.core.environment.offline import OfflineEnvironment


class OdeEnvironment(OfflineEnvironment):
    """Environment governed by an ordinary differential equation.

    This environment integrates an OdeSystem to generate a sequence of states.
    """

    flow: ODETerm
    solver: AbstractSolver
    t0: float
    ts: Sequence[float] | Array
    dt0: float
    x0: PyTree
    args: PyTree
    stepsize_controller: AbstractStepSizeController

    def __init__(
        self,
        flow: Callable,
        t0: float,
        x0: PyTree,
        ts: Sequence[float] | Array,
        args: PyTree | None = None,
        solver: AbstractSolver | None = None,
        dt0: float = 1e-3,
        stepsize_controller: AbstractStepSizeController | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            flow: Flow function defining the system dynamics.
            t0: Initial time.
            x0: Initial state.
            ts: Sequence of times at which to save the state.
            args: Additional arguments to pass to the flow
            solver: ODE solver to use.
            dt0: Initial time step for the solver.
            stepsize_controller: Step size controller to use.
        """
        super().__init__()
        self.flow = ODETerm(flow)
        self.t0 = t0
        self.x0 = x0
        self.ts = ts
        self.args = args
        self.solver = solver if solver is not None else Tsit5()
        self.dt0 = dt0
        self.stepsize_controller = (
            stepsize_controller if stepsize_controller else ConstantStepSize()
        )

    @override
    def _observe(self) -> tuple[PyTree, PyTree]:
        solution = diffeqsolve(
            self.flow,
            self.solver,
            self.t0,
            self.ts[-1],
            self.dt0,
            self.x0,
            self.args,
            max_steps=None,
            saveat=SaveAt(ts=self.ts),
            stepsize_controller=self.stepsize_controller,
        )
        return (solution.ts, solution.ys)
