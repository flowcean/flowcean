"""Ordered simulation evidence without retries or partial scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]

MATRIX_DIMENSIONS = 2


@dataclass(frozen=True)
class SimulationBatch:
    """Aligned evidence for all original rows, including unavailable traces.

    Invalid rows contain only NaN placeholders, not physical trajectories.
    All generated scenario rows have been attempted; only valid rows count as
    successful simulations. Exact duplicates are diagnostic, not removed.
    """

    scenarios: FloatArray
    trajectories: FloatArray
    valid: NDArray[np.bool_]
    errors: tuple[str, ...]

    def __post_init__(self) -> None:
        # Bytes-backed copies cannot be made writable and do not alias callers.
        for name in ("scenarios", "trajectories", "valid"):
            values = getattr(self, name)
            immutable = np.frombuffer(
                values.tobytes(),
                dtype=values.dtype,
            ).reshape(values.shape)
            object.__setattr__(self, name, immutable)

    def __reduce__(
        self,
    ) -> tuple[
        type[SimulationBatch],
        tuple[FloatArray, FloatArray, NDArray[np.bool_], tuple[str, ...]],
    ]:
        # Re-run construction so process transport preserves read-only arrays.
        return type(self), (
            self.scenarios,
            self.trajectories,
            self.valid,
            self.errors,
        )

    @property
    def generated_count(self) -> int:
        """Number of supplied scenario rows, including exact duplicates."""
        return self.scenarios.shape[0]

    @property
    def attempted_count(self) -> int:
        return self.generated_count

    @property
    def successful_count(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def unique_scenario_count(self) -> int:
        return np.unique(self.scenarios, axis=0).shape[0]

    @property
    def unique_valid_trajectory_count(self) -> int:
        return np.unique(self.trajectories[self.valid], axis=0).shape[0]

    def require_complete(self, label: str) -> FloatArray:
        """Return all targets or raise with evidence, never just survivors."""
        if not np.all(self.valid):
            raise IncompleteSimulationError(label, self)
        return self.trajectories


class IncompleteSimulationError(RuntimeError):
    """A strict consumer rejected a batch with unavailable trajectories."""

    def __init__(self, label: str, batch: SimulationBatch) -> None:
        self.label = label
        self.batch = batch
        super().__init__(
            f"{label}: {batch.successful_count}/{batch.attempted_count} "
            "simulations succeeded; complete batch required",
        )

    def __reduce__(
        self,
    ) -> tuple[type[IncompleteSimulationError], tuple[str, SimulationBatch]]:
        # ProcessPoolExecutor transports worker exceptions via pickle.
        return type(self), (self.label, self.batch)


def execute_simulations(
    scenarios: ArrayLike,
    simulate_row: Callable[[FloatArray], ArrayLike],
    trajectory_width: int,
    progress: Callable[[int], object] | None = None,
) -> SimulationBatch:
    """Attempt every row once, retaining recoverable failures in place.

    Only RuntimeError, ValueError, and ArithmeticError from the simulator or
    output conversion/validation are recoverable. Progress callbacks run
    outside that boundary, once per attempted row, with an increment of one.
    """
    values = validate_simulation_inputs(
        scenarios,
        simulate_row,
        trajectory_width,
        progress,
    )

    trajectories = np.full(
        (values.shape[0], trajectory_width),
        np.nan,
        dtype=np.float64,
    )
    valid = np.zeros(values.shape[0], dtype=np.bool_)
    errors: list[str] = []
    for index, scenario in enumerate(values):
        try:
            output = _validated_trajectory(
                simulate_row(scenario.copy()),
                trajectory_width,
            )
        except (RuntimeError, ValueError, ArithmeticError) as error:
            errors.append(f"{type(error).__name__}: {error}")
        else:
            trajectories[index] = output
            valid[index] = True
            errors.append("")
        if progress is not None:
            progress(1)

    return SimulationBatch(values, trajectories, valid, tuple(errors))


def validate_simulation_inputs(
    scenarios: ArrayLike,
    simulate_row: Callable[[FloatArray], ArrayLike],
    trajectory_width: int,
    progress: Callable[[int], object] | None = None,
) -> FloatArray:
    """Validate the entire batch before attempting any row; return a copy."""
    values = np.array(scenarios, dtype=np.float64, copy=True)
    if (
        values.ndim != MATRIX_DIMENSIONS
        or 0 in values.shape
        or not np.all(np.isfinite(values))
    ):
        message = "scenarios must be a finite, nonempty 2D matrix"
        raise ValueError(message)
    if (
        isinstance(trajectory_width, (bool, np.bool_))
        or not isinstance(trajectory_width, (int, np.integer))
        or trajectory_width < 1
    ):
        message = "trajectory width must be a positive integer"
        raise ValueError(message)
    if not callable(simulate_row) or (
        progress is not None and not callable(progress)
    ):
        message = "simulator and optional progress must be callable"
        raise TypeError(message)

    return values


def _validated_trajectory(output: ArrayLike, width: int) -> FloatArray:
    values = np.asarray(output, dtype=np.float64)
    if values.shape != (width,):
        message = f"trajectory shape {values.shape}; expected ({width},)"
        raise ValueError(message)
    if not np.all(np.isfinite(values)):
        message = "trajectory must contain only finite values"
        raise ValueError(message)
    return values
