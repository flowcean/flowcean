from dataclasses import dataclass

import numpy as np

from flowcean.ode import Trace


@dataclass(frozen=True)
class StateTraceComparison:
    absolute_error: np.ndarray
    mae: float
    rmse: float
    max_error: float


def compare_state_traces(
    reference: Trace,
    predicted: Trace,
) -> StateTraceComparison:
    if reference.t.shape != predicted.t.shape or not np.allclose(
        reference.t,
        predicted.t,
        rtol=0.0,
        atol=1e-12,
    ):
        message = "Trace time grids must match."
        raise ValueError(message)
    if reference.x.shape != predicted.x.shape:
        message = "Trace state shapes must match."
        raise ValueError(message)

    absolute_error = np.abs(reference.x - predicted.x)
    return StateTraceComparison(
        absolute_error=absolute_error,
        mae=float(np.mean(absolute_error)),
        rmse=float(np.sqrt(np.mean(np.square(reference.x - predicted.x)))),
        max_error=(
            float(np.max(absolute_error)) if absolute_error.size else 0.0
        ),
    )
