"""Sampled affine ARX fitting from independently discovered locations."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from ._data import FloatArray, columns, design_row, finite_array
from .discovery import FamosDiscovery


@dataclass(frozen=True)
class FamosDynamics:
    """Output-only affine ARX models with exact output-history shifting.

    This intentionally fits only the next output, not the artifact's freely
    fitted augmented lag-state equations. Derivatives are not targets.

    Attributes:
        discovery: Independent discovery artifact providing feature order,
            sampling period, training scales, and location IDs.
        order: Number of output-history rows, not derivative order.
        coefficients: One array per location of shape
            ``(order * n_outputs + n_inputs + 1, n_outputs)``. Rows are
            normalized outputs newest to oldest, source-row inputs, bias.
        sample_counts: Number of within-segment regression targets per location.
        ranks: Least-squares design ranks. Rank deficiency is reported,
            not rejected, and does not establish parameter identifiability.
        rmse: Per-location arrays of per-output training RMSE in physical
            units, not an accuracy acceptance criterion.
    """

    discovery: FamosDiscovery
    order: int
    coefficients: tuple[FloatArray, ...]
    sample_counts: tuple[int, ...]
    ranks: tuple[int, ...]
    rmse: tuple[FloatArray, ...]

    def predict(
        self, location_id: int, output_history: ArrayLike, inputs: ArrayLike
    ) -> FloatArray:
        """Predict y[t+1] using q[t], history through t, and u[t].

        Supply physical-unit output_history of shape ``(order, n_outputs)``,
        oldest first, and inputs of shape ``(n_inputs,)``. Column order is
        defined by discovery. Return a physical-unit ``(n_outputs,)`` vector.
        Stored training scales are used without clipping.
        """
        if not 0 <= location_id < len(self.coefficients):
            raise ValueError("Unknown dynamics location")
        history = finite_array(output_history, "Output history")
        current_input = finite_array(inputs, "Inputs")
        if history.shape != (self.order, len(self.discovery.output_features)):
            raise ValueError("Supply exactly order seed output observations")
        if current_input.shape != (len(self.discovery.input_features),):
            raise ValueError("Inputs must match selected input features")
        row = design_row(
            history / self.discovery.output_scales,
            current_input / self.discovery.input_scales,
        )
        return finite_array(
            (row @ self.coefficients[location_id])
            * self.discovery.output_scales,
            "Predicted outputs",
        )


def fit_famos_dynamics(
    traces: Sequence[pl.DataFrame],
    discovery: FamosDiscovery,
    *,
    order: int = 1,
) -> FamosDynamics:
    """Fit y[t+1] = f_q[t](y[t], ..., y[t-order+1], u[t]).

    Args:
        traces: Original discovery frames in the same trace/row order.
        discovery: Partial location assignments; null rows are not fitted.
        order: Number of output-history rows. History may cross location
            boundaries but never trace boundaries. Source and target must
            lie in the SAME discovered segment, even if adjacent segments
            share a label.

    Returns:
        Least-squares dynamics and diagnostics without a rank or RMSE gate.
        A location with no usable regression targets raises ValueError;
        the independent discovery artifact remains usable.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    discovery._check_traces(traces)
    outputs = [
        columns(trace, discovery.output_features) / discovery.output_scales
        for trace in traces
    ]
    inputs = [
        columns(trace, discovery.input_features) / discovery.input_scales
        for trace in traces
    ]
    rows: list[list[FloatArray]] = [[] for _ in discovery.location_keys]
    targets: list[list[FloatArray]] = [[] for _ in discovery.location_keys]
    for segment in discovery.segments:
        location = segment.location_id
        if location is None:
            continue
        y, u = outputs[segment.trace_index], inputs[segment.trace_index]
        for t in range(max(segment.start, order - 1), segment.stop - 1):
            rows[location].append(design_row(y[t - order + 1 : t + 1], u[t]))
            targets[location].append(y[t + 1])
    coefficients, counts, ranks, errors = [], [], [], []
    for location, (x, y) in enumerate(zip(rows, targets, strict=True)):
        if not x:
            raise ValueError(
                f"Location {location} has no usable regression targets"
            )
        matrix, target = np.asarray(x), np.asarray(y)
        solution, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
        residual = (matrix @ solution - target) * discovery.output_scales
        coefficients.append(finite_array(solution, "Fitted coefficients"))
        counts.append(len(x))
        ranks.append(int(rank))
        errors.append(np.sqrt(np.mean(residual**2, axis=0)))
    return FamosDynamics(
        discovery,
        order,
        tuple(coefficients),
        tuple(counts),
        tuple(ranks),
        tuple(errors),
    )
