"""Composable sampled FaMoS rollout, without native continuous-flow types."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from ._data import columns, finite_array, sampling_period
from .dynamics import FamosDynamics
from .transitions import FamosTransitionSelector


@dataclass(frozen=True)
class FamosModel:
    """Compose sampled dynamics and next-location selection.

    Feature order and location count must match. The caller is responsible
    for matching location meanings and source-time training alignment;
    structural checks cannot certify provenance. Each artifact uses its own
    training scales and exchanges physical-unit values, so scales may differ.

    Attributes:
        dynamics: Affine output-history dynamics.
        selector: Source-conditioned next-location trees with matching
            output/input feature order and location count.
    """

    dynamics: FamosDynamics
    selector: FamosTransitionSelector

    def rollout(
        self,
        inputs: pl.DataFrame,
        *,
        output_history: ArrayLike,
        initial_location: int,
    ) -> pl.DataFrame:
        """Predict recursively using old-location dynamics before advancing.

        Compute BOTH y[t+1] and q[t+1] from q[t], y through t, and u[t].
        A row's location governs the interval beginning at that row. This
        source-input convention intentionally corrects the artifact's use
        of target-time inputs during rollout despite source-time fitting.

        Args:
            inputs: N timestamp/input rows at discovery.dt; unrelated columns
                are ignored. Return N rows, including seeds. The final input
                is unused. N must be at least order; seed-only is supported.
            output_history: Exactly ``(order, n_outputs)`` physical-unit seed
                outputs, oldest first. No later measured outputs are used.
            initial_location: Learned ID at the LAST seed row. Earlier seed
                location IDs are null. Initialization is the caller's choice,
                not inferred by clustering a held-out trajectory.

        Returns:
            Timestamp, output columns and nullable location_id. The last
            predicted location is included even without a following output.
        """
        discovery = self.dynamics.discovery
        if (
            self.selector.output_features != discovery.output_features
            or self.selector.input_features != discovery.input_features
            or self.selector.n_locations != len(discovery.location_keys)
        ):
            raise ValueError(
                "Dynamics and selector must match feature order and location count"
            )
        if not 0 <= initial_location < len(discovery.location_keys):
            raise ValueError(
                "initial_location must identify a learned location"
            )
        sampling_period(inputs, discovery.time_column, discovery.dt)
        u = columns(inputs, discovery.input_features)
        order = self.dynamics.order
        history = finite_array(output_history, "Output history")
        if history.shape != (order, len(discovery.output_features)):
            raise ValueError("Supply exactly order seed output observations")
        if len(u) < order:
            raise ValueError("Input stream must include all seed rows")
        y = np.empty((len(u), len(discovery.output_features)))
        y[:order] = history
        labels: list[int | None] = [None] * (order - 1) + [initial_location]
        location = initial_location
        for t in range(order - 1, len(u) - 1):
            y[t + 1] = self.dynamics.predict(
                location, y[t - order + 1 : t + 1], u[t]
            )
            location = self.selector.predict(location, y[t], u[t])
            labels.append(location)
        return pl.DataFrame(
            {
                discovery.time_column: inputs[discovery.time_column],
                **{
                    name: y[:, i]
                    for i, name in enumerate(discovery.output_features)
                },
                "location_id": pl.Series(labels, dtype=pl.Int64),
            }
        )
