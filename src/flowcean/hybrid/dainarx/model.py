"""Discrete NARX locations and transitions, independent of native flows."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn.svm import SVC

from ._narx import NarxModel, columns, finite_array, sampling_period


@dataclass(frozen=True)
class DainarxTransition:
    """An output-and-input SVM guard and ordered post-switch predictors.

    Attributes:
        source: Source location ID, indexing ``DainarxModel.locations``.
        destination: Destination location ID, indexing ``DainarxModel.locations``.
        guard: Fitted SVM whose feature vector is selected outputs followed
            by selected inputs, in their feature order. ``True`` enables
            this edge from its source location.
        reset_predictors: Predictors for the first ``order`` post-switch
            outputs, applied in tuple order before destination dynamics.
            These are not instantaneous physical reset maps.
    """

    source: int
    destination: int
    guard: SVC
    reset_predictors: tuple[NarxModel, ...]


@dataclass(frozen=True)
class DainarxModel:
    """Learned sampled dynamics and transitions between location IDs.

    Attributes:
        locations: NARX models indexed by contiguous location IDs
            ``0`` through ``len(locations) - 1``. Models share selected
            features, output-history order, and sampling period.
        transitions: Guarded edges whose source and destination are IDs
            indexing ``locations``.
    """

    locations: tuple[NarxModel, ...]
    transitions: tuple[DainarxTransition, ...]

    def rollout(
        self,
        inputs: pl.DataFrame,
        *,
        output_history: ArrayLike,
        initial_location: int,
    ) -> pl.DataFrame:
        """Recursively predict outputs from exactly ``order`` seed rows.

        At row ``t``, guards inspect ``output[t]`` and ``input[t]``; a switch
        affects row ``t + 1``. Its predictor uses the preceding output history
        and ``input[t + 1]``. Post-switch predictors run in tuple order before
        destination dynamics resume. Multiple enabled guards or an interrupted
        reset sequence raise ``ValueError``; no priority is invented.

        Args:
            inputs: Polars frame with known timestamps and selected inputs
                for all rows, including seeds, at the models' shared sampling
                period. Selected columns must be finite and numeric;
                unrelated columns are ignored. Exactly ``order`` rows are
                allowed for a seed-only return.
            output_history: Finite ``(order, n_outputs)`` seed array, oldest
                row first, in selected output-feature order. ``n_outputs``
                is the number of selected outputs.
            initial_location: Learned location ID at the last seed row,
                which must not be inside an unfinished reset sequence.

        Returns:
            A Polars frame with the timestamp column, selected output columns,
                and ``location_id``, including all seeds. Earlier seed IDs
                are null; the last seed ID is ``initial_location``.
        """
        if (
            isinstance(initial_location, bool)
            or not isinstance(initial_location, int)
            or not 0 <= initial_location < len(self.locations)
        ):
            raise ValueError(
                "initial_location must identify a learned location"
            )
        template = self.locations[initial_location]
        order = template.order
        sampling_period(inputs, template.time_column, template.dt)
        u = columns(inputs, template.input_features)
        history = finite_array(output_history, "Output history")
        if history.shape != (order, len(template.output_features)):
            raise ValueError("Supply exactly order seed output observations")
        if len(u) < order:
            raise ValueError("Input stream must include all seed rows")
        x = np.empty((len(u), len(template.output_features)))
        x[:order] = history
        labels: list[int | None] = [None] * (order - 1) + [initial_location]
        location = initial_location
        pending: tuple[NarxModel, ...] = ()
        phase = 0
        for t in range(order, len(u)):
            state = np.concatenate((x[t - 1], u[t - 1]))[None, :]
            enabled = [
                transition
                for transition in self.transitions
                if transition.source == location
                and transition.guard.predict(state)[0]
            ]
            if len(enabled) > 1:
                raise ValueError(f"Multiple enabled guards at row {t - 1}")
            if enabled:
                if pending:
                    raise ValueError(
                        f"Transition interrupts reset sequence at row {t - 1}"
                    )
                transition = enabled[0]
                location = transition.destination
                pending = transition.reset_predictors
                phase = 0
            predictor = pending[phase] if pending else self.locations[location]
            x[t] = predictor.predict(x[t - order : t], u[t])
            if pending:
                phase += 1
                if phase == len(pending):
                    pending = ()
            labels.append(location)
        return pl.DataFrame(
            {
                template.time_column: inputs[template.time_column],
                **{
                    name: x[:, i]
                    for i, name in enumerate(template.output_features)
                },
                "location_id": pl.Series(labels, dtype=pl.Int64),
            }
        )
