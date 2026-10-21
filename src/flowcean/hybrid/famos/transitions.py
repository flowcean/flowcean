"""Source-conditioned next-location decision trees."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn.tree import DecisionTreeClassifier

from ._data import FloatArray, columns, finite_array


@dataclass(frozen=True)
class FamosTransitionSelector:
    """One inspectable next-location tree per categorical source location.

    Explicit source dispatch differs from the artifact's single joint
    categorical tree; numeric location IDs are never ordered input features.

    Attributes:
        output_features: Current observed physical outputs in vector order.
        input_features: Current exogenous controls in vector order.
        n_locations: Declared number of categorical locations.
        output_scales: Training maximum absolute outputs, zero replaced by one.
        input_scales: Corresponding training maximum absolute inputs.
        trees: Source-indexed fitted sklearn trees. Features are normalized
            current outputs followed by current inputs; targets are next-row
            location IDs, including self transitions.
    """

    output_features: tuple[str, ...]
    input_features: tuple[str, ...]
    n_locations: int
    output_scales: FloatArray
    input_scales: FloatArray
    trees: tuple[DecisionTreeClassifier, ...]

    def predict(
        self, current_location: int, outputs: ArrayLike, inputs: ArrayLike
    ) -> int:
        """Return q[t+1] from q[t], physical-unit y[t], and u[t].

        Output and input vectors have shapes ``(n_outputs,)`` and
        ``(n_inputs,)``, in stored feature order. No future observations
        or target-time inputs are used. Training scales are not refitted
        or clipped for prediction.
        """
        if not 0 <= current_location < self.n_locations:
            raise ValueError("Unknown transition source location")
        y = finite_array(outputs, "Outputs")
        u = finite_array(inputs, "Inputs")
        if y.shape != (len(self.output_features),) or u.shape != (
            len(self.input_features),
        ):
            raise ValueError("Outputs and inputs must match selected features")
        state = np.concatenate((y / self.output_scales, u / self.input_scales))
        return int(self.trees[current_location].predict(state[None, :])[0])


def fit_famos_transitions(
    traces: Sequence[pl.DataFrame],
    *,
    output_features: Sequence[str],
    input_features: Sequence[str],
    n_locations: int,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
) -> FamosTransitionSelector:
    """Fit next-location trees from adjacent explicitly labeled training rows.

    Args:
        traces: Independent eager frames in caller-supplied row order, with
            nullable integer ``location_id`` in ``[0, n_locations)``. A label
            governs the interval beginning at its row. Exclude pairs with
            either label null, without bridging gaps or trace boundaries.
            No timestamps or sampling periods are inferred; unrelated
            columns are ignored. FaMoS discovery labels can be supplied via
            ``discovery.to_labeled_frames(original_traces)``.
        output_features: Ordered current observed physical outputs, not
            next-row targets. At least one output is required.
        input_features: Ordered current exogenous controls; may be empty.
            Feature names must be unique and cannot include ``location_id``.
        n_locations: Positive number of declared source locations. The caller
            is responsible for location meanings and source-time alignment.
        max_depth: Maximum depth of each source's sklearn decision tree.
        min_samples_leaf: Minimum training observations in each tree leaf.
        random_state: Seed controlling sklearn split tie breaking.

    Returns:
        Source-conditioned trees, including valid constant-target trees,
        with training-only maximum-absolute feature scales (zero becomes one).
        A source without any outgoing training pair raises ValueError
        rather than acquiring an invented self loop.
    """
    outputs, inputs = tuple(output_features), tuple(input_features)
    names = (*outputs, *inputs)
    if not outputs or len(names) != len(set(names)) or "location_id" in names:
        raise ValueError("Select distinct features and at least one output")
    if not isinstance(n_locations, int) or n_locations < 1:
        raise ValueError("n_locations must be a positive integer")
    if not traces or not any(trace.height for trace in traces):
        raise ValueError("Supply nonempty training observations")
    features = []
    for trace in traces:
        if (
            "location_id" not in trace.columns
            or not trace.schema["location_id"].is_integer()
        ):
            raise ValueError("location_id must be a nullable integer column")
        if any(
            label is not None and not 0 <= label < n_locations
            for label in trace["location_id"].to_list()
        ):
            raise ValueError("location_id must lie in [0, n_locations)")
        features.append(columns(trace, names))
    scales = np.max(np.abs(np.concatenate(features)), axis=0)
    scales[scales == 0] = 1.0
    rows: list[list[FloatArray]] = [[] for _ in range(n_locations)]
    targets: list[list[int]] = [[] for _ in range(n_locations)]
    for trace, values in zip(traces, features, strict=True):
        normalized = values / scales
        labels = trace["location_id"].to_list()
        for t in range(trace.height - 1):
            source, destination = labels[t : t + 2]
            if source is not None and destination is not None:
                rows[source].append(normalized[t])
                targets[source].append(destination)
    trees = []
    for location, (x, y) in enumerate(zip(rows, targets, strict=True)):
        if not x:
            raise ValueError(
                f"Location {location} has no outgoing training pairs"
            )
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        tree.fit(np.asarray(x), np.asarray(y))
        trees.append(tree)
    return FamosTransitionSelector(
        outputs,
        inputs,
        n_locations,
        scales[: len(outputs)],
        scales[len(outputs) :],
        tuple(trees),
    )
