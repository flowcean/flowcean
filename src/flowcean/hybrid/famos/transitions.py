"""Source-conditioned next-location decision trees."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from sklearn.tree import DecisionTreeClassifier

from ._data import FloatArray, columns, finite_array
from .discovery import FamosDiscovery


@dataclass(frozen=True)
class FamosTransitionSelector:
    """One inspectable next-location tree per categorical source location.

    Explicit source dispatch differs from the artifact's single joint
    categorical tree; numeric location IDs are never ordered input features.

    Attributes:
        discovery: Feature order, scales, and location IDs used for training.
        trees: Source-indexed fitted sklearn trees. Features are normalized
            current outputs followed by current inputs; targets are next-row
            location IDs, including self transitions.
    """

    discovery: FamosDiscovery
    trees: tuple[DecisionTreeClassifier, ...]

    def predict(
        self, current_location: int, outputs: ArrayLike, inputs: ArrayLike
    ) -> int:
        """Return q[t+1] from q[t], physical-unit y[t], and u[t].

        Output and input vectors have shapes ``(n_outputs,)`` and
        ``(n_inputs,)``, in discovery feature order. No future observations
        or target-time inputs are used.
        """
        if not 0 <= current_location < len(self.trees):
            raise ValueError("Unknown transition source location")
        y = finite_array(outputs, "Outputs")
        u = finite_array(inputs, "Inputs")
        if y.shape != (len(self.discovery.output_features),) or u.shape != (
            len(self.discovery.input_features),
        ):
            raise ValueError("Outputs and inputs must match selected features")
        state = np.concatenate(
            (y / self.discovery.output_scales, u / self.discovery.input_scales)
        )
        return int(self.trees[current_location].predict(state[None, :])[0])


def fit_famos_transitions(
    traces: Sequence[pl.DataFrame],
    discovery: FamosDiscovery,
    *,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
) -> FamosTransitionSelector:
    """Fit next-location trees from adjacent assigned training rows.

    Args:
        traces: Original discovery frames in the same trace/row order.
        discovery: Partial assignments. Include switches but exclude pairs
            with either label null. Never create cross-trace pairs.
        max_depth: Maximum depth of each source's sklearn decision tree.
        min_samples_leaf: Minimum training observations in each tree leaf.
        random_state: Seed controlling sklearn split tie breaking.

    Returns:
        Source-conditioned trees, including valid constant-target trees.
        A source without any outgoing training pair raises ValueError
        rather than acquiring an invented self loop.
    """
    selected = [
        trace.select(
            discovery.time_column,
            *discovery.output_features,
            *discovery.input_features,
        )
        for trace in traces
    ]
    labeled = discovery.to_labeled_frames(selected)
    rows: list[list[FloatArray]] = [[] for _ in discovery.location_keys]
    targets: list[list[int]] = [[] for _ in discovery.location_keys]
    for trace in labeled:
        y = columns(trace, discovery.output_features) / discovery.output_scales
        u = columns(trace, discovery.input_features) / discovery.input_scales
        labels = trace["location_id"].to_list()
        for t in range(trace.height - 1):
            source, destination = labels[t : t + 2]
            if source is not None and destination is not None:
                rows[source].append(np.concatenate((y[t], u[t])))
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
    return FamosTransitionSelector(discovery, tuple(trees))
