from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeRegressor

FloatArray = NDArray[np.float64]
TREE_LEAF = -1
MATRIX_DIMENSIONS = 2


@dataclass(frozen=True)
class PredictionMetrics:
    fitting_rmse: float
    assessment_rmse: float | None
    historical_mean_rmse: float | None
    prediction_ratio: float | None


@dataclass(frozen=True)
class LeafRealizationMetrics:
    leaf_id: int
    assessment_members: int | None
    midpoint_realization_distance: float | None
    held_leaf_distance: float | None
    realization_ratio: float | None


@dataclass(frozen=True)
class RealizationMetrics:
    midpoint_realization_distance: float | None
    held_leaf_distance: float | None
    realization_ratio: float | None
    assessment_unrepresented_volume: float | None
    leaves: tuple[LeafRealizationMetrics, ...]


@dataclass(frozen=True)
class CoverageMetrics:
    reference_assignments: NDArray[np.int64]
    reference_distances: FloatArray
    mean_distance: float
    q95_distance: float
    q99_distance: float
    max_distance: float
    used_members: int
    mean_input_spacing: float
    scaled_input_spacing: float


def coordinate_rms(first: FloatArray, second: FloatArray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        message = "coordinate RMS needs equal nonempty shapes"
        raise ValueError(message)
    value = float(np.sqrt(np.mean(np.square(left - right))))
    if not np.isfinite(value):
        message = "coordinate RMS is not finite"
        raise ValueError(message)
    return value


def pairwise_coordinate_rms(values: FloatArray) -> FloatArray:
    matrix = _matrix(values, "distance values")
    distances = _distance_matrix(matrix, matrix)
    distances = (distances + distances.T) / 2.0
    np.fill_diagonal(distances, 0.0)
    return distances


def cross_coordinate_rms(first: FloatArray, second: FloatArray) -> FloatArray:
    left = _matrix(first, "first distance values")
    right = _matrix(second, "second distance values")
    if left.shape[1] != right.shape[1]:
        message = "cross-distance matrices need equal column counts"
        raise ValueError(message)
    return _distance_matrix(left, right)


def _distance_matrix(left: FloatArray, right: FloatArray) -> FloatArray:
    """Fast Gram distances for the example's standardized/bounded inputs.

    Cancellation can affect near-zero distances and large common offsets;
    this is not a general-purpose numerically robust distance kernel.
    """
    squared = (
        np.sum(np.square(left), axis=1)[:, None]
        + np.sum(np.square(right), axis=1)[None, :]
        - 2 * left @ right.T
    )
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared / left.shape[1]).astype(np.float64)


def prediction_metrics(
    tree: DecisionTreeRegressor,
    fitting_scenarios: FloatArray,
    fitting_targets: FloatArray,
    assessment_scenarios: FloatArray,
    assessment_targets: FloatArray | None,
) -> PredictionMetrics:
    fitting = _matrix(fitting_scenarios, "fitting scenarios")
    fitting_truth = _matrix(fitting_targets, "fitting targets")
    if assessment_targets is None:
        predicted = np.asarray(
            tree.predict(fitting),
            dtype=np.float64,
        ).reshape(
            fitting_truth.shape,
        )
        return PredictionMetrics(
            coordinate_rms(fitting_truth, predicted),
            None,
            None,
            None,
        )
    assessment = _matrix(assessment_scenarios, "assessment scenarios")
    assessment_truth = _matrix(assessment_targets, "assessment targets")
    if fitting.shape[0] != fitting_truth.shape[0]:
        message = "fitting scenarios and targets need equal rows"
        raise ValueError(message)
    if assessment.shape[0] != assessment_truth.shape[0]:
        message = "assessment scenarios and targets need equal rows"
        raise ValueError(message)
    fitting_prediction = np.asarray(
        tree.predict(fitting),
        dtype=np.float64,
    ).reshape(
        fitting_truth.shape,
    )
    assessment_prediction = np.asarray(
        tree.predict(assessment),
        dtype=np.float64,
    ).reshape(assessment_truth.shape)
    historical = np.broadcast_to(
        fitting_truth.mean(axis=0),
        assessment_truth.shape,
    )
    fitting_error = coordinate_rms(fitting_truth, fitting_prediction)
    assessment_error = coordinate_rms(assessment_truth, assessment_prediction)
    historical_error = coordinate_rms(assessment_truth, historical)
    return PredictionMetrics(
        fitting_rmse=fitting_error,
        assessment_rmse=assessment_error,
        historical_mean_rmse=historical_error,
        prediction_ratio=(
            assessment_error / historical_error if historical_error else None
        ),
    )


def midpoint_realization_metrics(
    tree: DecisionTreeRegressor,
    assessment_scenarios: FloatArray,
    assessment_targets: FloatArray,
    midpoint_scenarios: FloatArray,
    midpoint_targets: FloatArray,
    leaf_volumes: FloatArray,
) -> RealizationMetrics:
    assessment = _matrix(assessment_scenarios, "assessment scenarios")
    truth = _matrix(assessment_targets, "assessment targets")
    midpoints = _matrix(midpoint_scenarios, "leaf midpoints")
    midpoint_truth = _matrix(midpoint_targets, "midpoint targets")
    if (
        assessment.shape[0] != truth.shape[0]
        or midpoints.shape[0] != midpoint_truth.shape[0]
    ):
        message = "realization scenarios and targets need aligned rows"
        raise ValueError(message)
    leaf_ids = tuple(
        index
        for index, child in enumerate(tree.tree_.children_left)
        if int(child) == TREE_LEAF
    )
    routed_midpoints = tuple(int(value) for value in tree.apply(midpoints))
    if routed_midpoints != leaf_ids:
        message = "midpoints must route one-for-one to sorted tree leaves"
        raise ValueError(message)
    assignments = np.asarray(tree.apply(assessment), dtype=np.int64)
    prototypes = np.asarray(tree.predict(midpoints), dtype=np.float64).reshape(
        midpoint_truth.shape,
    )
    volumes = np.asarray(leaf_volumes, dtype=np.float64)
    if (
        volumes.shape != (len(leaf_ids),)
        or not np.all(np.isfinite(volumes))
        or np.any(volumes <= 0)
        or not np.isclose(volumes.sum(), 1.0, atol=1e-12)
    ):
        message = "leaf volumes must be positive and sum to one"
        raise ValueError(message)
    midpoint_errors = np.sqrt(
        np.mean(np.square(midpoint_truth - prototypes), axis=1),
    )
    held_errors = np.zeros(len(leaf_ids), dtype=np.float64)
    represented = np.zeros(len(leaf_ids), dtype=bool)
    leaf_metrics: list[LeafRealizationMetrics] = []
    for index, (leaf_id, prototype) in enumerate(
        zip(leaf_ids, prototypes, strict=True),
    ):
        members = truth[assignments == leaf_id]
        held_distance: float | None = None
        ratio: float | None = None
        if len(members) > 0:
            represented[index] = True
            held_distance = float(
                np.sqrt(
                    np.mean(np.square(members - prototype), axis=1),
                ).mean(),
            )
            held_errors[index] = held_distance
            if held_distance > 0:
                ratio = float(midpoint_errors[index] / held_distance)
        leaf_metrics.append(
            LeafRealizationMetrics(
                leaf_id=leaf_id,
                assessment_members=len(members),
                midpoint_realization_distance=float(midpoint_errors[index]),
                held_leaf_distance=held_distance,
                realization_ratio=ratio,
            ),
        )
    occupancies = np.asarray(
        [leaf.assessment_members for leaf in leaf_metrics],
        dtype=np.int64,
    )
    midpoint_error = float(np.average(midpoint_errors, weights=occupancies))
    held_error = float(np.average(held_errors, weights=occupancies))
    return RealizationMetrics(
        midpoint_realization_distance=midpoint_error,
        held_leaf_distance=held_error,
        realization_ratio=midpoint_error / held_error if held_error else None,
        assessment_unrepresented_volume=float(volumes[~represented].sum()),
        leaves=tuple(leaf_metrics),
    )


def available_realization_metrics(
    tree: DecisionTreeRegressor,
    assessment_scenarios: FloatArray,
    assessment_targets: FloatArray | None,
    midpoint_scenarios: FloatArray,
    midpoint_targets: tuple[FloatArray | None, ...],
    leaf_volumes: FloatArray,
) -> RealizationMetrics:
    """Independent leaf evidence; aggregate midpoints require both batches."""
    complete_midpoints = all(value is not None for value in midpoint_targets)
    if assessment_targets is not None and complete_midpoints:
        return midpoint_realization_metrics(
            tree,
            assessment_scenarios,
            assessment_targets,
            midpoint_scenarios,
            np.stack(
                [target for target in midpoint_targets if target is not None],
            ),
            leaf_volumes,
        )
    leaf_ids = tuple(int(value) for value in tree.apply(midpoint_scenarios))
    expected = tuple(np.flatnonzero(tree.tree_.children_left == TREE_LEAF))
    if leaf_ids != expected:
        message = "midpoints must route one-for-one to sorted tree leaves"
        raise ValueError(message)
    prototypes = np.asarray(
        tree.predict(midpoint_scenarios),
        dtype=np.float64,
    ).reshape(
        len(leaf_ids),
        -1,
    )
    assignments = (
        tree.apply(assessment_scenarios)
        if assessment_targets is not None
        else None
    )
    leaves: list[LeafRealizationMetrics] = []
    for leaf_id, prototype, target in zip(
        leaf_ids,
        prototypes,
        midpoint_targets,
        strict=True,
    ):
        midpoint_error = (
            coordinate_rms(target, prototype) if target is not None else None
        )
        count: int | None = None
        held: float | None = None
        if assessment_targets is not None:
            members = assessment_targets[assignments == leaf_id]
            count = len(members)
            if count:
                held = float(
                    np.sqrt(
                        np.mean(np.square(members - prototype), axis=1),
                    ).mean(),
                )
        leaves.append(
            LeafRealizationMetrics(
                leaf_id,
                count,
                midpoint_error,
                held,
                midpoint_error / held
                if midpoint_error is not None and held
                else None,
            ),
        )
    held_error: float | None = None
    unrepresented: float | None = None
    if assessment_targets is not None:
        occupancies = np.asarray(
            [leaf.assessment_members for leaf in leaves],
            dtype=np.int64,
        )
        held_errors = np.asarray(
            [leaf.held_leaf_distance or 0.0 for leaf in leaves],
        )
        held_error = float(np.average(held_errors, weights=occupancies))
        unrepresented = float(leaf_volumes[occupancies == 0].sum())
    return RealizationMetrics(
        None,
        held_error,
        None,
        unrepresented,
        tuple(leaves),
    )


def reference_coverage_metrics(
    suite_targets: FloatArray,
    reference_targets: FloatArray,
    suite_scenarios: FloatArray,
    bounds: FloatArray,
) -> CoverageMetrics:
    suite = _matrix(suite_targets, "suite trajectories")
    reference = _matrix(reference_targets, "reference trajectories")
    scenarios = _matrix(suite_scenarios, "suite scenarios")
    domain = np.asarray(bounds, dtype=np.float64)
    if (
        suite.shape[1] != reference.shape[1]
        or suite.shape[0] != scenarios.shape[0]
    ):
        message = "coverage suite scenarios and trajectories are not aligned"
        raise ValueError(message)
    if (
        domain.shape != (scenarios.shape[1], 2)
        or np.any(domain[:, 1] <= domain[:, 0])
        or not np.all(np.isfinite(domain))
    ):
        message = "coverage bounds are invalid"
        raise ValueError(message)
    distances = cross_coordinate_rms(reference, suite)
    assignments = np.argmin(distances, axis=1).astype(np.int64, copy=False)
    nearest = distances[np.arange(reference.shape[0]), assignments]
    normalized = (scenarios - domain[:, 0]) / (domain[:, 1] - domain[:, 0])
    if np.any(normalized < 0) or np.any(normalized > 1):
        message = "suite scenarios leave the declared bounds"
        raise ValueError(message)
    if len(suite) == 1:
        spacing = 0.0
    else:
        spacing_matrix = pairwise_coordinate_rms(normalized)
        np.fill_diagonal(spacing_matrix, np.inf)
        spacing = float(np.min(spacing_matrix, axis=1).mean())
    budget = len(suite)
    return CoverageMetrics(
        reference_assignments=assignments,
        reference_distances=nearest,
        mean_distance=float(nearest.mean()),
        q95_distance=float(np.quantile(nearest, 0.95, method="linear")),
        q99_distance=float(np.quantile(nearest, 0.99, method="linear")),
        max_distance=float(nearest.max()),
        used_members=int(np.unique(assignments).size),
        mean_input_spacing=spacing,
        scaled_input_spacing=spacing * budget ** (1 / scenarios.shape[1]),
    )


def _matrix(values: FloatArray, label: str) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if (
        matrix.ndim != MATRIX_DIMENSIONS
        or min(matrix.shape) < 1
        or not np.all(np.isfinite(matrix))
    ):
        message = f"{label} must be a finite nonempty matrix"
        raise ValueError(message)
    return matrix
