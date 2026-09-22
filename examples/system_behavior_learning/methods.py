from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc
from sklearn.tree import DecisionTreeRegressor

FloatArray = NDArray[np.float64]
TREE_LEAF = -1
MATRIX_DIMENSIONS = 2
MIN_BEHAVIOR_TREE_CAPACITY = 2


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class LeafBox:
    leaf_id: int
    depth: int
    intervals: tuple[Interval, ...]
    midpoint: FloatArray
    relative_volume: float


def sklearn_random_state(seed: np.random.SeedSequence) -> int:
    return int(seed.generate_state(1, dtype=np.uint32)[0])


def fit_behavior_tree(
    scenarios: FloatArray,
    trajectories: FloatArray,
    capacity: int,
    seed: np.random.SeedSequence,
) -> DecisionTreeRegressor:
    if capacity < MIN_BEHAVIOR_TREE_CAPACITY:
        message = "behavior-tree capacity must be at least two"
        raise ValueError(message)
    return _fit_behavior_tree(scenarios, trajectories, capacity, seed)


def fit_unbounded_behavior_tree(
    scenarios: FloatArray,
    trajectories: FloatArray,
    seed: np.random.SeedSequence,
) -> DecisionTreeRegressor:
    return _fit_behavior_tree(scenarios, trajectories, None, seed)


def _fit_behavior_tree(
    scenarios: FloatArray,
    trajectories: FloatArray,
    capacity: int | None,
    seed: np.random.SeedSequence,
) -> DecisionTreeRegressor:
    features = _finite_matrix(scenarios, "fitting scenarios")
    targets = _finite_matrix(trajectories, "fitting trajectories")
    if features.shape[0] != targets.shape[0]:
        message = "fitting scenarios and trajectories need equal rows"
        raise ValueError(message)
    return DecisionTreeRegressor(
        criterion="squared_error",
        splitter="best",
        max_leaf_nodes=capacity,
        min_samples_leaf=1,
        random_state=sklearn_random_state(seed),
    ).fit(features, targets)


def normalize_inputs(scenarios: FloatArray, bounds: FloatArray) -> FloatArray:
    features = _finite_matrix(scenarios, "scenarios")
    domain = _bounds(bounds, features.shape[1])
    if np.any(features < domain[:, 0]) or np.any(features > domain[:, 1]):
        message = "scenarios leave the declared input bounds"
        raise ValueError(message)
    return np.asarray(
        (features - domain[:, 0]) / (domain[:, 1] - domain[:, 0]),
        dtype=np.float64,
    )


def fit_input_only_tree(
    scenarios: FloatArray,
    bounds: FloatArray,
    capacity: int,
    seed: np.random.SeedSequence,
) -> DecisionTreeRegressor:
    """Fit a partition using normalized inputs as targets, never behavior."""
    features = _finite_matrix(scenarios, "fitting scenarios")
    normalized = normalize_inputs(features, bounds)
    if capacity < 1:
        message = "input-only tree capacity must be positive"
        raise ValueError(message)
    random_state = sklearn_random_state(seed)
    if capacity == 1:
        tree = DecisionTreeRegressor(
            criterion="squared_error",
            splitter="best",
            min_samples_split=features.shape[0] + 1,
            min_samples_leaf=1,
            random_state=random_state,
        ).fit(features, normalized)
    else:
        tree = DecisionTreeRegressor(
            criterion="squared_error",
            splitter="best",
            max_leaf_nodes=capacity,
            min_samples_leaf=1,
            random_state=random_state,
        ).fit(features, normalized)
    return tree


def effective_input_boundary(threshold: float) -> tuple[float, bool]:
    """Return sklearn's float64 preimage boundary and left equality owner."""
    if not np.isfinite(threshold):
        message = "tree threshold must be finite"
        raise ValueError(message)
    rounded = np.float32(threshold)
    lower = (
        np.nextafter(rounded, np.float32(-np.inf))
        if float(rounded) > threshold
        else rounded
    )
    upper = np.nextafter(lower, np.float32(np.inf))
    if not np.isfinite(lower) or not np.isfinite(upper):
        message = f"threshold has no adjacent finite float32 pair: {threshold}"
        raise ValueError(message)
    boundary = float(lower) + (float(upper) - float(lower)) / 2.0
    return boundary, bool(float(np.float32(boundary)) <= threshold)


def _split_interval(
    current: Interval,
    boundary: float,
    *,
    left_inclusive: bool,
) -> tuple[Interval, Interval]:
    if boundary < current.upper:
        left = Interval(
            lower=current.lower,
            upper=boundary,
            lower_inclusive=current.lower_inclusive,
            upper_inclusive=left_inclusive,
        )
    elif boundary == current.upper:
        left = Interval(
            lower=current.lower,
            upper=current.upper,
            lower_inclusive=current.lower_inclusive,
            upper_inclusive=current.upper_inclusive and left_inclusive,
        )
    else:
        left = current

    if boundary > current.lower:
        right = Interval(
            lower=boundary,
            upper=current.upper,
            lower_inclusive=not left_inclusive,
            upper_inclusive=current.upper_inclusive,
        )
    elif boundary == current.lower:
        right = Interval(
            lower=current.lower,
            upper=current.upper,
            lower_inclusive=current.lower_inclusive and not left_inclusive,
            upper_inclusive=current.upper_inclusive,
        )
    else:
        right = current
    return left, right


def _collect_leaf_intervals(
    tree: DecisionTreeRegressor,
    initial: tuple[Interval, ...],
) -> list[tuple[int, int, tuple[Interval, ...]]]:
    leaves: list[tuple[int, int, tuple[Interval, ...]]] = []

    def visit(
        node: int,
        depth: int,
        intervals: tuple[Interval, ...],
    ) -> None:
        left_child = int(tree.tree_.children_left[node])
        right_child = int(tree.tree_.children_right[node])
        if left_child == TREE_LEAF:
            if right_child != TREE_LEAF:
                message = f"tree node {node} has only one child"
                raise ValueError(message)
            leaves.append((node, depth, intervals))
            return

        feature = int(tree.tree_.feature[node])
        boundary, left_inclusive = effective_input_boundary(
            float(tree.tree_.threshold[node]),
        )
        left, right = _split_interval(
            intervals[feature],
            boundary,
            left_inclusive=left_inclusive,
        )
        left_intervals = list(intervals)
        left_intervals[feature] = left
        right_intervals = list(intervals)
        right_intervals[feature] = right
        visit(left_child, depth + 1, tuple(left_intervals))
        visit(right_child, depth + 1, tuple(right_intervals))

    visit(0, 0, initial)
    return leaves


def leaf_boxes(
    tree: DecisionTreeRegressor,
    bounds: FloatArray,
) -> tuple[LeafBox, ...]:
    domain = _bounds(bounds, int(tree.n_features_in_))
    initial = tuple(
        Interval(lower=float(lower), upper=float(upper))
        for lower, upper in domain
    )
    leaves = _collect_leaf_intervals(tree, initial)
    widths = domain[:, 1] - domain[:, 0]
    result: list[LeafBox] = []
    for leaf_id, depth, intervals in sorted(leaves):
        box_widths = np.asarray([item.width for item in intervals])
        if np.any(box_widths <= 0) or not np.all(np.isfinite(box_widths)):
            message = f"leaf {leaf_id} has an empty box inside the domain"
            raise ValueError(message)
        midpoint = np.asarray(
            [item.lower + item.width / 2 for item in intervals],
            dtype=np.float64,
        )
        routed = int(tree.apply(midpoint.reshape(1, -1))[0])
        if routed != leaf_id:
            message = (
                f"verified midpoint for leaf {leaf_id} routes to leaf {routed}"
            )
            raise ValueError(message)
        result.append(
            LeafBox(
                leaf_id=leaf_id,
                depth=depth,
                intervals=intervals,
                midpoint=midpoint,
                relative_volume=float(np.prod(box_widths / widths)),
            ),
        )
    if not np.isclose(
        sum(item.relative_volume for item in result),
        1.0,
        atol=1e-12,
    ):
        message = "relative leaf-box volumes do not sum to one"
        raise ValueError(message)
    return tuple(result)


def tree_midpoints(
    tree: DecisionTreeRegressor,
    bounds: FloatArray,
) -> FloatArray:
    boxes = leaf_boxes(tree, bounds)
    return np.stack([item.midpoint for item in boxes]).astype(np.float64)


def pam_medoids(
    distances: FloatArray,
    count: int,
    tolerance: float = 1e-12,
) -> tuple[int, ...]:
    matrix = np.asarray(distances, dtype=np.float64)
    if (
        matrix.ndim != MATRIX_DIMENSIONS
        or matrix.shape[0] != matrix.shape[1]
        or not 1 <= count <= matrix.shape[0]
        or tolerance < 0
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0)
        or not np.allclose(matrix, matrix.T, rtol=0, atol=tolerance)
        or not np.allclose(np.diag(matrix), 0, rtol=0, atol=tolerance)
    ):
        message = (
            "PAM needs a finite symmetric distance matrix and valid count"
        )
        raise ValueError(message)

    def objective(indices: tuple[int, ...] | list[int]) -> float:
        return float(np.min(matrix[:, indices], axis=1).sum())

    medoids: list[int] = []
    while len(medoids) < count:
        candidates = [
            index for index in range(len(matrix)) if index not in medoids
        ]
        nearest = (
            np.min(matrix[:, medoids], axis=1)
            if medoids
            else np.full(len(matrix), np.inf)
        )
        values = [
            (float(np.minimum(nearest, matrix[:, index]).sum()), index)
            for index in candidates
        ]
        best_value = min(value for value, _ in values)
        medoids.append(
            min(
                index
                for value, index in values
                if value <= best_value + tolerance
            ),
        )

    current = objective(medoids)
    while True:
        swaps: list[tuple[float, int, int]] = []
        selected = set(medoids)
        for removed in sorted(medoids):
            remaining = [item for item in medoids if item != removed]
            nearest_without_removed = (
                np.min(matrix[:, remaining], axis=1)
                if remaining
                else np.full(len(matrix), np.inf)
            )
            for added in range(len(matrix)):
                if added not in selected:
                    value = float(
                        np.minimum(
                            nearest_without_removed,
                            matrix[:, added],
                        ).sum(),
                    )
                    swaps.append((value, removed, added))
        if not swaps:
            break
        best_value = min(value for value, _, _ in swaps)
        if best_value == np.inf or current - best_value <= tolerance:
            break
        best = min(
            (removed, added)
            for value, removed, added in swaps
            if value < current and value - best_value <= tolerance
        )
        medoids[medoids.index(best[0])] = best[1]
        current = objective(medoids)
    return tuple(sorted(medoids))


def sample_unit_suite(
    method: str,
    dimension: int,
    budget: int,
    seed: np.random.SeedSequence,
) -> FloatArray:
    if dimension < 1 or budget < 1:
        message = "sampler dimension and budget must be positive"
        raise ValueError(message)
    copied = np.random.SeedSequence(
        seed.entropy,
        spawn_key=seed.spawn_key,
        pool_size=seed.pool_size,
    )
    rng = np.random.default_rng(copied)
    if method == "sobol":
        exponent = ceil(log2(budget)) if budget > 1 else 0
        values = qmc.Sobol(
            d=dimension,
            scramble=True,
            bits=64,
            optimization=None,
            rng=rng,
        ).random_base2(exponent)[:budget]
    elif method == "random":
        values = rng.random((budget, dimension))
    elif method == "lhs":
        values = qmc.LatinHypercube(
            d=dimension,
            scramble=True,
            strength=1,
            optimization=None,
            rng=rng,
        ).random(budget)
    else:
        message = f"unknown suite sampler: {method}"
        raise ValueError(message)
    return np.asarray(values, dtype=np.float64)


def scale_to_bounds(unit_values: FloatArray, bounds: FloatArray) -> FloatArray:
    values = _finite_matrix(unit_values, "unit scenarios")
    domain = _bounds(bounds, values.shape[1])
    if np.any(values < 0) or np.any(values > 1):
        message = "unit scenarios must lie in [0, 1]"
        raise ValueError(message)
    return np.asarray(
        domain[:, 0] + values * (domain[:, 1] - domain[:, 0]),
        dtype=np.float64,
    )


def _finite_matrix(values: FloatArray, label: str) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if (
        matrix.ndim != MATRIX_DIMENSIONS
        or min(matrix.shape) < 1
        or not np.all(np.isfinite(matrix))
    ):
        message = f"{label} must be a finite nonempty matrix"
        raise ValueError(message)
    return matrix


def _bounds(values: FloatArray, dimension: int) -> FloatArray:
    bounds = np.asarray(values, dtype=np.float64)
    if (
        bounds.shape != (dimension, 2)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[:, 1] <= bounds[:, 0])
    ):
        message = "bounds need one finite positive-width interval per input"
        raise ValueError(message)
    return bounds
