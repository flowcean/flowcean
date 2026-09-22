from __future__ import annotations

import inspect
import sys
from itertools import combinations
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from methods import (
    FloatArray,
    effective_input_boundary,
    fit_behavior_tree,
    fit_input_only_tree,
    fit_unbounded_behavior_tree,
    leaf_boxes,
    pam_medoids,
    sample_unit_suite,
    scale_to_bounds,
)
from metrics import pairwise_coordinate_rms

if TYPE_CHECKING:
    from types import FrameType

    from _typeshed import TraceFunction


def test_float32_path_boundary_midpoint_routing_and_volume() -> None:
    scenarios = np.array([[0.1], [0.2], [0.8], [0.9]], dtype=np.float64)
    targets = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float64)
    tree = fit_behavior_tree(
        scenarios,
        targets,
        2,
        np.random.SeedSequence(1),
    )
    boxes = leaf_boxes(tree, np.array([[0.0, 1.0]], dtype=np.float64))
    boundary, left_inclusive = effective_input_boundary(
        tree.tree_.threshold[0],
    )
    boundary_leaf = int(
        tree.apply(np.array([[boundary]], dtype=np.float64))[0],
    )
    expected_leaf = int(
        tree.tree_.children_left[0]
        if left_inclusive
        else tree.tree_.children_right[0],
    )

    assert boundary_leaf == expected_leaf
    assert len(boxes) == 2
    assert [box.depth for box in boxes] == [1, 1]
    assert sum(box.relative_volume for box in boxes) == pytest.approx(1.0)
    np.testing.assert_array_equal(
        tree.apply(np.stack([box.midpoint for box in boxes])),
        [box.leaf_id for box in boxes],
    )


def test_behavior_tree_capacity_is_an_upper_bound() -> None:
    scenarios = np.array([[0.1], [0.2], [0.8], [0.9]], dtype=np.float64)
    targets = np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float64)

    tree = fit_behavior_tree(
        scenarios,
        targets,
        4,
        np.random.SeedSequence(2),
    )

    assert tree.get_n_leaves() == 2


def test_input_only_tree_reports_its_realized_size() -> None:
    scenarios = np.full((4, 1), 0.5, dtype=np.float64)
    tree = fit_input_only_tree(
        scenarios,
        np.array([[0.0, 1.0]], dtype=np.float64),
        2,
        np.random.SeedSequence(3),
    )

    assert tree.get_n_leaves() == 1


def test_unbounded_behavior_tree_is_deterministic_and_fully_grown() -> None:
    scenarios = np.linspace(0.0, 1.0, 12, dtype=np.float64).reshape(-1, 1)
    targets = np.square(scenarios)

    first = fit_unbounded_behavior_tree(
        scenarios,
        targets,
        np.random.SeedSequence(3),
    )
    second = fit_unbounded_behavior_tree(
        scenarios,
        targets,
        np.random.SeedSequence(3),
    )

    assert first.get_n_leaves() == len(scenarios)
    np.testing.assert_array_equal(
        first.apply(scenarios),
        second.apply(scenarios),
    )


def test_input_only_tree_uses_only_normalized_inputs() -> None:
    assert (
        "trajectories" not in inspect.signature(fit_input_only_tree).parameters
    )
    scenarios = np.linspace(10.0, 20.0, 20, dtype=np.float64).reshape(-1, 1)
    bounds = np.array([[10.0, 20.0]], dtype=np.float64)
    tree = fit_input_only_tree(
        scenarios,
        bounds,
        4,
        np.random.SeedSequence(2),
    )
    assignments = tree.apply(scenarios)
    leaves = sorted(np.unique(assignments))
    normalized = (scenarios - 10.0) / 10.0
    expected = [normalized[assignments == leaf].mean() for leaf in leaves]

    assert tree.get_n_leaves() == 4
    np.testing.assert_allclose(tree.tree_.value[leaves, 0, 0], expected)
    assert len(leaf_boxes(tree, bounds)) == 4


def test_samplers_are_reproducible_bounded_and_follow_sobol_prefix() -> None:
    for method in ("sobol", "random", "lhs"):
        seed = np.random.SeedSequence(19)
        first = sample_unit_suite(method, 3, 3, seed)
        second = sample_unit_suite(method, 3, 3, seed)
        np.testing.assert_array_equal(first, second)
        assert first.shape == (3, 3)
        assert np.all((first >= 0.0) & (first <= 1.0))
        scaled = scale_to_bounds(
            first,
            np.array([[1.0, 2.0], [-2.0, 2.0], [10.0, 20.0]]),
        )
        assert np.all(scaled[:, 0] >= 1.0)
        assert np.all(scaled[:, 2] <= 20.0)

    sobol_three = sample_unit_suite("sobol", 2, 3, np.random.SeedSequence(7))
    sobol_four = sample_unit_suite("sobol", 2, 4, np.random.SeedSequence(7))
    np.testing.assert_array_equal(sobol_three, sobol_four[:3])


def test_pairwise_distances_support_high_dimensional_pam() -> None:
    values = np.random.default_rng(13).normal(size=(64, 512))
    distances = pairwise_coordinate_rms(values)

    np.testing.assert_array_equal(np.diag(distances), np.zeros(64))
    np.testing.assert_allclose(distances, distances.T, rtol=0, atol=0)
    assert len(pam_medoids(distances, 4)) == 4


def test_pam_known_case_and_lowest_index_tie() -> None:
    line = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    distances = pairwise_coordinate_rms(line)

    assert pam_medoids(distances, 1) == (1,)
    assert pam_medoids(distances, 2) == (1, 2)


def _original_pam_medoids(
    distances: FloatArray,
    count: int,
    tolerance: float = 1e-12,
) -> tuple[int, ...]:
    """Frozen original BUILD/SWAP implementation for valid test inputs."""
    matrix = np.asarray(distances, dtype=np.float64)

    def objective(indices: tuple[int, ...] | list[int]) -> float:
        return float(np.min(matrix[:, indices], axis=1).sum())

    medoids: list[int] = []
    while len(medoids) < count:
        candidates = [
            index for index in range(len(matrix)) if index not in medoids
        ]
        values = [
            (objective((*medoids, index)), index) for index in candidates
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
            for added in range(len(matrix)):
                if added not in selected:
                    trial = [
                        added if item == removed else item for item in medoids
                    ]
                    swaps.append((objective(trial), removed, added))
        if not swaps:
            break
        best_value = min(value for value, _, _ in swaps)
        best = min(
            (removed, added)
            for value, removed, added in swaps
            if value <= best_value + tolerance
        )
        if best_value >= current - tolerance:
            break
        medoids[medoids.index(best[0])] = best[1]
        current = best_value
    return tuple(sorted(medoids))


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("tolerance", [0.0, 1e-12, 0.25])
def test_pam_matches_original_random_matrices(
    order: Literal["C", "F"],
    tolerance: float,
) -> None:
    rng = np.random.default_rng(20260720)
    for size in (7, 17):
        raw = rng.random((size, size))
        symmetric = (raw + raw.T) / 2
        np.fill_diagonal(symmetric, 0.0)
        tied = rng.integers(0, 4, size=(size, size)).astype(np.float64)
        tied = tied + tied.T
        np.fill_diagonal(tied, 0.0)
        distances = pairwise_coordinate_rms(rng.normal(size=(size, 11)))
        for values in (symmetric, tied, distances):
            matrix = np.array(values, order=order)
            for count in (1, 2, 4, size):
                assert pam_medoids(matrix, count, tolerance) == (
                    _original_pam_medoids(matrix, count, tolerance)
                )


@pytest.mark.parametrize("order", ["C", "F"])
def test_pam_matches_original_zero_distances(
    order: Literal["C", "F"],
) -> None:
    for size in (1, 7):
        matrix = np.zeros((size, size), order=order)
        for count in (1, size):
            for tolerance in (0.0, 1e-12, 0.25):
                assert (
                    pam_medoids(matrix, count, tolerance)
                    == (_original_pam_medoids(matrix, count, tolerance))
                    == tuple(range(count))
                )


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("gap_factor", [0.25, 0.5, 0.75])
def test_pam_matches_original_near_tolerance(
    order: Literal["C", "F"],
    gap_factor: float,
) -> None:
    # Binary fractions put the gap below, exactly at, and above tolerance.
    tolerance = 2.0**-40
    gap = gap_factor * tolerance
    matrix = np.array(
        [
            [0.0, 1.0, 1.0 + gap],
            [1.0, 0.0, 1.0 - gap],
            [1.0 + gap, 1.0 - gap, 0.0],
        ],
        order=order,
    )
    expected = (0,) if gap_factor <= 0.5 else (1,)
    assert pam_medoids(matrix, 1, tolerance) == expected
    assert pam_medoids(matrix, 1, 0.0) == (1,)
    for count in (1, 2, 3):
        assert pam_medoids(matrix, count) == _original_pam_medoids(
            matrix,
            count,
        )
        for threshold in (0.0, tolerance, 2 * tolerance):
            assert pam_medoids(matrix, count, threshold) == (
                _original_pam_medoids(matrix, count, threshold)
            )


def _medoid_objective(matrix: FloatArray, medoids: tuple[int, ...]) -> float:
    # Enumerate row minima independently of PAM's cached nearest vectors.
    return float(
        np.sum(
            [min(float(row[index]) for index in medoids) for row in matrix],
        ),
    )


def _assert_one_swap_optimal(
    matrix: FloatArray,
    medoids: tuple[int, ...],
    tolerance: float,
) -> None:
    current = _medoid_objective(matrix, medoids)
    for candidate in combinations(range(len(matrix)), len(medoids)):
        if len(set(candidate) - set(medoids)) == 1:
            assert current - _medoid_objective(matrix, candidate) <= tolerance


def _bounded_pam_medoids(
    matrix: FloatArray,
    count: int,
    tolerance: float,
) -> tuple[int, ...]:
    # Fail fast on a cycling SWAP without adding a production iteration cap.
    line_count = 0

    def trace(
        frame: FrameType,
        event: str,
        _arg: object,
    ) -> TraceFunction:
        nonlocal line_count
        if frame.f_code is pam_medoids.__code__ and event == "line":
            line_count += 1
            if line_count > 10_000:
                pytest.fail("PAM exceeded the boundary-test execution budget")
        return trace

    previous_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        return pam_medoids(matrix, count, tolerance)
    finally:
        sys.settrace(previous_trace)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("scale", [2.0**-40, 1.0, 2.0**40])
def test_pam_terminates_at_rounded_tolerance_boundary(
    order: Literal["C", "F"],
    scale: float,
) -> None:
    below = np.nextafter(0.125, 0.0)
    matrix = scale * np.array(
        [[0.0, 0.375, below], [0.375, 0.0, below], [below, below, 0.0]],
        order=order,
    )
    tolerance = 0.25 * scale
    current = _medoid_objective(matrix, (0,))
    best = _medoid_objective(matrix, (2,))
    assert current == _medoid_objective(matrix, (1,))
    assert best + tolerance == current
    assert best < current - tolerance
    # The actual floating-point difference is exactly tolerance, so no swap
    # is required. Rounded thresholds previously allowed a 0 -> 1 -> 0 cycle.
    assert current - best == tolerance

    medoids = _bounded_pam_medoids(matrix, 1, tolerance)
    assert medoids == (0,)
    _assert_one_swap_optimal(matrix, medoids, tolerance)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("scale", [2.0**-40, 1.0, 2.0**40])
def test_pam_accepts_one_ulp_improvement_above_tolerance(
    order: Literal["C", "F"],
    scale: float,
) -> None:
    above = np.nextafter(5e7, np.inf)
    below = np.nextafter(5e7, 0.0)
    matrix = scale * np.array(
        [[0.0, above, below], [above, 0.0, below], [below, below, 0.0]],
        order=order,
    )
    tolerance = 1e-8 * scale
    current = _medoid_objective(matrix, (0,))
    best = _medoid_objective(matrix, (2,))
    assert current == _medoid_objective(matrix, (1,)) == 1e8 * scale
    assert current - best == np.spacing(current)
    assert current - best > tolerance
    assert best >= current - tolerance
    # BUILD picks 0; the rounded tie threshold also admits nonimproving 1.
    # SWAP must still select the genuinely improving candidate 2.
    assert best + tolerance == current

    medoids = _bounded_pam_medoids(matrix, 1, tolerance)
    assert medoids == (2,)
    _assert_one_swap_optimal(matrix, medoids, tolerance)


def test_pam_handles_all_swap_objectives_overflowing() -> None:
    matrix = np.full((3, 3), np.finfo(np.float64).max)
    np.fill_diagonal(matrix, 0.0)
    with np.errstate(over="ignore"):
        assert _bounded_pam_medoids(matrix, 1, 0.0) == (0,)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    ("scale", "tolerance"),
    [(1.0, 1.0), (2.0**-40, 2.0**-40), (2.0**-40, 1e-12)],
)
def test_pam_tracks_actual_selected_swap_objective(
    order: Literal["C", "F"],
    scale: float,
    tolerance: float,
) -> None:
    matrix = scale * np.array(
        [
            [0, 13, 10, 9, 12, 3],
            [13, 0, 10, 3, 4, 2],
            [10, 10, 0, 6, 6, 8],
            [9, 3, 6, 0, 5, 15],
            [12, 4, 6, 5, 0, 15],
            [3, 2, 8, 15, 15, 0],
        ],
        dtype=np.float64,
        order=order,
    )
    # Independent full objectives: BUILD [1, 0] costs 19s. Its first
    # lexicographic tolerance-tied swap selects [1, 5] (18s), not [3, 0]
    # (17s). The best next swap [3, 5] (16s) improves by MORE than tolerance.
    # The next lexicographic tie selects [2, 5] (17s), locally optimal within
    # tolerance, even though [3, 5] is globally better.
    objectives = {
        indices: _medoid_objective(matrix, indices)
        for indices in combinations(range(len(matrix)), 2)
    }
    assert objectives[(0, 1)] == 19 * scale
    assert objectives[(0, 3)] == 17 * scale
    assert objectives[(1, 5)] == 18 * scale
    assert objectives[(2, 5)] == 17 * scale
    assert objectives[(3, 5)] == 16 * scale
    assert objectives[(1, 5)] - objectives[(3, 5)] > tolerance
    assert objectives[(2, 5)] - objectives[(3, 5)] <= tolerance

    medoids = pam_medoids(matrix, 2, tolerance)
    assert medoids == (2, 5)
    _assert_one_swap_optimal(matrix, medoids, tolerance)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("tolerance", [0.0, 1e-12, 0.25, 2.0])
@pytest.mark.parametrize("scale", [1.0, 2.0**-40])
def test_pam_is_one_swap_optimal(
    order: Literal["C", "F"],
    tolerance: float,
    scale: float,
) -> None:
    for seed in range(3):
        rng = np.random.default_rng(seed)
        for size in (4, 7, 10):
            raw = rng.random((size, size))
            symmetric = (raw + raw.T) / 2
            tied = rng.integers(0, 4, size=(size, size)).astype(np.float64)
            tied += tied.T.copy()
            rms = pairwise_coordinate_rms(rng.normal(size=(size, 5)))
            for values in (symmetric, tied, rms):
                matrix = np.array(scale * values, order=order)
                np.fill_diagonal(matrix, 0.0)
                for count in (1, 2, 3, size):
                    medoids = pam_medoids(matrix, count, tolerance)
                    assert len(set(medoids)) == count
                    _assert_one_swap_optimal(matrix, medoids, tolerance)
