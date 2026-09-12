from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from methods import fit_behavior_tree, tree_midpoints
from metrics import (
    coordinate_rms,
    cross_coordinate_rms,
    midpoint_realization_metrics,
    pairwise_coordinate_rms,
    prediction_metrics,
    reference_coverage_metrics,
)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dimensions", [1, 3, 128, 513])
def test_rms_roundoff_for_identical_standard_scale_rows(
    order: Literal["C", "F"],
    dimensions: int,
) -> None:
    row = np.random.default_rng(1).normal(size=(1, dimensions))
    # Gram cancellation near zero has an O(sqrt(machine epsilon)) floor.
    np.testing.assert_allclose(
        cross_coordinate_rms(row, row),
        [[0.0]],
        rtol=0,
        atol=1e-7,
    )
    values = np.array(np.repeat(row, 3, axis=0), order=order)
    np.testing.assert_allclose(
        cross_coordinate_rms(values, values.copy()),
        np.zeros((3, 3)),
        rtol=0,
        atol=1e-7,
    )
    pairwise = pairwise_coordinate_rms(values)
    np.testing.assert_allclose(pairwise, np.zeros((3, 3)), rtol=0, atol=1e-7)
    np.testing.assert_array_equal(np.diag(pairwise), np.zeros(3))


def test_cross_rms_on_centered_large_offset_rows() -> None:
    # Matrix distances expect well-scaled inputs, unlike the direct scalar RMS.
    offset = 1e8
    left = np.array([[offset], [offset + 2]]) - offset
    right = np.array([[offset + 1], [offset + 4]]) - offset
    assert coordinate_rms(np.array([offset]), np.array([offset + 1])) == 1.0
    np.testing.assert_array_equal(
        cross_coordinate_rms(left, right),
        [[1, 4], [1, 2]],
    )
    np.testing.assert_array_equal(
        pairwise_coordinate_rms(left),
        [[0, 2], [2, 0]],
    )


@pytest.mark.parametrize("left_order", ["C", "F"])
@pytest.mark.parametrize("right_order", ["C", "F"])
@pytest.mark.parametrize("dimensions", [1, 3, 128, 513])
def test_rms_matches_independent_direct_difference_oracle(
    left_order: Literal["C", "F"],
    right_order: Literal["C", "F"],
    dimensions: int,
) -> None:
    rng = np.random.default_rng(13)
    left = np.array(rng.normal(size=(5, dimensions)), order=left_order)
    right = np.array(rng.normal(size=(7, dimensions)), order=right_order)
    for second in (left, right):
        expected = np.array(
            [
                [
                    np.sqrt(np.mean((first_row - second_row) ** 2))
                    for second_row in second
                ]
                for first_row in left
            ],
        )
        actual = cross_coordinate_rms(left, second)
        assert actual.dtype == np.float64
        nonzero = expected > 0
        np.testing.assert_allclose(
            actual[nonzero],
            expected[nonzero],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(actual[~nonzero], 0, rtol=0, atol=1e-7)
        if second is left:
            pairwise = pairwise_coordinate_rms(left)
            np.testing.assert_allclose(
                pairwise[nonzero],
                expected[nonzero],
                rtol=1e-12,
                atol=1e-12,
            )
            np.testing.assert_array_equal(np.diag(pairwise), np.zeros(5))


def test_prediction_and_midpoint_realization_hand_calculations() -> None:
    fitting_x = np.array([[0.0], [1.0]], dtype=np.float64)
    fitting_y = np.array([[0.0], [2.0]], dtype=np.float64)
    assessment_x = np.array([[0.25], [0.75]], dtype=np.float64)
    assessment_y = np.array([[0.5], [1.5]], dtype=np.float64)
    tree = fit_behavior_tree(
        fitting_x,
        fitting_y,
        2,
        np.random.SeedSequence(3),
    )
    prediction = prediction_metrics(
        tree,
        fitting_x,
        fitting_y,
        assessment_x,
        assessment_y,
    )
    midpoints = tree_midpoints(tree, np.array([[0.0, 1.0]], dtype=np.float64))
    midpoint_y = np.array([[0.25], [1.75]], dtype=np.float64)
    realization = midpoint_realization_metrics(
        tree,
        assessment_x,
        assessment_y,
        midpoints,
        midpoint_y,
        np.array([0.5, 0.5], dtype=np.float64),
    )

    assert coordinate_rms(
        np.array([0.0, 2.0]),
        np.array([0.0, 0.0]),
    ) == pytest.approx(
        np.sqrt(2.0),
    )
    assert prediction.fitting_rmse == 0.0
    assert prediction.assessment_rmse == pytest.approx(0.5)
    assert prediction.historical_mean_rmse == pytest.approx(0.5)
    assert prediction.prediction_ratio == pytest.approx(1.0)
    assert realization.midpoint_realization_distance == pytest.approx(0.25)
    assert realization.held_leaf_distance == pytest.approx(0.5)
    assert realization.realization_ratio == pytest.approx(0.5)
    assert realization.assessment_unrepresented_volume == pytest.approx(0.0)
    assert [leaf.assessment_members for leaf in realization.leaves] == [1, 1]
    assert [
        leaf.midpoint_realization_distance for leaf in realization.leaves
    ] == pytest.approx([0.25, 0.25])
    assert [leaf.held_leaf_distance for leaf in realization.leaves] == (
        pytest.approx([0.5, 0.5])
    )
    assert [leaf.realization_ratio for leaf in realization.leaves] == (
        pytest.approx([0.5, 0.5])
    )


def test_midpoint_realization_uses_assessment_occupancy() -> None:
    fitting_x = np.array([[0.0], [1.0]], dtype=np.float64)
    fitting_y = np.array([[0.0], [2.0]], dtype=np.float64)
    tree = fit_behavior_tree(
        fitting_x,
        fitting_y,
        2,
        np.random.SeedSequence(4),
    )
    assessment_x = np.array([[0.1], [0.2], [0.3], [0.9]])
    assessment_y = np.array([[1.0], [1.0], [1.0], [4.0]])
    midpoints = tree_midpoints(tree, np.array([[0.0, 1.0]]))
    midpoint_y = np.array([[0.0], [4.0]])

    result = midpoint_realization_metrics(
        tree,
        assessment_x,
        assessment_y,
        midpoints,
        midpoint_y,
        np.array([0.5, 0.5]),
    )

    assert result.midpoint_realization_distance == pytest.approx(0.5)
    assert result.held_leaf_distance == pytest.approx(1.25)
    assert result.realization_ratio == pytest.approx(0.4)
    assert result.assessment_unrepresented_volume == pytest.approx(0.0)


def test_midpoint_realization_reports_unrepresented_leaf_volume() -> None:
    fitting_x = np.array([[0.0], [1.0]], dtype=np.float64)
    fitting_y = np.array([[0.0], [2.0]], dtype=np.float64)
    tree = fit_behavior_tree(
        fitting_x,
        fitting_y,
        2,
        np.random.SeedSequence(5),
    )
    assessment_x = np.array([[0.1], [0.2]], dtype=np.float64)
    assessment_y = np.array([[1.0], [1.0]], dtype=np.float64)
    midpoints = tree_midpoints(tree, np.array([[0.0, 1.0]]))

    result = midpoint_realization_metrics(
        tree,
        assessment_x,
        assessment_y,
        midpoints,
        np.array([[0.5], [4.0]]),
        np.array([0.5, 0.5]),
    )

    assert result.midpoint_realization_distance == pytest.approx(0.5)
    assert result.held_leaf_distance == pytest.approx(1.0)
    assert result.realization_ratio == pytest.approx(0.5)
    assert result.assessment_unrepresented_volume == pytest.approx(0.5)
    represented, missing = result.leaves
    assert represented.assessment_members == 2
    assert represented.midpoint_realization_distance == pytest.approx(0.5)
    assert represented.held_leaf_distance == pytest.approx(1.0)
    assert represented.realization_ratio == pytest.approx(0.5)
    assert missing.assessment_members == 0
    assert missing.midpoint_realization_distance == pytest.approx(2.0)
    assert missing.held_leaf_distance is None
    assert missing.realization_ratio is None


def test_reference_coverage_and_spacing_hand_calculation() -> None:
    suite_targets = np.array([[0.0], [2.0]], dtype=np.float64)
    reference_targets = np.array([[0.0], [1.0], [3.0]], dtype=np.float64)
    scenarios = np.array([[0.0], [1.0]], dtype=np.float64)

    result = reference_coverage_metrics(
        suite_targets,
        reference_targets,
        scenarios,
        np.array([[0.0, 1.0]], dtype=np.float64),
    )

    nearest = np.array([0.0, 1.0, 1.0])
    # The middle reference row ties: the first suite row wins.
    np.testing.assert_array_equal(result.reference_assignments, [0, 0, 1])
    np.testing.assert_array_equal(result.reference_distances, nearest)
    assert result.reference_assignments.dtype == np.int64
    assert result.reference_distances.dtype == np.float64
    assert result.mean_distance == pytest.approx(2.0 / 3.0)
    assert result.q95_distance == pytest.approx(np.quantile(nearest, 0.95))
    assert result.q99_distance == pytest.approx(np.quantile(nearest, 0.99))
    assert result.max_distance == 1.0
    assert result.used_members == 2
    assert result.mean_input_spacing == 1.0
    assert result.scaled_input_spacing == 2.0


def test_zero_denominators_are_undefined_not_simulation_failures() -> None:
    fitting_x = np.array([[0.0], [1.0]])
    truth = np.zeros((2, 1))
    tree = fit_behavior_tree(fitting_x, truth, 2, np.random.SeedSequence(3))
    prediction = prediction_metrics(tree, fitting_x, truth, fitting_x, truth)
    assert prediction.fitting_rmse == 0
    assert prediction.assessment_rmse == 0
    assert prediction.historical_mean_rmse == 0
    assert prediction.prediction_ratio is None
    realization = midpoint_realization_metrics(
        tree,
        fitting_x,
        truth,
        np.array([[0.5]]),
        np.array([[1.0]]),
        np.array([1.0]),
    )
    assert realization.midpoint_realization_distance == 1
    assert realization.held_leaf_distance == 0
    assert realization.realization_ratio is None
    assert realization.leaves[0].realization_ratio is None
