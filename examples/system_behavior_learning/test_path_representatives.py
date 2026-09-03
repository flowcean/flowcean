# Assertions and fixed numerical fixtures are intentional in tests.
# ruff: noqa: PLR2004, S101

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

REPOSITORY_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.system_behavior_learning import (  # noqa: E402
    run_path_representatives as runner,
)
from examples.system_behavior_learning.experiment import (  # noqa: E402
    CONFIG,
    SYSTEMS,
    PreparedSystem,
    SystemSpec,
    TargetTransform,
    prepare_system,
    sample_all_scenarios,
)
from examples.system_behavior_learning.path_representatives import (  # noqa: E402
    Interval,
    PathRepresentativeEvaluation,
    PathRepresentativeStudy,
    build_report,
    coordinate_rms,
    effective_input_boundary,
    evaluate_path_representatives,
    extract_leaf_regions,
    inverse_transform_complete_targets,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _stump() -> DecisionTreeRegressor:
    features = np.array([[0.5], [0.75], [1.25], [1.5]], dtype=np.float64)
    targets = np.array([0.0, 0.0, 1.0, 1.0])
    return DecisionTreeRegressor(max_depth=1, random_state=0).fit(
        features,
        targets,
    )


@pytest.mark.parametrize(
    ("lower", "equality_routes_left"),
    [
        (np.float32(1.0), True),
        (np.nextafter(np.float32(1.0), np.float32(np.inf)), False),
    ],
)
def test_effective_boundary_matches_apply_below_at_and_above(
    lower: np.float32,
    *,
    equality_routes_left: bool,
) -> None:
    tree = _stump()
    upper = np.nextafter(lower, np.float32(np.inf))
    threshold = float(lower) + (float(upper) - float(lower)) / 2.0
    tree.tree_.threshold[0] = threshold

    boundary, inclusive_left = effective_input_boundary(threshold)
    points = np.array(
        [
            [np.nextafter(boundary, -np.inf)],
            [boundary],
            [np.nextafter(boundary, np.inf)],
        ],
        dtype=np.float64,
    )
    routed = tree.apply(points)
    left_id = tree.tree_.children_left[0]
    right_id = tree.tree_.children_right[0]

    assert boundary == threshold
    assert inclusive_left is equality_routes_left
    np.testing.assert_array_equal(
        routed,
        [
            left_id,
            left_id if equality_routes_left else right_id,
            right_id,
        ],
    )


def _repeated_feature_tree() -> tuple[
    DecisionTreeRegressor,
    NDArray[np.float64],
]:
    first = np.linspace(0.01, 0.99, 120)
    features = np.column_stack((first, np.zeros(first.size)))
    targets = np.where(first < 0.3, -2.0, np.where(first < 0.7, 1.0, 4.0))
    tree = DecisionTreeRegressor(
        max_leaf_nodes=3,
        min_samples_leaf=10,
        random_state=0,
    ).fit(features, targets)
    return tree, features


def test_repeated_feature_intersections_partition_domain() -> None:
    tree, features = _repeated_feature_tree()
    bounds = ((0.0, 1.0), (-1.0, 1.0))
    regions = extract_leaf_regions(tree, bounds, ("repeated", "unused"))
    domain_widths = np.diff(np.asarray(bounds), axis=1).reshape(-1)
    volumes = [
        float(
            np.prod(
                np.asarray(
                    [interval.width for interval in region.intervals],
                )
                / domain_widths,
            ),
        )
        for region in regions
    ]

    assert len(regions) == 3
    assert any(
        sum(item.feature_index == 0 for item in region.predicates) == 2
        for region in regions
    )
    assert sum(volumes) == pytest.approx(1.0)
    assert all(volume > 0.0 for volume in volumes)
    for region in regions:
        assert int(tree.apply(region.midpoint.reshape(1, -1))[0]) == (
            region.leaf_id
        )
        fitting_members = features[tree.apply(features) == region.leaf_id]
        assert fitting_members.size > 0


def _synthetic_evaluation(
    name: str = "Synthetic",
    state_names: tuple[str, ...] = ("output",),
) -> PathRepresentativeEvaluation:
    fitting_features = np.linspace(0.05, 0.95, 12).reshape(-1, 1)
    assessment_features = np.array([[0.1], [0.3], [0.7], [0.9]])
    coordinate_count = 2 * len(state_names)
    multipliers = np.arange(1, coordinate_count + 1, dtype=np.float64)
    means = np.linspace(10.0, 10.0 + coordinate_count - 1, coordinate_count)
    scales = np.linspace(2.0, 2.0 + coordinate_count - 1, coordinate_count)

    def standardized_targets(
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return features * multipliers

    def physical_targets(
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return standardized_targets(features) * scales + means

    def simulator(
        scenario: NDArray[np.float64],
        _times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return physical_targets(scenario.reshape(1, -1))[0]

    spec = SystemSpec(
        name,
        ("input",),
        ((0.0, 1.0),),
        (0.0, 1.0),
        2,
        state_names,
        simulator,
    )
    transform = TargetTransform(
        means,
        scales,
        np.arange(coordinate_count, dtype=np.int64),
        np.array([], dtype=np.int64),
    )
    prepared = PreparedSystem(
        spec,
        fitting_features,
        assessment_features,
        standardized_targets(fitting_features),
        standardized_targets(assessment_features),
        transform,
    )
    tree = DecisionTreeRegressor(
        max_leaf_nodes=2,
        min_samples_leaf=2,
        random_state=0,
    ).fit(prepared.fitting_features, prepared.fitting_targets)
    return evaluate_path_representatives(prepared, tree)


def _synthetic_study(*, two_states: bool = False) -> PathRepresentativeStudy:
    state_names = ("position", "velocity") if two_states else ("output",)
    return PathRepresentativeStudy(
        tuple(
            _synthetic_evaluation(name, state_names)
            for name in runner.PROTOTYPE_NAMES
        ),
    )


def test_metric_prototypes_and_occupancies() -> None:
    evaluation = _synthetic_evaluation()
    fitting_assignments = evaluation.tree.apply(
        evaluation.prepared.fitting_features,
    )

    assert coordinate_rms(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == (
        pytest.approx(np.sqrt(12.5))
    )
    assert sum(leaf.fitting_occupancy for leaf in evaluation.leaves) == 12
    assert sum(leaf.assessment_occupancy for leaf in evaluation.leaves) == 4
    assert sum(leaf.relative_volume for leaf in evaluation.leaves) == (
        pytest.approx(1.0)
    )
    for leaf in evaluation.leaves:
        members = evaluation.prepared.fitting_targets[
            fitting_assignments == leaf.region.leaf_id
        ]
        np.testing.assert_allclose(leaf.prototype, members.mean(axis=0))
        np.testing.assert_allclose(
            leaf.prototype,
            evaluation.tree.predict(leaf.region.midpoint.reshape(1, -1))[0],
        )
        assert leaf.assessment_occupancy == leaf.held_residuals.size
        assert 0.0 <= leaf.midpoint_empirical_cdf_fraction <= 1.0


@pytest.fixture(scope="module")
def all_system_evaluations() -> tuple[PathRepresentativeEvaluation, ...]:
    scenario_pairs = sample_all_scenarios()
    return tuple(
        evaluate_path_representatives(
            prepare_system(spec, fitting, assessment),
        )
        for spec, (fitting, assessment) in zip(
            SYSTEMS,
            scenario_pairs,
            strict=True,
        )
    )


def test_all_systems_have_routable_occupied_eight_leaf_regions(
    all_system_evaluations: tuple[PathRepresentativeEvaluation, ...],
) -> None:
    assert [item.prepared.spec.name for item in all_system_evaluations] == [
        spec.name for spec in SYSTEMS
    ]
    for evaluation in all_system_evaluations:
        assert evaluation.tree.get_n_leaves() == 8
        assert len(evaluation.leaves) == 8
        assert all(leaf.assessment_occupancy > 0 for leaf in evaluation.leaves)
        assert evaluation.prepared.transform.excluded.size == 0
        assert sum(leaf.relative_volume for leaf in evaluation.leaves) == (
            pytest.approx(1.0)
        )
        for leaf in evaluation.leaves:
            routed = evaluation.tree.apply(leaf.region.midpoint.reshape(1, -1))
            assert int(routed[0]) == leaf.region.leaf_id


def test_report_schema_and_metadata_come_from_all_evaluations(
    all_system_evaluations: tuple[PathRepresentativeEvaluation, ...],
) -> None:
    study = PathRepresentativeStudy(all_system_evaluations)
    report = build_report(study)

    assert report["config"]["scenario_counts"] == {
        "fitting": 256,
        "assessment": 256,
        "path_midpoints_per_system": 8,
    }
    assert report["config"]["tree"] == {
        "estimator": "sklearn.tree.DecisionTreeRegressor",
        "max_leaf_nodes": 8,
        "fitted_leaf_count_per_system": 8,
        "min_samples_leaf": 16,
        "random_state": 0,
        "capacity_note": (
            "8 maximum leaves are fixed to isolate representative "
            "construction; this analysis does not select capacity."
        ),
    }
    assert [system["name"] for system in report["systems"]] == [
        spec.name for spec in SYSTEMS
    ]
    for system, evaluation in zip(
        report["systems"],
        all_system_evaluations,
        strict=True,
    ):
        prepared = evaluation.prepared
        assert system["state_names"] == list(prepared.spec.state_names)
        assert system["target_coordinates"] == {
            "total": int(prepared.transform.means.size),
            "retained": int(prepared.transform.retained.size),
            "excluded": int(prepared.transform.excluded.size),
        }
        assert len(system["domain"]) == len(prepared.spec.feature_names)
        assert len(system["leaves"]) == 8
        assert set(system["aggregate"]) == {
            "volume_weighted_held_mean",
            "volume_weighted_midpoint_error",
        }
        for leaf in system["leaves"]:
            assert "prototype" not in leaf
            assert "held_residuals" not in leaf


def test_report_is_compact_serializable_and_excludes_research_fields() -> None:
    report = build_report(_synthetic_study())
    serialized = json.dumps(report, allow_nan=False)

    for excluded in (
        "provenance",
        "sha256",
        "support_distance",
        "audit",
        "archive",
        "status",
        "criterion",
        "tolerance",
    ):
        assert excluded not in serialized.lower()

    report["systems"][0]["aggregate"]["volume_weighted_midpoint_error"] = (
        float("nan")
    )
    with pytest.raises(ValueError, match="Out of range float"):
        json.dumps(report, allow_nan=False)


def test_complete_inverse_transform_is_exact_and_rejects_exclusions() -> None:
    evaluation = _synthetic_evaluation()
    transform = evaluation.prepared.transform
    standardized = evaluation.prepared.fitting_targets
    expected = standardized * transform.scales + transform.means

    physical = inverse_transform_complete_targets(standardized, transform)

    np.testing.assert_allclose(physical, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        transform.transform(physical),
        standardized,
        rtol=1e-12,
        atol=1e-12,
    )
    incomplete = TargetTransform(
        np.zeros(2),
        np.ones(2),
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="every target coordinate"):
        inverse_transform_complete_targets(np.zeros((2, 1)), incomplete)


def test_combined_plot_smoke_has_one_panel_per_system() -> None:
    figure = runner.create_combined_figure(_synthetic_study())

    try:
        assert len(figure.axes) == 4
        assert {axis.get_title() for axis in figure.axes} == set(
            runner.PROTOTYPE_NAMES,
        )
        assert all(
            "Standardized-coordinate RMS" in axis.get_xlabel()
            for axis in figure.axes
        )
    finally:
        plt.close(figure)


def test_prototype_grid_has_state_columns_and_shared_column_limits() -> None:
    evaluation = _synthetic_evaluation(
        "Thermostat",
        ("position", "velocity"),
    )
    figure = runner.create_prototype_figure(evaluation)
    prepared = evaluation.prepared
    trajectory_shape = (
        -1,
        prepared.spec.sample_count,
        prepared.spec.state_count,
    )
    fitting = inverse_transform_complete_targets(
        prepared.fitting_targets,
        prepared.transform,
    ).reshape(trajectory_shape)
    prototypes = inverse_transform_complete_targets(
        np.stack([leaf.prototype for leaf in evaluation.leaves]),
        prepared.transform,
    ).reshape(trajectory_shape)
    midpoints = inverse_transform_complete_targets(
        np.stack([leaf.midpoint_target for leaf in evaluation.leaves]),
        prepared.transform,
    ).reshape(trajectory_shape)
    assignments = evaluation.tree.apply(prepared.fitting_features)

    try:
        axes = np.asarray(figure.axes, dtype=object).reshape(2, 2)
        assert [axis.get_title() for axis in axes[0]] == [
            "position",
            "velocity",
        ]
        for column in range(2):
            assert axes[0, column].get_ylim() == axes[1, column].get_ylim()
        for row, leaf in enumerate(evaluation.leaves):
            assert f"leaf {leaf.region.leaf_id}" in axes[row, 0].get_ylabel()
            members = fitting[assignments == leaf.region.leaf_id]
            for column in range(2):
                lines = axes[row, column].lines
                assert len(lines) == leaf.fitting_occupancy + 2
                for line, member in zip(
                    lines[:-2],
                    members[:, :, column],
                    strict=True,
                ):
                    np.testing.assert_allclose(line.get_ydata(), member)
                np.testing.assert_allclose(
                    lines[-2].get_ydata(),
                    prototypes[row, :, column],
                )
                np.testing.assert_allclose(
                    lines[-1].get_ydata(),
                    midpoints[row, :, column],
                )
    finally:
        plt.close(figure)


def test_cli_smoke_writes_exact_six_artifact_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _synthetic_study(two_states=True)
    report = build_report(study, CONFIG)
    monkeypatch.setattr(
        runner,
        "run_path_representatives",
        lambda: (report, study),
    )

    runner.main(["--output-dir", str(tmp_path)])

    assert {path.name for path in tmp_path.iterdir()} == {
        "path_representatives.json",
        "path_representatives.png",
        *runner.PROTOTYPE_NAMES.values(),
    }
    json.loads(
        (tmp_path / "path_representatives.json").read_text(encoding="utf-8"),
    )


def test_cli_requires_output_directory() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args([])


def test_interval_midpoint_uses_arithmetic_box_center() -> None:
    interval = Interval(
        0.1,
        0.7,
        lower_inclusive=False,
        upper_inclusive=True,
    )

    assert interval.width == pytest.approx(0.6)
    assert interval.midpoint == pytest.approx(0.4)
