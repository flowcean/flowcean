from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if __package__:
    from .experiment import (
        CONFIG,
        SYSTEMS,
        Config,
        PreparedSystem,
        TargetTransform,
        make_tree,
        prepare_system,
        sample_all_scenarios,
    )
else:
    from experiment import (  # pyright: ignore[reportImplicitRelativeImport]
        CONFIG,
        SYSTEMS,
        Config,
        PreparedSystem,
        TargetTransform,
        make_tree,
        prepare_system,
        sample_all_scenarios,
    )

if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeRegressor

FloatArray = NDArray[np.float64]
TREE_LEAF = -1
EXPECTED_LEAF_COUNT = 8
VOLUME_ATOL = 1e-12
MATRIX_DIMENSIONS = 2


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        return self.lower + self.width / 2.0


@dataclass(frozen=True)
class PathPredicate:
    node_id: int
    feature_index: int
    feature_name: str
    operator: str
    model_threshold: float
    effective_boundary: float
    inclusive: bool


@dataclass(frozen=True)
class LeafRegion:
    leaf_id: int
    predicates: tuple[PathPredicate, ...]
    intervals: tuple[Interval, ...]

    @property
    def midpoint(self) -> FloatArray:
        return np.asarray(
            [interval.midpoint for interval in self.intervals],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class LeafComparison:
    region: LeafRegion
    prototype: FloatArray
    midpoint_target: FloatArray
    held_residuals: FloatArray
    relative_volume: float
    fitting_occupancy: int
    assessment_occupancy: int
    midpoint_error: float
    midpoint_empirical_cdf_fraction: float


@dataclass(frozen=True)
class PathRepresentativeEvaluation:
    prepared: PreparedSystem
    tree: DecisionTreeRegressor
    leaves: tuple[LeafComparison, ...]
    volume_weighted_midpoint_error: float
    volume_weighted_held_mean: float


@dataclass(frozen=True)
class PathRepresentativeStudy:
    evaluations: tuple[PathRepresentativeEvaluation, ...]

    def __post_init__(self) -> None:
        if not self.evaluations:
            msg = "a path-representative study needs at least one system"
            raise ValueError(msg)
        names = [item.prepared.spec.name for item in self.evaluations]
        if len(set(names)) != len(names):
            msg = "a path-representative study cannot repeat a system"
            raise ValueError(msg)


def inverse_transform_complete_targets(
    standardized_targets: FloatArray,
    transform: TargetTransform,
) -> FloatArray:
    """Recover physical targets only when every coordinate was retained."""
    values = np.asarray(standardized_targets, dtype=np.float64)
    was_vector = values.ndim == 1
    matrix = values.reshape(1, -1) if was_vector else values
    expected_retained = np.arange(transform.means.size, dtype=np.int64)
    if transform.excluded.size or not np.array_equal(
        transform.retained,
        expected_retained,
    ):
        msg = (
            "exact physical-coordinate inversion requires every target "
            f"coordinate to be retained; {transform.excluded.size} excluded"
        )
        raise ValueError(msg)
    if (
        matrix.ndim != MATRIX_DIMENSIONS
        or matrix.shape[1] != transform.retained.size
        or not np.all(np.isfinite(matrix))
    ):
        msg = f"invalid standardized target shape {values.shape}"
        raise ValueError(msg)

    physical = (
        matrix * transform.scales[transform.retained]
        + transform.means[transform.retained]
    )
    round_trip = transform.transform(physical)
    if not np.allclose(round_trip, matrix, rtol=1e-12, atol=1e-12):
        msg = "target inverse failed its standardization round-trip check"
        raise ValueError(msg)
    return physical[0] if was_vector else physical


def coordinate_rms(first: FloatArray, second: FloatArray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.size == 0:
        msg = (
            "coordinate RMS requires equal nonempty shapes: "
            f"{left.shape} and {right.shape}"
        )
        raise ValueError(msg)
    value = float(np.sqrt(np.mean(np.square(left - right))))
    if not np.isfinite(value):
        msg = "coordinate RMS must be finite"
        raise ValueError(msg)
    return value


def effective_input_boundary(model_threshold: float) -> tuple[float, bool]:
    """Return the float64 boundary and whether equality routes left."""
    threshold = float(model_threshold)
    float32_max = float(np.finfo(np.float32).max)
    if (
        not np.isfinite(threshold)
        or threshold < -float32_max
        or threshold >= float32_max
    ):
        msg = f"threshold has no finite adjacent float32 pair: {threshold}"
        raise ValueError(msg)

    rounded = np.float32(threshold)
    lower = (
        np.nextafter(rounded, np.float32(-np.inf))
        if float(rounded) > threshold
        else rounded
    )
    upper = np.nextafter(lower, np.float32(np.inf))
    if not np.isfinite(lower) or not np.isfinite(upper):
        msg = f"threshold has no finite adjacent float32 pair: {threshold}"
        raise ValueError(msg)

    boundary = float(lower) + (float(upper) - float(lower)) / 2.0
    equality_routes_left = float(np.float32(boundary)) <= threshold
    return boundary, equality_routes_left


def _tighten_upper(
    interval: Interval,
    boundary: float,
    *,
    inclusive: bool,
) -> Interval:
    if boundary < interval.upper:
        return Interval(
            interval.lower,
            boundary,
            interval.lower_inclusive,
            inclusive,
        )
    if boundary == interval.upper:
        return Interval(
            interval.lower,
            interval.upper,
            interval.lower_inclusive,
            interval.upper_inclusive and inclusive,
        )
    return interval


def _tighten_lower(
    interval: Interval,
    boundary: float,
    *,
    inclusive: bool,
) -> Interval:
    if boundary > interval.lower:
        return Interval(
            boundary,
            interval.upper,
            inclusive,
            interval.upper_inclusive,
        )
    if boundary == interval.lower:
        return Interval(
            interval.lower,
            interval.upper,
            interval.lower_inclusive and inclusive,
            interval.upper_inclusive,
        )
    return interval


def _checked_domain(
    domain_bounds: tuple[tuple[float, float], ...],
    feature_names: tuple[str, ...],
) -> FloatArray:
    bounds = np.asarray(domain_bounds, dtype=np.float64)
    if (
        bounds.shape != (len(feature_names), 2)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[:, 1] <= bounds[:, 0])
    ):
        msg = (
            "the declared domain must have one finite positive-width "
            "interval per feature"
        )
        raise ValueError(msg)
    return bounds


def extract_leaf_regions(
    tree: DecisionTreeRegressor,
    domain_bounds: tuple[tuple[float, float], ...],
    feature_names: tuple[str, ...],
) -> tuple[LeafRegion, ...]:
    """Intersect root-to-leaf predicates in the declared float64 domain."""
    bounds = _checked_domain(domain_bounds, feature_names)
    if tree.n_features_in_ != len(feature_names):
        msg = "tree and declared domain feature counts differ"
        raise ValueError(msg)

    initial = tuple(
        Interval(float(lower), float(upper)) for lower, upper in bounds
    )
    structure = tree.tree_
    regions: list[LeafRegion] = []

    def visit(
        node_id: int,
        predicates: tuple[PathPredicate, ...],
        intervals: tuple[Interval, ...],
    ) -> None:
        left_id = int(structure.children_left[node_id])
        right_id = int(structure.children_right[node_id])
        if left_id == TREE_LEAF:
            if right_id != TREE_LEAF:
                msg = f"tree node {node_id} has only one child"
                raise ValueError(msg)
            regions.append(LeafRegion(node_id, predicates, intervals))
            return

        feature_index = int(structure.feature[node_id])
        threshold = float(structure.threshold[node_id])
        boundary, left_inclusive = effective_input_boundary(threshold)
        feature_name = feature_names[feature_index]

        left_intervals = list(intervals)
        left_intervals[feature_index] = _tighten_upper(
            intervals[feature_index],
            boundary,
            inclusive=left_inclusive,
        )
        visit(
            left_id,
            (
                *predicates,
                PathPredicate(
                    node_id,
                    feature_index,
                    feature_name,
                    "<=" if left_inclusive else "<",
                    threshold,
                    boundary,
                    left_inclusive,
                ),
            ),
            tuple(left_intervals),
        )

        right_inclusive = not left_inclusive
        right_intervals = list(intervals)
        right_intervals[feature_index] = _tighten_lower(
            intervals[feature_index],
            boundary,
            inclusive=right_inclusive,
        )
        visit(
            right_id,
            (
                *predicates,
                PathPredicate(
                    node_id,
                    feature_index,
                    feature_name,
                    ">=" if right_inclusive else ">",
                    threshold,
                    boundary,
                    right_inclusive,
                ),
            ),
            tuple(right_intervals),
        )

    visit(0, (), initial)
    return tuple(sorted(regions, key=lambda region: region.leaf_id))


def _relative_volume(region: LeafRegion, domain: FloatArray) -> float:
    widths = np.asarray(
        [interval.width for interval in region.intervals],
        dtype=np.float64,
    )
    return float(np.prod(widths / (domain[:, 1] - domain[:, 0])))


def _validate_regions(
    tree: DecisionTreeRegressor,
    regions: tuple[LeafRegion, ...],
    fitting_features: FloatArray,
    domain_bounds: tuple[tuple[float, float], ...],
    feature_names: tuple[str, ...],
) -> tuple[float, ...]:
    structural_ids = {
        node_id
        for node_id, child in enumerate(tree.tree_.children_left)
        if child == TREE_LEAF
    }
    fitted_ids = {int(value) for value in tree.apply(fitting_features)}
    region_ids = {region.leaf_id for region in regions}
    if region_ids != structural_ids or region_ids != fitted_ids:
        msg = "regions must exactly match the structural and fitted leaves"
        raise ValueError(msg)

    domain = _checked_domain(domain_bounds, feature_names)
    volumes: list[float] = []
    for region in regions:
        if any(
            not np.isfinite(interval.lower)
            or not np.isfinite(interval.upper)
            or interval.width <= 0.0
            for interval in region.intervals
        ):
            msg = f"leaf {region.leaf_id} has a non-finite or empty box"
            raise ValueError(msg)
        actual_leaf = int(tree.apply(region.midpoint.reshape(1, -1))[0])
        if actual_leaf != region.leaf_id:
            msg = (
                f"leaf {region.leaf_id} midpoint routes to leaf {actual_leaf}"
            )
            raise ValueError(msg)
        volume = _relative_volume(region, domain)
        if not np.isfinite(volume) or volume <= 0.0:
            msg = f"leaf {region.leaf_id} has invalid relative volume"
            raise ValueError(msg)
        volumes.append(volume)

    total = float(sum(volumes))
    if not np.isclose(total, 1.0, rtol=0.0, atol=VOLUME_ATOL):
        msg = f"relative leaf volumes sum to {total:.17g}, not one"
        raise ValueError(msg)
    return tuple(volumes)


def _leaf_prototypes(
    tree: DecisionTreeRegressor,
    fitting_features: FloatArray,
    fitting_targets: FloatArray,
) -> dict[int, FloatArray]:
    assignments = np.asarray(tree.apply(fitting_features), dtype=np.int64)
    prototypes: dict[int, FloatArray] = {}
    for leaf_id in np.unique(assignments):
        members = fitting_targets[assignments == leaf_id]
        prototype = np.asarray(members.mean(axis=0), dtype=np.float64)
        prediction = np.asarray(
            tree.predict(fitting_features[assignments == leaf_id][:1])[0],
            dtype=np.float64,
        ).reshape(prototype.shape)
        tree_value = np.asarray(
            tree.tree_.value[int(leaf_id)],
            dtype=np.float64,
        ).reshape(prototype.shape)
        if not np.allclose(prototype, prediction, rtol=1e-12, atol=1e-12):
            msg = f"leaf {leaf_id} fitting mean differs from tree prediction"
            raise ValueError(msg)
        if not np.allclose(prototype, tree_value, rtol=1e-12, atol=1e-12):
            msg = f"leaf {leaf_id} fitting mean differs from tree value"
            raise ValueError(msg)
        prototypes[int(leaf_id)] = prototype
    return prototypes


def _residuals(targets: FloatArray, prototype: FloatArray) -> FloatArray:
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != MATRIX_DIMENSIONS or values.shape[1:] != prototype.shape:
        msg = "targets and prototype have incompatible shapes"
        raise ValueError(msg)
    residuals = np.sqrt(np.mean(np.square(values - prototype), axis=1))
    if residuals.size == 0 or not np.all(np.isfinite(residuals)):
        msg = "held residual distribution must be finite and nonempty"
        raise ValueError(msg)
    return np.asarray(residuals, dtype=np.float64)


def evaluate_path_representatives(
    prepared: PreparedSystem,
    tree: DecisionTreeRegressor | None = None,
) -> PathRepresentativeEvaluation:
    fitted_tree = tree
    if fitted_tree is None:
        fitted_tree = make_tree(EXPECTED_LEAF_COUNT).fit(
            prepared.fitting_features,
            prepared.fitting_targets,
        )
        if fitted_tree.get_n_leaves() != EXPECTED_LEAF_COUNT:
            msg = (
                f"expected {EXPECTED_LEAF_COUNT} fitted leaves, got "
                f"{fitted_tree.get_n_leaves()}"
            )
            raise ValueError(msg)

    regions = extract_leaf_regions(
        fitted_tree,
        prepared.spec.bounds,
        prepared.spec.feature_names,
    )
    volumes = _validate_regions(
        fitted_tree,
        regions,
        prepared.fitting_features,
        prepared.spec.bounds,
        prepared.spec.feature_names,
    )
    prototypes = _leaf_prototypes(
        fitted_tree,
        prepared.fitting_features,
        prepared.fitting_targets,
    )
    fitting_assignments = np.asarray(
        fitted_tree.apply(prepared.fitting_features),
        dtype=np.int64,
    )
    assessment_assignments = np.asarray(
        fitted_tree.apply(prepared.assessment_features),
        dtype=np.int64,
    )

    comparisons: list[LeafComparison] = []
    for region, volume in zip(regions, volumes, strict=True):
        leaf_id = region.leaf_id
        fitting_occupancy = int(
            np.count_nonzero(fitting_assignments == leaf_id),
        )
        assessment_members = prepared.assessment_targets[
            assessment_assignments == leaf_id
        ]
        if assessment_members.shape[0] == 0:
            msg = f"leaf {leaf_id} has no assessment members"
            raise ValueError(msg)

        prototype = prototypes[leaf_id]
        held_residuals = _residuals(assessment_members, prototype)
        raw_midpoint_target = prepared.spec.simulate_scenario(region.midpoint)
        midpoint_target = prepared.transform.transform(
            raw_midpoint_target.reshape(1, -1),
        )[0]
        midpoint_error = coordinate_rms(midpoint_target, prototype)
        comparisons.append(
            LeafComparison(
                region,
                prototype,
                midpoint_target,
                held_residuals,
                volume,
                fitting_occupancy,
                int(assessment_members.shape[0]),
                midpoint_error,
                float(np.count_nonzero(held_residuals <= midpoint_error))
                / held_residuals.size,
            ),
        )

    leaves = tuple(comparisons)
    return PathRepresentativeEvaluation(
        prepared,
        fitted_tree,
        leaves,
        float(
            sum(leaf.relative_volume * leaf.midpoint_error for leaf in leaves),
        ),
        float(
            sum(
                leaf.relative_volume * float(leaf.held_residuals.mean())
                for leaf in leaves
            ),
        ),
    )


def _residual_summary(values: FloatArray) -> dict[str, Any]:
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "count": int(values.size),
        "minimum": float(values.min()),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def _leaf_report(
    leaf: LeafComparison,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "leaf_id": leaf.region.leaf_id,
        "predicates": [
            {
                "node_id": predicate.node_id,
                "feature_index": predicate.feature_index,
                "feature": predicate.feature_name,
                "operator": predicate.operator,
                "inclusive": predicate.inclusive,
                "model_threshold": predicate.model_threshold,
                "effective_boundary": predicate.effective_boundary,
            }
            for predicate in leaf.region.predicates
        ],
        "box": [
            {
                "feature_index": index,
                "feature": name,
                "lower": interval.lower,
                "upper": interval.upper,
                "lower_inclusive": interval.lower_inclusive,
                "upper_inclusive": interval.upper_inclusive,
            }
            for index, (name, interval) in enumerate(
                zip(feature_names, leaf.region.intervals, strict=True),
            )
        ],
        "midpoint": [float(value) for value in leaf.region.midpoint],
        "relative_volume": leaf.relative_volume,
        "fitting_occupancy": leaf.fitting_occupancy,
        "assessment_occupancy": leaf.assessment_occupancy,
        "midpoint_to_prototype_error": leaf.midpoint_error,
        "held_residual_summary": _residual_summary(leaf.held_residuals),
        "midpoint_empirical_cdf_fraction": (
            leaf.midpoint_empirical_cdf_fraction
        ),
    }


def _system_report(evaluation: PathRepresentativeEvaluation) -> dict[str, Any]:
    prepared = evaluation.prepared
    spec = prepared.spec
    transform = prepared.transform
    return {
        "name": spec.name,
        "domain": [
            {"feature": name, "lower": lower, "upper": upper}
            for name, (lower, upper) in zip(
                spec.feature_names,
                spec.bounds,
                strict=True,
            )
        ],
        "state_names": list(spec.state_names),
        "sample_grid": {
            "start": spec.horizon[0],
            "stop": spec.horizon[1],
            "count": spec.sample_count,
        },
        "target_coordinates": {
            "total": int(transform.means.size),
            "retained": int(transform.retained.size),
            "excluded": int(transform.excluded.size),
        },
        "aggregate": {
            "volume_weighted_midpoint_error": (
                evaluation.volume_weighted_midpoint_error
            ),
            "volume_weighted_held_mean": (
                evaluation.volume_weighted_held_mean
            ),
        },
        "leaves": [
            _leaf_report(leaf, spec.feature_names)
            for leaf in evaluation.leaves
        ],
    }


def _shared_value(values: list[Any], description: str) -> Any:
    first = values[0]
    if any(value != first for value in values[1:]):
        msg = f"all study evaluations must share {description}"
        raise ValueError(msg)
    return first


def build_report(
    study: PathRepresentativeStudy,
    config: Config = CONFIG,
) -> dict[str, Any]:
    evaluations = study.evaluations
    fitting_count = _shared_value(
        [item.prepared.fitting_features.shape[0] for item in evaluations],
        "the fitting scenario count",
    )
    assessment_count = _shared_value(
        [item.prepared.assessment_features.shape[0] for item in evaluations],
        "the assessment scenario count",
    )
    leaf_count = _shared_value(
        [len(item.leaves) for item in evaluations],
        "the fitted leaf count",
    )
    tree_parameters = [item.tree.get_params() for item in evaluations]
    max_leaf_nodes = _shared_value(
        [parameters["max_leaf_nodes"] for parameters in tree_parameters],
        "max_leaf_nodes",
    )
    min_samples_leaf = _shared_value(
        [parameters["min_samples_leaf"] for parameters in tree_parameters],
        "min_samples_leaf",
    )
    random_state = _shared_value(
        [parameters["random_state"] for parameters in tree_parameters],
        "tree random_state",
    )
    cutoff = config.constant_scale_cutoff
    return {
        "experiment": "system_behavior_learning_path_representatives",
        "config": {
            "master_seed": config.master_seed,
            "scenario_counts": {
                "fitting": int(fitting_count),
                "assessment": int(assessment_count),
                "path_midpoints_per_system": int(leaf_count),
            },
            "tree": {
                "estimator": "sklearn.tree.DecisionTreeRegressor",
                "max_leaf_nodes": max_leaf_nodes,
                "fitted_leaf_count_per_system": int(leaf_count),
                "min_samples_leaf": min_samples_leaf,
                "random_state": random_state,
                "capacity_note": (
                    f"{max_leaf_nodes} maximum leaves are fixed to isolate "
                    "representative construction; this analysis does not "
                    "select capacity."
                ),
            },
            "scenario_distribution": (
                "independent uniform over each declared coordinate"
            ),
            "target": "complete retained standardized state trajectories",
            "target_transform": {
                "statistics_source": "fitting trajectories only",
                "standard_deviation_ddof": 0,
                "constant_coordinate_cutoff": cutoff,
            },
            "metric": (
                "RMS over equally weighted retained standardized trajectory "
                "coordinates"
            ),
            "weighting": (
                "Aggregates use exact relative path-box volumes under the "
                "independent-uniform declared domain."
            ),
            "float32_input_note": (
                "sklearn converts apply/predict inputs to float32; reported "
                "effective boundaries are float64 preimage boundaries with "
                "round-to-nearest-even equality ownership."
            ),
            "midpoint_empirical_cdf": (
                "count(held residual <= midpoint error) / held count"
            ),
        },
        "systems": [_system_report(item) for item in evaluations],
        "interpretation": (
            "These are descriptive results for one deterministic fitting and "
            "assessment split per system and one fixed tree capacity."
        ),
    }


def run_path_representatives(
    config: Config = CONFIG,
) -> tuple[dict[str, Any], PathRepresentativeStudy]:
    scenario_pairs = sample_all_scenarios(config)
    evaluations = tuple(
        evaluate_path_representatives(
            prepare_system(spec, fitting, assessment, config),
        )
        for spec, (fitting, assessment) in zip(
            SYSTEMS,
            scenario_pairs,
            strict=True,
        )
    )
    study = PathRepresentativeStudy(evaluations)
    return build_report(study, config), study
