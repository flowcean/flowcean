from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeRegressor

from flowcean.hybrid import HybridSystem, InputStream, simulate
from flowcean.hybrid.benchmarks import (
    bouncing_ball,
    hybrid_oscillator,
    registry,
    tank_valves,
    thermostat,
    thermostat_target_stream,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Simulator = Callable[[FloatArray, FloatArray], FloatArray]
CAPACITIES: tuple[int | None, ...] = (*range(2, 17), None)
MATRIX_DIMENSIONS = 2


@dataclass(frozen=True)
class Config:
    master_seed: int = 1
    fitting_scenarios: int = 256
    assessment_scenarios: int = 256
    constant_scale_cutoff: float = 1e-12


CONFIG = Config()


@dataclass(frozen=True)
class SystemSpec:
    name: str
    feature_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    horizon: tuple[float, float]
    sample_count: int
    state_names: tuple[str, ...]
    simulator: Simulator

    def __post_init__(self) -> None:
        if len(self.feature_names) != len(self.bounds):
            msg = f"feature/bound count mismatch for {self.name}"
            raise ValueError(msg)
        if not self.state_names:
            msg = f"no states configured for {self.name}"
            raise ValueError(msg)

    @property
    def state_count(self) -> int:
        return len(self.state_names)

    def sample_times(self) -> FloatArray:
        return np.linspace(*self.horizon, self.sample_count, dtype=np.float64)

    def simulate_scenario(
        self,
        scenario: FloatArray,
        sample_times: FloatArray | None = None,
    ) -> FloatArray:
        values = np.asarray(scenario, dtype=np.float64)
        if values.shape != (len(self.feature_names),) or not np.all(
            np.isfinite(values),
        ):
            msg = f"invalid scenario for {self.name}: {values.tolist()}"
            raise ValueError(msg)
        expected_times = self.sample_times()
        times = (
            expected_times
            if sample_times is None
            else np.asarray(sample_times, dtype=np.float64)
        )
        if not np.array_equal(times, expected_times):
            msg = f"invalid sample grid for {self.name}"
            raise ValueError(msg)
        return self.simulator(values, times)


def _complete_state(
    system: HybridSystem,
    sample_times: FloatArray,
    scenario: FloatArray,
    state_count: int,
    input_stream: InputStream | None = None,
) -> FloatArray:
    trace = simulate(
        system,
        t_span=(float(sample_times[0]), float(sample_times[-1])),
        input_stream=input_stream,
        sample_times=sample_times,
    )
    if not np.array_equal(trace.t, sample_times):
        msg = f"unexpected sample times for scenario {scenario.tolist()}"
        raise ValueError(msg)
    states = np.asarray(trace.x, dtype=np.float64)
    expected_shape = (sample_times.size, state_count)
    if states.shape != expected_shape or not np.all(np.isfinite(states)):
        msg = (
            f"invalid complete-state trace {states.shape} for scenario "
            f"{scenario.tolist()}"
        )
        raise ValueError(msg)
    return states.reshape(-1)


def _thermostat(scenario: FloatArray, times: FloatArray) -> FloatArray:
    system = thermostat(
        ambient=float(scenario[0]),
        heating_power=float(scenario[1]),
        cooling_rate=float(scenario[2]),
        hysteresis=float(scenario[3]),
        initial_state=np.array([scenario[4]], dtype=float),
    )
    return _complete_state(
        system,
        times,
        scenario,
        1,
        thermostat_target_stream,
    )


def _bouncing_ball(scenario: FloatArray, times: FloatArray) -> FloatArray:
    system = bouncing_ball(
        gravity=float(scenario[0]),
        restitution=float(scenario[1]),
        initial_state=np.array(scenario[2:4], dtype=float),
    )
    return _complete_state(system, times, scenario, 2)


def _hybrid_oscillator(scenario: FloatArray, times: FloatArray) -> FloatArray:
    system = hybrid_oscillator(
        damping_left=float(scenario[0]),
        damping_right=float(scenario[1]),
        frequency=float(scenario[2]),
        initial_state=np.array(scenario[3:5], dtype=float),
    )
    return _complete_state(system, times, scenario, 2)


def _tank_valves(scenario: FloatArray, times: FloatArray) -> FloatArray:
    system = tank_valves(
        area_1=1.0,
        area_2=1.2,
        inflow=float(scenario[0]),
        outflow=float(scenario[1]),
        outflow_1=0.2,
        valve_gain=float(scenario[2]),
        high_level=1.2,
        low_level=0.4,
        gravity=9.81,
        initial_state=np.array(scenario[3:5], dtype=float),
    )
    return _complete_state(system, times, scenario, 2)


def _horizon(name: str) -> tuple[float, float]:
    return registry()[name].t_span


THERMOSTAT = SystemSpec(
    "Thermostat",
    (
        "ambient",
        "heating_power",
        "cooling_rate",
        "hysteresis",
        "initial_temperature",
    ),
    (
        (17.0, 19.5),
        (3.5, 7.0),
        (0.15, 0.45),
        (1.0, 3.0),
        (18.0, 22.0),
    ),
    _horizon("Thermostat"),
    128,
    ("temperature",),
    _thermostat,
)
BOUNCING_BALL = SystemSpec(
    "Bouncing Ball",
    ("gravity", "restitution", "initial_height", "initial_velocity"),
    ((8.0, 12.0), (0.85, 0.95), (0.5, 2.0), (-1.0, 1.0)),
    _horizon("Bouncing Ball"),
    128,
    ("height", "velocity"),
    _bouncing_ball,
)
HYBRID_OSCILLATOR = SystemSpec(
    "Hybrid Oscillator",
    (
        "damping_left",
        "damping_right",
        "frequency",
        "initial_position",
        "initial_velocity",
    ),
    (
        (0.05, 0.4),
        (0.05, 0.4),
        (1.5, 2.5),
        (-1.5, -0.5),
        (-0.5, 0.5),
    ),
    _horizon("Hybrid Oscillator"),
    128,
    ("position", "velocity"),
    _hybrid_oscillator,
)
TANK_VALVES = SystemSpec(
    "Tank Valves",
    (
        "inflow",
        "outflow",
        "valve_gain",
        "initial_level_1",
        "initial_level_2",
    ),
    (
        (0.6, 1.0),
        (0.4, 0.8),
        (0.7, 1.3),
        (0.4, 0.8),
        (0.1, 0.4),
    ),
    _horizon("Tank Valves"),
    128,
    ("level_1", "level_2"),
    _tank_valves,
)
SYSTEMS = (THERMOSTAT, BOUNCING_BALL, HYBRID_OSCILLATOR, TANK_VALVES)


def _sample_pair(
    spec: SystemSpec,
    fitting_seed: np.random.SeedSequence,
    assessment_seed: np.random.SeedSequence,
    config: Config,
) -> tuple[FloatArray, FloatArray]:
    lower, upper = np.asarray(spec.bounds, dtype=np.float64).T
    fitting = np.random.default_rng(fitting_seed).uniform(
        lower,
        upper,
        size=(config.fitting_scenarios, len(spec.feature_names)),
    )
    assessment = np.random.default_rng(assessment_seed).uniform(
        lower,
        upper,
        size=(config.assessment_scenarios, len(spec.feature_names)),
    )
    if np.unique(fitting, axis=0).shape[0] != fitting.shape[0]:
        msg = f"duplicate fitting scenarios for {spec.name}"
        raise ValueError(msg)
    if np.unique(assessment, axis=0).shape[0] != assessment.shape[0]:
        msg = f"duplicate assessment scenarios for {spec.name}"
        raise ValueError(msg)
    if np.any(np.all(fitting[:, np.newaxis, :] == assessment, axis=2)):
        msg = f"fitting and assessment scenarios overlap for {spec.name}"
        raise ValueError(msg)
    return fitting, assessment


def sample_all_scenarios(
    config: Config = CONFIG,
) -> tuple[tuple[FloatArray, FloatArray], ...]:
    children = np.random.SeedSequence(config.master_seed).spawn(
        2 * len(SYSTEMS),
    )
    return tuple(
        _sample_pair(
            spec,
            children[2 * index],
            children[2 * index + 1],
            config,
        )
        for index, spec in enumerate(SYSTEMS)
    )


def sample_scenarios(
    spec: SystemSpec,
    config: Config = CONFIG,
) -> tuple[FloatArray, FloatArray]:
    try:
        index = SYSTEMS.index(spec)
    except ValueError as error:
        msg = f"unknown system: {spec.name}"
        raise ValueError(msg) from error
    children = np.random.SeedSequence(config.master_seed).spawn(
        2 * len(SYSTEMS),
    )
    return _sample_pair(
        spec,
        children[2 * index],
        children[2 * index + 1],
        config,
    )


def simulate_scenarios(spec: SystemSpec, scenarios: FloatArray) -> FloatArray:
    values = np.asarray(scenarios, dtype=np.float64)
    expected_columns = len(spec.feature_names)
    if values.ndim != MATRIX_DIMENSIONS or values.shape[1] != expected_columns:
        msg = f"invalid scenario matrix for {spec.name}: {values.shape}"
        raise ValueError(msg)
    trajectories: list[FloatArray] = []
    for index, scenario in enumerate(values):
        try:
            trajectories.append(spec.simulate_scenario(scenario))
        except Exception as error:
            msg = (
                f"{spec.name} simulation failed at index {index} for "
                f"{scenario.tolist()}: {error}"
            )
            raise RuntimeError(msg) from error
    return np.stack(trajectories)


@dataclass(frozen=True)
class TargetTransform:
    means: FloatArray
    scales: FloatArray
    retained: IntArray
    excluded: IntArray

    def transform(self, targets: FloatArray) -> FloatArray:
        values = np.asarray(targets, dtype=np.float64)
        if (
            values.ndim != MATRIX_DIMENSIONS
            or values.shape[1] != self.means.size
        ):
            msg = f"invalid target matrix shape {values.shape}"
            raise ValueError(msg)
        if not np.all(np.isfinite(values)):
            msg = "targets must be finite"
            raise ValueError(msg)
        return (
            values[:, self.retained] - self.means[self.retained]
        ) / self.scales[self.retained]


def fit_target_transform(
    fitting_targets: FloatArray,
    cutoff: float = CONFIG.constant_scale_cutoff,
) -> TargetTransform:
    values = np.asarray(fitting_targets, dtype=np.float64)
    if values.ndim != MATRIX_DIMENSIONS or not np.all(np.isfinite(values)):
        msg = "fitting targets must be a finite matrix"
        raise ValueError(msg)
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    retained = np.flatnonzero(scales > cutoff).astype(np.int64)
    excluded = np.flatnonzero(scales <= cutoff).astype(np.int64)
    if retained.size == 0:
        msg = "all target coordinates are constant"
        raise ValueError(msg)
    return TargetTransform(means, scales, retained, excluded)


def historical_mean_predictions(
    standardized_fitting_targets: FloatArray,
    assessment_count: int,
) -> FloatArray:
    mean = np.asarray(standardized_fitting_targets, dtype=np.float64).mean(
        axis=0,
    )
    return np.broadcast_to(mean, (assessment_count, mean.size)).copy()


def make_tree(max_leaf_nodes: int | None) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_leaf=16,
        max_leaf_nodes=max_leaf_nodes,
        ccp_alpha=0.0,
        random_state=0,
    )


def rmse(truth: FloatArray, prediction: FloatArray) -> float:
    actual = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if actual.shape != predicted.shape:
        msg = f"shape mismatch: {actual.shape} != {predicted.shape}"
        raise ValueError(msg)
    value = float(np.sqrt(np.mean(np.square(actual - predicted))))
    if not np.isfinite(value):
        msg = "non-finite RMSE"
        raise ValueError(msg)
    return value


@dataclass(frozen=True)
class PreparedSystem:
    spec: SystemSpec
    fitting_features: FloatArray
    assessment_features: FloatArray
    fitting_targets: FloatArray
    assessment_targets: FloatArray
    transform: TargetTransform


@dataclass(frozen=True)
class StateMetrics:
    state: str
    retained_coordinate_count: int
    tree_assessment_rmse: float | None
    historical_mean_assessment_rmse: float | None
    rmse_ratio: float | None
    normalized_squared_error_reduction: float | None


@dataclass(frozen=True)
class CapacityResult:
    system: str
    max_leaf_nodes: int | None
    fitted_leaf_count: int
    fitted_depth: int
    tree_fitting_rmse: float
    tree_assessment_rmse: float
    historical_mean_assessment_rmse: float
    assessment_rmse_ratio: float
    normalized_squared_error_reduction: float
    per_state_assessment_metrics: tuple[StateMetrics, ...]
    impurity_based_feature_importances: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class DescriptiveSummary:
    system: str
    lowest_observed_max_leaf_nodes: int | None
    lowest_observed_rmse_ratio: float
    saturation_max_leaf_nodes: int | None
    saturation_fitted_leaf_count: int


@dataclass(frozen=True)
class SystemEvaluation:
    prepared: PreparedSystem
    capacity_results: tuple[CapacityResult, ...]
    summary: DescriptiveSummary


def prepare_system(
    spec: SystemSpec,
    fitting_features: FloatArray,
    assessment_features: FloatArray,
    config: Config = CONFIG,
) -> PreparedSystem:
    fitting_trajectories = simulate_scenarios(spec, fitting_features)
    assessment_trajectories = simulate_scenarios(spec, assessment_features)
    transform = fit_target_transform(
        fitting_trajectories,
        config.constant_scale_cutoff,
    )
    return PreparedSystem(
        spec,
        fitting_features,
        assessment_features,
        transform.transform(fitting_trajectories),
        transform.transform(assessment_trajectories),
        transform,
    )


def _state_metrics(
    prepared: PreparedSystem,
    tree_prediction: FloatArray,
    mean_prediction: FloatArray,
) -> tuple[StateMetrics, ...]:
    metrics: list[StateMetrics] = []
    for state_index, state_name in enumerate(prepared.spec.state_names):
        columns = np.flatnonzero(
            prepared.transform.retained % prepared.spec.state_count
            == state_index,
        )
        if columns.size == 0:
            metrics.append(
                StateMetrics(state_name, 0, None, None, None, None),
            )
            continue
        truth = prepared.assessment_targets[:, columns]
        tree_error = rmse(truth, tree_prediction[:, columns])
        mean_error = rmse(truth, mean_prediction[:, columns])
        if mean_error == 0.0:
            msg = f"historical-mean RMSE is zero for {prepared.spec.name}"
            raise ValueError(msg)
        ratio = tree_error / mean_error
        metrics.append(
            StateMetrics(
                state_name,
                int(columns.size),
                tree_error,
                mean_error,
                ratio,
                1.0 - ratio**2,
            ),
        )
    return tuple(metrics)


def summarize_capacities(
    results: Sequence[CapacityResult],
) -> DescriptiveSummary:
    if not results:
        msg = "cannot summarize empty capacity results"
        raise ValueError(msg)
    system = results[0].system
    if any(result.system != system for result in results):
        msg = "capacity results must describe one system"
        raise ValueError(msg)
    unbounded = [result for result in results if result.max_leaf_nodes is None]
    if len(unbounded) != 1:
        msg = "capacity results require one unbounded tree"
        raise ValueError(msg)
    lowest = min(results, key=lambda result: result.assessment_rmse_ratio)
    unbounded_leaves = unbounded[0].fitted_leaf_count
    saturation = next(
        result
        for result in results
        if result.fitted_leaf_count == unbounded_leaves
    )
    return DescriptiveSummary(
        system,
        lowest.max_leaf_nodes,
        lowest.assessment_rmse_ratio,
        saturation.max_leaf_nodes,
        saturation.fitted_leaf_count,
    )


def evaluate_prepared_system(
    prepared: PreparedSystem,
    capacities: Sequence[int | None] = CAPACITIES,
) -> SystemEvaluation:
    mean_prediction = historical_mean_predictions(
        prepared.fitting_targets,
        prepared.assessment_targets.shape[0],
    )
    mean_error = rmse(prepared.assessment_targets, mean_prediction)
    if mean_error == 0.0:
        msg = f"historical-mean RMSE is zero for {prepared.spec.name}"
        raise ValueError(msg)
    results: list[CapacityResult] = []
    for capacity in capacities:
        tree = make_tree(capacity).fit(
            prepared.fitting_features,
            prepared.fitting_targets,
        )
        fitting_prediction = np.asarray(
            tree.predict(prepared.fitting_features),
            dtype=np.float64,
        ).reshape(prepared.fitting_targets.shape)
        assessment_prediction = np.asarray(
            tree.predict(prepared.assessment_features),
            dtype=np.float64,
        ).reshape(prepared.assessment_targets.shape)
        assessment_error = rmse(
            prepared.assessment_targets,
            assessment_prediction,
        )
        results.append(
            CapacityResult(
                prepared.spec.name,
                capacity,
                int(tree.get_n_leaves()),
                int(tree.get_depth()),
                rmse(prepared.fitting_targets, fitting_prediction),
                assessment_error,
                mean_error,
                assessment_error / mean_error,
                1.0 - (assessment_error / mean_error) ** 2,
                _state_metrics(
                    prepared,
                    assessment_prediction,
                    mean_prediction,
                ),
                tuple(
                    zip(
                        prepared.spec.feature_names,
                        (float(value) for value in tree.feature_importances_),
                        strict=True,
                    ),
                ),
            ),
        )
    frozen_results = tuple(results)
    return SystemEvaluation(
        prepared,
        frozen_results,
        summarize_capacities(frozen_results),
    )


def _target_metadata(
    spec: SystemSpec,
    transform: TargetTransform,
) -> dict[str, Any]:
    times = spec.sample_times()
    retained_by_state = {
        state: int(
            np.count_nonzero(transform.retained % spec.state_count == index),
        )
        for index, state in enumerate(spec.state_names)
    }
    excluded = []
    for flat_index in transform.excluded:
        time_index, state_index = divmod(int(flat_index), spec.state_count)
        excluded.append(
            {
                "flat_index": int(flat_index),
                "time_index": time_index,
                "sample_time": float(times[time_index]),
                "state_name": spec.state_names[state_index],
            },
        )
    return {
        "flattening_order": "time-major, state-minor",
        "total_coordinate_count": int(transform.means.size),
        "retained_coordinate_count": int(transform.retained.size),
        "excluded_coordinate_count": int(transform.excluded.size),
        "retained_coordinate_counts_by_state": retained_by_state,
        "excluded_coordinates": excluded,
    }


def _capacity_report(result: CapacityResult) -> dict[str, Any]:
    return {
        "max_leaf_nodes": result.max_leaf_nodes,
        "fitted_leaf_count": result.fitted_leaf_count,
        "fitted_depth": result.fitted_depth,
        "metrics": {
            "tree_fitting_rmse": result.tree_fitting_rmse,
            "tree_assessment_rmse": result.tree_assessment_rmse,
            "historical_mean_assessment_rmse": (
                result.historical_mean_assessment_rmse
            ),
            "assessment_rmse_ratio": result.assessment_rmse_ratio,
            "normalized_squared_error_reduction": (
                result.normalized_squared_error_reduction
            ),
            "per_state_assessment": [
                {
                    "state": metric.state,
                    "retained_coordinate_count": (
                        metric.retained_coordinate_count
                    ),
                    "tree_rmse": metric.tree_assessment_rmse,
                    "historical_mean_rmse": (
                        metric.historical_mean_assessment_rmse
                    ),
                    "rmse_ratio": metric.rmse_ratio,
                    "normalized_squared_error_reduction": (
                        metric.normalized_squared_error_reduction
                    ),
                }
                for metric in result.per_state_assessment_metrics
            ],
        },
        "model_diagnostics": {
            "impurity_based_feature_importances": dict(
                result.impurity_based_feature_importances,
            ),
            "caveat": (
                "Impurity-based feature importances are model diagnostics, "
                "not causal or stability measures."
            ),
        },
    }


def _system_report(evaluation: SystemEvaluation) -> dict[str, Any]:
    spec = evaluation.prepared.spec
    summary = evaluation.summary
    return {
        "name": spec.name,
        "feature_mapping": [
            {
                "index": index,
                "name": name,
                "domain": {"lower": lower, "upper": upper},
            }
            for index, (name, (lower, upper)) in enumerate(
                zip(spec.feature_names, spec.bounds, strict=True),
            )
        ],
        "horizon": {
            "start": spec.horizon[0],
            "end": spec.horizon[1],
            "source": "flowcean.hybrid.benchmarks.registry",
        },
        "sample_count": spec.sample_count,
        "state_mapping": [
            {"index": index, "name": name}
            for index, name in enumerate(spec.state_names)
        ],
        "target_metadata": _target_metadata(
            spec,
            evaluation.prepared.transform,
        ),
        "capacity_results": [
            _capacity_report(result) for result in evaluation.capacity_results
        ],
        "descriptive_summary": {
            "lowest_observed": {
                "max_leaf_nodes": summary.lowest_observed_max_leaf_nodes,
                "assessment_rmse_ratio": (summary.lowest_observed_rmse_ratio),
            },
            "saturation": {
                "max_leaf_nodes": summary.saturation_max_leaf_nodes,
                "fitted_leaf_count": summary.saturation_fitted_leaf_count,
                "definition": (
                    "First reported capacity with the unbounded tree's "
                    "fitted leaf count."
                ),
            },
        },
    }


def build_report(
    evaluations: Sequence[SystemEvaluation],
    config: Config = CONFIG,
) -> dict[str, Any]:
    if not evaluations:
        msg = "cannot report an empty experiment"
        raise ValueError(msg)
    capacity_grid = tuple(
        result.max_leaf_nodes for result in evaluations[0].capacity_results
    )
    if any(
        tuple(result.max_leaf_nodes for result in evaluation.capacity_results)
        != capacity_grid
        for evaluation in evaluations
    ):
        msg = "all systems must use the same capacity grid"
        raise ValueError(msg)

    return {
        "experiment": "system_behavior_learning",
        "config": {
            "master_seed": config.master_seed,
            "scenario_counts": {
                "fitting": config.fitting_scenarios,
                "assessment": config.assessment_scenarios,
            },
            "sample_counts_by_system": {
                evaluation.prepared.spec.name: (
                    evaluation.prepared.spec.sample_count
                )
                for evaluation in evaluations
            },
            "constant_scale_cutoff": config.constant_scale_cutoff,
            "capacity_grid": list(capacity_grid),
            "tree": {
                "estimator": "sklearn.tree.DecisionTreeRegressor",
                "criterion": "squared_error",
                "splitter": "best",
                "max_depth": None,
                "min_samples_leaf": 16,
                "ccp_alpha": 0.0,
                "random_state": 0,
                "max_leaf_nodes": "varied by capacity_grid",
            },
        },
        "seed_strategy": {
            "generator": "numpy.random.SeedSequence",
            "master_entropy": config.master_seed,
            "ordered_child_pairs": [
                {
                    "system": spec.name,
                    "fitting_child_index": 2 * index,
                    "assessment_child_index": 2 * index + 1,
                }
                for index, spec in enumerate(SYSTEMS)
            ],
            "independence": (
                "One ordered child per system and split; no child is reused."
            ),
        },
        "target_weighting": {
            "target": "complete trace.x state trajectory",
            "flattening_order": "time-major, state-minor",
            "standardization": (
                "Per-coordinate fitting mean and population standard "
                "deviation (ddof=0); assessment targets reuse that transform."
            ),
            "exclusion": (
                "Coordinates with fitting scale <= "
                f"{config.constant_scale_cutoff:g} are excluded."
            ),
            "aggregate_rmse": (
                "Every retained time-state coordinate and every assessment "
                "scenario has equal weight."
            ),
            "per_state_rmse": (
                "Each state uses only that state's retained time coordinates "
                "with equal assessment-scenario and coordinate weight."
            ),
            "ratio_reference": (
                "1.0 is the assessment RMSE of the fitting historical-mean "
                "predictor."
            ),
            "normalized_squared_error_reduction": (
                "One minus the squared RMSE ratio; positive values indicate "
                "less squared error than the historical-mean predictor."
            ),
        },
        "capacity_result_count": sum(
            len(evaluation.capacity_results) for evaluation in evaluations
        ),
        "systems": [_system_report(item) for item in evaluations],
        "interpretation": (
            "All capacity comparisons are descriptive observations on one "
            "fixed assessment split, not a model-selection procedure."
        ),
    }


def run_experiment(
    config: Config = CONFIG,
) -> tuple[dict[str, Any], tuple[SystemEvaluation, ...]]:
    scenario_pairs = sample_all_scenarios(config)
    evaluations = tuple(
        evaluate_prepared_system(
            prepare_system(spec, fitting, assessment, config),
        )
        for spec, (fitting, assessment) in zip(
            SYSTEMS,
            scenario_pairs,
            strict=True,
        )
    )
    return build_report(evaluations, config), evaluations
