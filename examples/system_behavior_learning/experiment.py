from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cache, partial
from multiprocessing import get_context
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from execution import (
    SimulationBatch,
    execute_simulations,
    validate_simulation_inputs,
)
from methods import (
    LeafBox,
    fit_behavior_tree,
    fit_input_only_tree,
    fit_unbounded_behavior_tree,
    leaf_boxes,
    pam_medoids,
    sample_unit_suite,
    scale_to_bounds,
    tree_midpoints,
)
from metrics import (
    PredictionMetrics,
    RealizationMetrics,
    available_realization_metrics,
    pairwise_coordinate_rms,
    prediction_metrics,
    reference_coverage_metrics,
)
from numpy.typing import NDArray
from settings import Settings, SystemSettings
from tqdm import tqdm

from flowcean.hybrid import HybridSystem, InputStream, Trace, simulate
from flowcean.hybrid.benchmarks import (
    bouncing_ball,
    hybrid_oscillator,
    pid_controlled_plant,
    tank_valves,
    thermostat,
    thermostat_target_stream,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.tree import DecisionTreeRegressor

FloatArray = NDArray[np.float64]
Scenario = Mapping[str, float]
Simulator = Callable[[SystemSettings, Scenario, FloatArray], FloatArray]
StageCallback = Callable[[str], None]
SimulationProgress = Callable[[int], object]
MATRIX_DIMENSIONS = 2
MIN_TRAJECTORY_SAMPLE_COUNT = 2


@dataclass(frozen=True)
class SystemSpec:
    settings: SystemSettings
    simulator: Simulator

    @property
    def name(self) -> str:
        return self.settings.name

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.settings.feature_names

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return self.settings.bounds

    @property
    def horizon(self) -> tuple[float, float]:
        return self.settings.horizon

    @property
    def state_names(self) -> tuple[str, ...]:
        return self.settings.state_names

    def sample_times(self, sample_count: int) -> FloatArray:
        if sample_count < MIN_TRAJECTORY_SAMPLE_COUNT:
            message = "trajectory sample count must be at least two"
            raise ValueError(message)
        return np.linspace(*self.horizon, sample_count, dtype=np.float64)

    def simulate_scenario(
        self,
        scenario: FloatArray,
        sample_count: int,
    ) -> FloatArray:
        values = np.asarray(scenario, dtype=np.float64)
        if values.shape != (len(self.feature_names),) or not np.all(
            np.isfinite(values),
        ):
            message = f"invalid scenario for {self.name}: {values.tolist()}"
            raise ValueError(message)
        named_scenario = dict(zip(self.feature_names, values, strict=True))
        return self.simulator(
            self.settings,
            named_scenario,
            self.sample_times(sample_count),
        )


BatchSimulator = Callable[
    [SystemSpec, FloatArray, int, SimulationProgress | None],
    SimulationBatch,
]


@dataclass(frozen=True)
class TargetTransform:
    means: FloatArray
    scales: FloatArray
    retained: NDArray[np.int64]

    def transform(self, targets: FloatArray) -> FloatArray:
        values = np.asarray(targets, dtype=np.float64)
        if (
            values.ndim != MATRIX_DIMENSIONS
            or values.shape[1] != self.means.size
        ):
            message = f"target matrix has invalid shape {values.shape}"
            raise ValueError(message)
        if not np.all(np.isfinite(values)):
            message = "targets must be finite"
            raise ValueError(message)
        return np.asarray(
            (values[:, self.retained] - self.means[self.retained])
            / self.scales[self.retained],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class TreeMetricRow:
    system: str
    replicate: int
    capacity: int
    realized_leaves: int
    depth: int
    fitting_rmse: float
    assessment_rmse: float | None
    historical_mean_rmse: float | None
    prediction_ratio: float | None
    midpoint_realization_distance: float | None
    held_leaf_distance: float | None
    realization_ratio: float | None
    assessment_unrepresented_volume: float | None


@dataclass(frozen=True)
class UnboundedTreeMetricRow:
    system: str
    replicate: int
    realized_leaves: int
    depth: int
    singleton_fraction: float
    fitting_rmse: float
    assessment_rmse: float | None
    historical_mean_rmse: float | None
    prediction_ratio: float | None
    midpoint_realization_distance: float | None
    held_leaf_distance: float | None
    realization_ratio: float | None
    assessment_unrepresented_volume: float | None


@dataclass(frozen=True)
class LeafMetricRow:
    system: str
    replicate: int
    capacity: int
    leaf_id: int
    depth: int
    relative_volume: float
    fitting_members: int
    assessment_members: int | None
    midpoint_realization_distance: float | None
    held_leaf_distance: float | None
    realization_ratio: float | None


@dataclass(frozen=True)
class UnboundedLeafMetricRow:
    system: str
    replicate: int
    leaf_id: int
    depth: int
    relative_volume: float
    fitting_members: int
    assessment_members: int | None
    midpoint_realization_distance: float | None
    held_leaf_distance: float | None
    realization_ratio: float | None


@dataclass(frozen=True)
class LeafBoxRow:
    system: str
    replicate: int
    capacity: int
    leaf_id: int
    parameter: str
    lower: float
    upper: float
    lower_inclusive: bool
    upper_inclusive: bool
    midpoint: float


@dataclass(frozen=True)
class UnboundedLeafBoxRow:
    system: str
    replicate: int
    leaf_id: int
    parameter: str
    lower: float
    upper: float
    lower_inclusive: bool
    upper_inclusive: bool
    midpoint: float


@dataclass(frozen=True)
class SuiteMetricRow:
    system: str
    replicate: int
    capacity: int
    suite_size: int
    method: str
    method_repetition: int | None
    mean_distance: float
    q95_distance: float
    q99_distance: float
    max_distance: float
    used_members: int
    mean_input_spacing: float
    scaled_input_spacing: float


@dataclass(frozen=True)
class SuiteStatusRow:
    system: str
    replicate: int
    capacity: int
    method: str
    method_repetition: int | None
    intended_size: int | None
    actual_size: int
    status: str
    attempted_count: int = 0
    successful_count: int = 0
    unique_scenario_count: int = 0
    unique_valid_trajectory_count: int = 0
    reused_count: int = 0
    reason: str = ""


@dataclass(frozen=True)
class ExecutionStatusRow:
    system: str
    replicate: int | None
    capacity: int | None
    stage: str
    status: str
    generated_count: int = 0
    attempted_count: int = 0
    successful_count: int = 0
    reason: str = ""


@dataclass(frozen=True)
class PrototypePlotData:
    system: str
    replicate: int
    capacity: int
    state_names: tuple[str, ...]
    sample_times: FloatArray
    leaf_ids: NDArray[np.int64]
    relative_volumes: FloatArray
    fitting_assignments: NDArray[np.int64]
    fitting_trajectories: FloatArray
    leaf_prototypes: FloatArray
    midpoint_trajectories: FloatArray


@dataclass(frozen=True)
class ExperimentRecords:
    tree_metrics: tuple[TreeMetricRow, ...]
    leaf_metrics: tuple[LeafMetricRow, ...]
    leaf_boxes: tuple[LeafBoxRow, ...]
    unbounded_tree_metrics: tuple[UnboundedTreeMetricRow, ...]
    unbounded_leaf_metrics: tuple[UnboundedLeafMetricRow, ...]
    unbounded_leaf_boxes: tuple[UnboundedLeafBoxRow, ...]
    suite_metrics: tuple[SuiteMetricRow, ...]
    suite_statuses: tuple[SuiteStatusRow, ...]
    prototype_plots: tuple[PrototypePlotData, ...]
    execution_statuses: tuple[ExecutionStatusRow, ...] = ()


@dataclass(frozen=True)
class GeometricSuite:
    capacity: int
    method: str
    repetition: int
    batch: SimulationBatch

    @property
    def scenarios(self) -> FloatArray:
        return self.batch.scenarios

    @property
    def trajectories(self) -> FloatArray:
        return self.batch.trajectories


@dataclass(frozen=True)
class _PreparedReplicate:
    fitting_scenarios: FloatArray
    assessment_scenarios: FloatArray
    fitting_trajectories: FloatArray
    assessment_trajectories: FloatArray
    transform: TargetTransform
    fitting_targets: FloatArray
    assessment_targets: FloatArray | None
    reference_targets: FloatArray | None
    archive_distances: FloatArray


@dataclass(frozen=True)
class _TreeDiagnostics:
    boxes: tuple[LeafBox, ...]
    midpoint_scenarios: FloatArray
    midpoint_batch: SimulationBatch
    prediction: PredictionMetrics
    realization: RealizationMetrics
    fitting_assignments: NDArray[np.int64]


@dataclass(frozen=True)
class _UnboundedEvaluation:
    tree_row: UnboundedTreeMetricRow
    leaf_rows: tuple[UnboundedLeafMetricRow, ...]
    box_rows: tuple[UnboundedLeafBoxRow, ...]
    tree_state: dict[str, NDArray[Any]]


@dataclass(frozen=True)
class _CapacityEvaluation:
    tree_row: TreeMetricRow
    leaf_rows: tuple[LeafMetricRow, ...]
    box_rows: tuple[LeafBoxRow, ...]
    suite_rows: tuple[SuiteMetricRow, ...]
    status_rows: tuple[SuiteStatusRow, ...]
    prototype_plot: PrototypePlotData | None
    tree_state: dict[str, NDArray[Any]]
    suite_arrays: dict[str, NDArray[Any]]


def pending_suite_summaries(
    records: ExperimentRecords,
) -> frozenset[tuple[str, int, str]]:
    """Withhold geometric summaries with incomplete planned evidence."""
    invalid = {
        (row.system, row.capacity, row.method)
        for row in records.suite_statuses
        if row.method in ("sobol", "random", "lhs") and row.status != "valid"
    }
    measured = {
        (row.system, row.capacity, row.method) for row in records.suite_metrics
    }
    return frozenset(invalid & measured)


def suite_summary_rows(
    records: ExperimentRecords,
) -> tuple[SuiteMetricRow, ...]:
    """Withhold incomplete groups, preserving every individual measurement."""
    pending = pending_suite_summaries(records)
    return tuple(
        row
        for row in records.suite_metrics
        if (row.system, row.capacity, row.method) not in pending
    )


def _geometric_suite_loader(
    spec: SystemSpec,
    settings: Settings,
    *,
    batch_simulator: BatchSimulator,
) -> tuple[
    Callable[[str, int, int], GeometricSuite],
    dict[tuple[str, int, int], GeometricSuite],
]:
    bounds = np.asarray(spec.bounds, dtype=np.float64)
    loaded: dict[tuple[str, int, int], GeometricSuite] = {}

    @cache
    def load(method: str, budget: int, repetition: int) -> GeometricSuite:
        unit = sample_unit_suite(
            method,
            len(spec.feature_names),
            budget,
            settings.geometric_seed_sequence(
                method,
                spec.name,
                capacity=budget,
                repetition=repetition,
            ),
        )
        scenarios = scale_to_bounds(unit, bounds)
        suite = GeometricSuite(
            capacity=budget,
            method=method,
            repetition=repetition,
            batch=batch_simulator(
                spec,
                scenarios,
                settings.trajectory_samples,
                None,
            ),
        )
        loaded[(method, budget, repetition)] = suite
        return suite

    return load, loaded


def _flatten_complete_state(
    trace: Trace,
    sample_times: FloatArray,
    scenario: Scenario,
    state_count: int,
) -> FloatArray:
    if not np.array_equal(trace.t, sample_times):
        message = f"unexpected sample times for scenario {dict(scenario)}"
        raise ValueError(message)
    states = np.asarray(trace.x, dtype=np.float64)
    expected = (sample_times.size, state_count)
    if states.shape != expected or not np.all(np.isfinite(states)):
        message = (
            f"invalid complete-state trace {states.shape} for scenario "
            f"{dict(scenario)}"
        )
        raise ValueError(message)
    return states.reshape(-1)


def _complete_state(
    system: HybridSystem,
    sample_times: FloatArray,
    scenario: Scenario,
    state_count: int,
    input_stream: InputStream | None = None,
) -> FloatArray:
    trace = simulate(
        system,
        t_span=(float(sample_times[0]), float(sample_times[-1])),
        input_stream=input_stream,
        sample_times=sample_times,
    )
    return _flatten_complete_state(
        trace,
        sample_times,
        scenario,
        state_count,
    )


def _thermostat(
    settings: SystemSettings,
    scenario: Scenario,
    times: FloatArray,
) -> FloatArray:
    system = thermostat(
        ambient=scenario["ambient"],
        heating_power=scenario["heating_power"],
        cooling_rate=scenario["cooling_rate"],
        hysteresis=scenario["hysteresis"],
        initial_state=np.array(
            [scenario["initial_temperature"]],
            dtype=float,
        ),
    )
    return _complete_state(
        system,
        times,
        scenario,
        len(settings.state_names),
        thermostat_target_stream,
    )


def _bouncing_ball(
    settings: SystemSettings,
    scenario: Scenario,
    times: FloatArray,
) -> FloatArray:
    system = bouncing_ball(
        gravity=scenario["gravity"],
        restitution=scenario["restitution"],
        initial_state=np.array(
            [scenario["initial_height"], scenario["initial_velocity"]],
            dtype=float,
        ),
    )
    return _complete_state(
        system,
        times,
        scenario,
        len(settings.state_names),
    )


def _hybrid_oscillator(
    settings: SystemSettings,
    scenario: Scenario,
    times: FloatArray,
) -> FloatArray:
    system = hybrid_oscillator(
        damping_left=scenario["damping_left"],
        damping_right=scenario["damping_right"],
        frequency=scenario["frequency"],
        initial_state=np.array(
            [scenario["initial_position"], scenario["initial_velocity"]],
            dtype=float,
        ),
    )
    return _complete_state(
        system,
        times,
        scenario,
        len(settings.state_names),
    )


def _pid_controlled_plant(
    settings: SystemSettings,
    scenario: Scenario,
    times: FloatArray,
) -> FloatArray:
    actuator_limit = scenario["actuator_limit"]
    system = pid_controlled_plant(
        kp=scenario["kp"],
        ki=scenario["ki"],
        kd=scenario["kd"],
        stiffness=scenario["stiffness"],
        damping=scenario["damping"],
        setpoint_amp=scenario["setpoint_amp"],
        setpoint_freq=scenario["setpoint_freq"],
        u_min=-actuator_limit,
        u_max=actuator_limit,
        initial_state=np.array(
            [
                scenario["initial_position"],
                scenario["initial_velocity"],
                scenario["initial_integral"],
            ],
            dtype=float,
        ),
    )
    return _complete_state(
        system,
        times,
        scenario,
        len(settings.state_names),
    )


def _tank_valves(
    settings: SystemSettings,
    scenario: Scenario,
    times: FloatArray,
) -> FloatArray:
    fixed = settings.fixed_parameter_values
    simulation_rtol = fixed["simulation_rtol"]
    simulation_atol = fixed["simulation_atol"]
    level_tolerance = fixed["level_tolerance"]
    wall_height = fixed["wall_height"]
    if simulation_rtol <= 0.0 or simulation_atol <= 0.0:
        message = "Tank Valves solver tolerances must be strictly positive"
        raise ValueError(message)
    if level_tolerance < 0.0 or wall_height <= 0.0:
        message = (
            "Tank Valves level_tolerance must be nonnegative and "
            "wall_height must be strictly positive"
        )
        raise ValueError(message)

    system = tank_valves(
        area_1=fixed["area_1"],
        area_2=fixed["area_2"],
        inflow=scenario["inflow"],
        outlet_area_1=scenario["outlet_area_1"],
        outlet_area_2=scenario["outlet_area_2"],
        valve_gain=scenario["valve_gain"],
        high_level=scenario["high_level"],
        low_level=scenario["low_level"],
        gravity=fixed["gravity"],
        initial_state=np.array(
            [scenario["initial_level_1"], scenario["initial_level_2"]],
            dtype=float,
        ),
    )
    trace = simulate(
        system,
        t_span=(float(times[0]), float(times[-1])),
        sample_times=times,
        rtol=simulation_rtol,
        atol=simulation_atol,
    )
    flattened = _flatten_complete_state(
        trace,
        times,
        scenario,
        len(settings.state_names),
    )
    levels = flattened.reshape(times.size, len(settings.state_names))
    if np.any(levels < -level_tolerance) or np.any(
        levels > wall_height + level_tolerance,
    ):
        message = (
            f"Tank Valves levels violate configured bounds: {dict(scenario)}"
        )
        raise ValueError(message)
    return flattened


SIMULATORS: dict[str, Simulator] = {
    "Thermostat": _thermostat,
    "Bouncing Ball": _bouncing_ball,
    "Hybrid Oscillator": _hybrid_oscillator,
    "Tank Valves": _tank_valves,
    "PID-Controlled Plant": _pid_controlled_plant,
}


def build_system_specs(
    systems: Sequence[SystemSettings],
) -> tuple[SystemSpec, ...]:
    missing = [
        system.name for system in systems if system.name not in SIMULATORS
    ]
    if missing:
        message = f"no simulator is available for systems: {missing}"
        raise ValueError(message)
    return tuple(
        SystemSpec(settings=system, simulator=SIMULATORS[system.name])
        for system in systems
    )


def sample_scenarios(
    spec: SystemSpec,
    count: int,
    seed: np.random.SeedSequence,
) -> FloatArray:
    if count < 1:
        message = "scenario count must be positive"
        raise ValueError(message)
    bounds = np.asarray(spec.bounds, dtype=np.float64)
    return np.asarray(
        np.random.default_rng(seed).uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(count, len(spec.feature_names)),
        ),
        dtype=np.float64,
    )


def sample_replicate(
    spec: SystemSpec,
    settings: Settings,
    replicate: int,
    reference: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    fitting = sample_scenarios(
        spec,
        settings.fitting_size,
        settings.seed_sequence("fitting", spec.name, replicate=replicate),
    )
    assessment = sample_scenarios(
        spec,
        settings.assessment_size,
        settings.seed_sequence("assessment", spec.name, replicate=replicate),
    )
    _require_disjoint(spec.name, fitting, assessment, reference)
    return fitting, assessment


def _require_disjoint(system: str, *samples: FloatArray) -> None:
    row_sets: list[set[bytes]] = []
    for sample in samples:
        if np.unique(sample, axis=0).shape[0] != sample.shape[0]:
            message = f"{system} sample contains duplicate scenarios"
            raise ValueError(message)
        row_sets.append({row.tobytes() for row in sample})
    if any(
        row_sets[i] & row_sets[j]
        for i in range(len(samples))
        for j in range(i + 1, len(samples))
    ):
        message = (
            f"{system} fitting, assessment, and reference samples overlap"
        )
        raise ValueError(message)


def simulate_scenarios(
    spec: SystemSpec,
    scenarios: FloatArray,
    sample_count: int,
    progress: SimulationProgress | None = None,
) -> FloatArray:
    return simulate_scenario_batch(
        spec,
        scenarios,
        sample_count,
        progress,
    ).require_complete(spec.name)


def simulate_scenario_batch(
    spec: SystemSpec,
    scenarios: FloatArray,
    sample_count: int,
    progress: SimulationProgress | None = None,
) -> SimulationBatch:
    values = _validate_scenario_batch(spec, scenarios, sample_count)
    return execute_simulations(
        values,
        lambda scenario: spec.simulate_scenario(scenario, sample_count),
        sample_count * len(spec.state_names),
        progress,
    )


def _validate_scenario_batch(
    spec: SystemSpec,
    scenarios: FloatArray,
    sample_count: int,
) -> FloatArray:
    values = np.asarray(scenarios, dtype=np.float64)
    if values.ndim != MATRIX_DIMENSIONS or values.shape[1] != len(
        spec.feature_names,
    ):
        message = f"invalid scenario matrix for {spec.name}: {values.shape}"
        raise ValueError(message)
    # Reject global sampling errors before entering the per-row boundary.
    spec.sample_times(sample_count)
    return values


def _simulate_scenario_chunk(
    spec: SystemSpec,
    scenarios: FloatArray,
    sample_count: int,
) -> SimulationBatch:
    return simulate_scenario_batch(spec, scenarios, sample_count, None)


def _simulate_scenario_batch_parallel(
    spec: SystemSpec,
    scenarios: FloatArray,
    sample_count: int,
    progress: SimulationProgress | None = None,
    *,
    executor: ProcessPoolExecutor,
    workers: int,
) -> SimulationBatch:
    """Simulate contiguous chunks, restoring original evidence order.

    Only the parent reports progress, once per row in a completed chunk.
    The executor belongs to the run and is reused across all its batches.
    """
    values = _validate_scenario_batch(spec, scenarios, sample_count)
    values = validate_simulation_inputs(
        values,
        lambda scenario: spec.simulate_scenario(scenario, sample_count),
        sample_count * len(spec.state_names),
        progress,
    )
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 1
    ):
        message = "workers must be a positive integer"
        raise ValueError(message)
    chunks = np.array_split(values, min(len(values), 4 * workers))
    futures = {
        executor.submit(
            _simulate_scenario_chunk,
            spec,
            chunk,
            sample_count,
        ): index
        for index, chunk in enumerate(chunks)
    }
    completed: dict[int, SimulationBatch] = {}
    for future in as_completed(futures):
        batch = future.result()
        completed[futures[future]] = batch
        if progress is not None:
            for _ in range(batch.attempted_count):
                progress(1)
    ordered = [completed[index] for index in range(len(chunks))]
    return SimulationBatch(
        np.concatenate([batch.scenarios for batch in ordered]),
        np.concatenate([batch.trajectories for batch in ordered]),
        np.concatenate([batch.valid for batch in ordered]),
        tuple(error for batch in ordered for error in batch.errors),
    )


def fit_target_transform(
    fitting_targets: FloatArray,
    cutoff: float = 1e-12,
) -> TargetTransform:
    values = np.asarray(fitting_targets, dtype=np.float64)
    if values.ndim != MATRIX_DIMENSIONS or not np.all(np.isfinite(values)):
        message = "fitting targets must be a finite matrix"
        raise ValueError(message)
    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(scales)):
        message = "fitting transform means and scales must be finite"
        raise ValueError(message)
    retained = np.flatnonzero(scales > cutoff).astype(np.int64)
    if retained.size == 0:
        message = "all fitting trajectory coordinates are constant"
        raise ValueError(message)
    return TargetTransform(
        means=means,
        scales=scales,
        retained=retained,
    )


def _suite_row(
    spec: SystemSpec,
    settings: Settings,
    transform: TargetTransform,
    reference_targets: FloatArray,
    scenarios: FloatArray,
    *,
    replicate: int,
    capacity: int,
    method: str,
    repetition: int | None,
    raw_arrays: dict[str, NDArray[Any]],
    raw_targets: FloatArray | None = None,
) -> SuiteMetricRow:
    raw = (
        simulate_scenarios(spec, scenarios, settings.trajectory_samples)
        if raw_targets is None
        else raw_targets
    )
    metric = reference_coverage_metrics(
        transform.transform(raw),
        reference_targets,
        scenarios,
        np.asarray(spec.bounds, dtype=np.float64),
    )
    prefix = (
        f"suite_capacity_{capacity:03d}__{method}"
        if repetition is None
        else (
            f"suite_{method}__budget_{len(scenarios):03d}"
            f"__repetition_{repetition:03d}"
        )
    )
    raw_arrays[f"{prefix}__reference_assignments"] = (
        metric.reference_assignments
    )
    raw_arrays[f"{prefix}__reference_distances"] = metric.reference_distances
    return SuiteMetricRow(
        system=spec.name,
        replicate=replicate,
        capacity=capacity,
        suite_size=len(scenarios),
        method=method,
        method_repetition=repetition,
        mean_distance=metric.mean_distance,
        q95_distance=metric.q95_distance,
        q99_distance=metric.q99_distance,
        max_distance=metric.max_distance,
        used_members=metric.used_members,
        mean_input_spacing=metric.mean_input_spacing,
        scaled_input_spacing=metric.scaled_input_spacing,
    )


def _batch_arrays(
    batch: SimulationBatch,
    prefix: str,
    separator: str = "__",
) -> dict[str, NDArray[Any]]:
    return {
        f"{prefix}{separator}scenarios": batch.scenarios,
        f"{prefix}{separator}trajectories": batch.trajectories,
        f"{prefix}{separator}valid": batch.valid,
        f"{prefix}{separator}errors": np.asarray(batch.errors, dtype=np.str_),
    }


def _batch_status(
    spec: SystemSpec,
    replicate: int | None,
    capacity: int | None,
    stage: str,
    batch: SimulationBatch,
) -> ExecutionStatusRow:
    complete = batch.successful_count == batch.generated_count
    return ExecutionStatusRow(
        spec.name,
        replicate,
        capacity,
        stage,
        "valid" if complete else "invalid_execution",
        batch.generated_count,
        batch.attempted_count,
        batch.successful_count,
        "" if complete else "complete batch required; see NPZ errors",
    )


def _blocked_slots(
    spec: SystemSpec,
    settings: Settings,
    replicate: int,
    capacity: int,
    reason: str,
) -> tuple[SuiteStatusRow, ...]:
    slots: list[tuple[str, int | None]] = [
        (method, None)
        for method in (
            "behavior_midpoint",
            "input_only_midpoint",
            "archive_pam",
        )
    ]
    slots.extend(
        (method, repetition)
        for method in ("sobol", "random", "lhs")
        for repetition in range(settings.geometric_repetitions)
    )
    return tuple(
        SuiteStatusRow(
            spec.name,
            replicate,
            capacity,
            method,
            repetition,
            None,
            0,
            "dependency_blocked",
            reason=reason,
        )
        for method, repetition in slots
    )


def _score_suite(
    spec: SystemSpec,
    settings: Settings,
    prepared: _PreparedReplicate,
    batch: SimulationBatch,
    *,
    replicate: int,
    capacity: int,
    budget: int,
    method: str,
    repetition: int | None,
    raw_arrays: dict[str, NDArray[Any]],
    reused: bool = False,
) -> tuple[SuiteMetricRow | None, SuiteStatusRow]:
    status, reason = "valid", ""
    if batch.generated_count != budget:
        status, reason = (
            "structurally_infeasible",
            "generated size differs from realized budget",
        )
    elif batch.unique_scenario_count != budget:
        status, reason = (
            "duplicate_scenarios",
            "exact unique scenario count differs from budget",
        )
    elif batch.successful_count != budget:
        status, reason = (
            "invalid_execution",
            "complete suite execution required; see NPZ errors",
        )
    elif prepared.reference_targets is None:
        status, reason = (
            "dependency_blocked",
            "complete reference batch required",
        )
    row = None
    if status == "valid" and prepared.reference_targets is not None:
        row = _suite_row(
            spec,
            settings,
            prepared.transform,
            prepared.reference_targets,
            batch.scenarios,
            replicate=replicate,
            capacity=capacity,
            method=method,
            repetition=repetition,
            raw_arrays=raw_arrays,
            raw_targets=batch.trajectories,
        )
    return row, SuiteStatusRow(
        spec.name,
        replicate,
        capacity,
        method,
        repetition,
        budget,
        batch.generated_count,
        status,
        0 if reused else batch.attempted_count,
        batch.successful_count,
        batch.unique_scenario_count,
        batch.unique_valid_trajectory_count,
        batch.generated_count if reused else 0,
        reason,
    )


def _configured_systems(
    settings: Settings,
    systems: Sequence[SystemSpec] | None,
) -> tuple[SystemSpec, ...]:
    if systems is None:
        return build_system_specs(settings.systems)
    available = {spec.name: spec for spec in systems}
    missing = [
        system.name
        for system in settings.systems
        if system.name not in available
    ]
    if missing:
        message = f"settings refer to unavailable systems: {missing}"
        raise ValueError(message)
    mismatched = [
        system.name
        for system in settings.systems
        if available[system.name].settings != system
    ]
    if mismatched:
        message = f"runtime systems do not match settings: {mismatched}"
        raise ValueError(message)
    return tuple(available[system.name] for system in settings.systems)


def run_experiment(
    settings: Settings,
    systems: Sequence[SystemSpec] | None = None,
    *,
    show_progress: bool = False,
    raw_output_dir: Path | None = None,
) -> ExperimentRecords:
    configured_systems = _configured_systems(settings, systems)
    with (
        ProcessPoolExecutor(
            max_workers=settings.workers,
            mp_context=get_context("spawn"),
        )
        if settings.workers > 1
        else nullcontext()
    ) as executor:
        batch_simulator = (
            simulate_scenario_batch
            if executor is None
            else partial(
                _simulate_scenario_batch_parallel,
                executor=executor,
                workers=settings.workers,
            )
        )
        system_records = tuple(
            _run_system(
                spec,
                settings,
                progress_position=0 if show_progress else None,
                raw_output_dir=raw_output_dir,
                raw_system_index=index,
                batch_simulator=batch_simulator,
            )
            for index, spec in enumerate(configured_systems)
        )
    return ExperimentRecords(
        tree_metrics=tuple(
            row for records in system_records for row in records.tree_metrics
        ),
        leaf_metrics=tuple(
            row for records in system_records for row in records.leaf_metrics
        ),
        leaf_boxes=tuple(
            row for records in system_records for row in records.leaf_boxes
        ),
        unbounded_tree_metrics=tuple(
            row
            for records in system_records
            for row in records.unbounded_tree_metrics
        ),
        unbounded_leaf_metrics=tuple(
            row
            for records in system_records
            for row in records.unbounded_leaf_metrics
        ),
        unbounded_leaf_boxes=tuple(
            row
            for records in system_records
            for row in records.unbounded_leaf_boxes
        ),
        suite_metrics=tuple(
            row for records in system_records for row in records.suite_metrics
        ),
        suite_statuses=tuple(
            row for records in system_records for row in records.suite_statuses
        ),
        execution_statuses=tuple(
            row
            for records in system_records
            for row in records.execution_statuses
        ),
        prototype_plots=tuple(
            plot
            for records in system_records
            for plot in records.prototype_plots
        ),
    )


def _run_system(
    spec: SystemSpec,
    settings: Settings,
    progress_position: int | None,
    raw_output_dir: Path | None,
    raw_system_index: int,
    *,
    batch_simulator: BatchSimulator,
) -> ExperimentRecords:
    progress_bar = (
        tqdm(
            total=settings.replicates,
            desc=spec.name,
            position=progress_position,
            leave=True,
            unit="replicate",
            dynamic_ncols=True,
            disable=None,
        )
        if progress_position is not None
        else None
    )
    capacity_bar = (
        tqdm(
            total=len(settings.capacities),
            desc=f"{spec.name} details",
            position=progress_position + 1,
            leave=False,
            unit="capacity",
            dynamic_ncols=True,
            disable=None,
        )
        if progress_position is not None
        else None
    )
    system_stage = (
        progress_bar.set_postfix_str if progress_bar is not None else None
    )
    _report_stage(system_stage, "reference trajectories")
    if capacity_bar is not None:
        capacity_bar.reset(total=settings.reference_size)
        capacity_bar.set_description_str(f"{spec.name} setup")
        capacity_bar.unit = "scenario"
        capacity_bar.set_postfix_str("reference trajectories")
    reference_scenarios = sample_scenarios(
        spec,
        settings.reference_size,
        settings.seed_sequence("reference", spec.name),
    )
    reference_batch = batch_simulator(
        spec,
        reference_scenarios,
        settings.trajectory_samples,
        (capacity_bar.update if capacity_bar is not None else None),
    )
    system_raw_dir = _initialize_raw_data(
        raw_output_dir,
        raw_system_index,
        spec,
    )
    geometric_suite, geometric_suites = _geometric_suite_loader(
        spec,
        settings,
        batch_simulator=batch_simulator,
    )
    records = _run_replicates(
        spec,
        settings,
        reference_batch,
        geometric_suite,
        system_raw_dir,
        progress_bar,
        capacity_bar,
        batch_simulator=batch_simulator,
    )
    _finish_system(
        system_raw_dir,
        spec,
        settings,
        reference_batch,
        geometric_suites,
        progress_bar,
        capacity_bar,
    )
    return records


def _run_replicates(
    spec: SystemSpec,
    settings: Settings,
    reference_batch: SimulationBatch,
    geometric_suite: Callable[[str, int, int], GeometricSuite],
    system_raw_dir: Path | None,
    progress_bar: Any | None,
    capacity_bar: Any | None,
    *,
    batch_simulator: BatchSimulator,
) -> ExperimentRecords:
    tree_metrics: list[TreeMetricRow] = []
    leaf_metrics: list[LeafMetricRow] = []
    leaf_boxes_rows: list[LeafBoxRow] = []
    unbounded_tree_metrics: list[UnboundedTreeMetricRow] = []
    unbounded_leaf_metrics: list[UnboundedLeafMetricRow] = []
    unbounded_leaf_boxes: list[UnboundedLeafBoxRow] = []
    suite_metrics: list[SuiteMetricRow] = []
    suite_statuses: list[SuiteStatusRow] = []
    prototype_plots: list[PrototypePlotData] = []
    execution_statuses = [
        _batch_status(spec, None, None, "reference", reference_batch),
    ]

    system_stage = (
        progress_bar.set_postfix_str if progress_bar is not None else None
    )
    for replicate in range(settings.replicates):
        _report_stage(
            system_stage,
            f"replicate {replicate + 1}/{settings.replicates}",
        )
        if capacity_bar is not None:
            capacity_bar.reset(
                total=settings.fitting_size + settings.assessment_size,
            )
            capacity_bar.set_description_str(
                f"{spec.name} replicate {replicate + 1} data",
            )
            capacity_bar.unit = "scenario"
            capacity_bar.set_postfix_str("fitting and assessment trajectories")
        replicate_raw_arrays: dict[str, NDArray[Any]] = {}
        prepared = _prepare_replicate(
            spec,
            settings,
            replicate,
            reference_batch.scenarios,
            reference_batch,
            raw_arrays=replicate_raw_arrays,
            statuses=execution_statuses,
            simulation_progress=(
                capacity_bar.update if capacity_bar is not None else None
            ),
            batch_simulator=batch_simulator,
        )
        if prepared is None:
            execution_statuses.append(
                ExecutionStatusRow(
                    spec.name,
                    replicate,
                    None,
                    "unbounded_tree_fit",
                    "dependency_blocked",
                    reason="fitting batch or transform unavailable",
                ),
            )
            suite_statuses.extend(
                row
                for capacity in settings.capacities
                for row in _blocked_slots(
                    spec,
                    settings,
                    replicate,
                    capacity,
                    "fitting batch or transform unavailable",
                )
            )
            _write_replicate_raw_data(
                system_raw_dir,
                replicate,
                None,
                replicate_raw_arrays,
            )
            if progress_bar is not None:
                progress_bar.update()
            continue
        if capacity_bar is not None:
            capacity_bar.reset(total=len(settings.capacities) + 1)
            capacity_bar.set_description_str(
                f"{spec.name} replicate {replicate + 1}",
            )
            capacity_bar.unit = "tree"
        capacity_stage = (
            capacity_bar.set_postfix_str if capacity_bar is not None else None
        )
        replicate_raw_arrays.update(
            _record_unbounded(
                spec,
                settings,
                prepared,
                replicate,
                unbounded_tree_metrics,
                unbounded_leaf_metrics,
                unbounded_leaf_boxes,
                capacity_bar,
                capacity_stage,
                execution_statuses,
                batch_simulator=batch_simulator,
            ),
        )
        for capacity in settings.capacities:
            result = _evaluate_capacity(
                spec,
                settings,
                prepared,
                geometric_suite,
                replicate,
                capacity,
                capacity_stage,
                execution_statuses,
                batch_simulator=batch_simulator,
            )
            if capacity_bar is not None:
                capacity_bar.update()
            if result is None:
                suite_statuses.extend(
                    _blocked_slots(
                        spec,
                        settings,
                        replicate,
                        capacity,
                        "bounded behavior tree fit unavailable",
                    ),
                )
                continue
            tree_metrics.append(result.tree_row)
            leaf_metrics.extend(result.leaf_rows)
            leaf_boxes_rows.extend(result.box_rows)
            suite_metrics.extend(result.suite_rows)
            suite_statuses.extend(result.status_rows)
            replicate_raw_arrays.update(result.tree_state)
            replicate_raw_arrays.update(result.suite_arrays)
            prototype_plots.extend(
                ()
                if result.prototype_plot is None
                else (result.prototype_plot,),
            )
        _write_replicate_raw_data(
            system_raw_dir,
            replicate,
            prepared,
            replicate_raw_arrays,
        )
        if progress_bar is not None:
            progress_bar.update()

    return ExperimentRecords(
        tree_metrics=tuple(tree_metrics),
        leaf_metrics=tuple(leaf_metrics),
        leaf_boxes=tuple(leaf_boxes_rows),
        unbounded_tree_metrics=tuple(unbounded_tree_metrics),
        unbounded_leaf_metrics=tuple(unbounded_leaf_metrics),
        unbounded_leaf_boxes=tuple(unbounded_leaf_boxes),
        suite_metrics=tuple(suite_metrics),
        suite_statuses=tuple(suite_statuses),
        prototype_plots=tuple(prototype_plots),
        execution_statuses=tuple(execution_statuses),
    )


def _finish_system(
    system_raw_dir: Path | None,
    spec: SystemSpec,
    settings: Settings,
    reference_batch: SimulationBatch,
    geometric_suites: Mapping[tuple[str, int, int], GeometricSuite],
    progress_bar: Any | None,
    capacity_bar: Any | None,
) -> None:
    _write_shared_raw_data(
        system_raw_dir,
        spec,
        settings,
        reference_batch,
        geometric_suites,
    )
    if capacity_bar is not None:
        capacity_bar.close()
    if progress_bar is not None:
        progress_bar.set_postfix_str("complete")
        progress_bar.close()


def _raw_system_name(index: int, name: str) -> str:
    slug = "".join(
        character.lower() if character.isalnum() else "_" for character in name
    ).strip("_")
    return f"{index:02d}_{slug or 'system'}"


def _initialize_raw_data(
    raw_output_dir: Path | None,
    raw_system_index: int,
    spec: SystemSpec,
) -> Path | None:
    if raw_output_dir is None:
        return None
    system_raw_dir = raw_output_dir / _raw_system_name(
        raw_system_index,
        spec.name,
    )
    system_raw_dir.mkdir(parents=True, exist_ok=True)
    return system_raw_dir


def _write_shared_raw_data(
    system_raw_dir: Path | None,
    spec: SystemSpec,
    settings: Settings,
    reference_batch: SimulationBatch,
    geometric_suites: Mapping[tuple[str, int, int], GeometricSuite],
) -> None:
    if system_raw_dir is None:
        return
    suite_arrays: dict[str, NDArray[Any]] = {}
    for (method, budget, repetition), suite in sorted(
        geometric_suites.items(),
    ):
        prefix = (
            f"suite_{method}__budget_{budget:03d}__repetition_{repetition:03d}"
        )
        suite_arrays.update(_batch_arrays(suite.batch, prefix))
    # These dynamic keys name arrays, not NumPy's keyword-only options.
    np.savez_compressed(
        system_raw_dir / "shared.npz",
        **cast(
            "dict[str, Any]", _batch_arrays(reference_batch, "reference", "_")
        ),
        sample_times=spec.sample_times(settings.trajectory_samples),
        **cast("dict[str, Any]", suite_arrays),
    )


def _write_replicate_raw_data(
    system_raw_dir: Path | None,
    replicate: int,
    prepared: _PreparedReplicate | None,
    extra_arrays: Mapping[str, NDArray[Any]],
) -> None:
    if system_raw_dir is None:
        return
    arrays = dict(extra_arrays)
    if prepared is not None:
        arrays.update(
            transform_means=prepared.transform.means,
            transform_scales=prepared.transform.scales,
            retained_coordinates=prepared.transform.retained,
        )
    np.savez_compressed(
        system_raw_dir / f"replicate_{replicate:03d}.npz",
        **cast("dict[str, Any]", arrays),
    )


def _record_unbounded(
    spec: SystemSpec,
    settings: Settings,
    prepared: _PreparedReplicate,
    replicate: int,
    tree_rows: list[UnboundedTreeMetricRow],
    leaf_rows: list[UnboundedLeafMetricRow],
    box_rows: list[UnboundedLeafBoxRow],
    progress_bar: Any | None,
    stage: StageCallback | None,
    statuses: list[ExecutionStatusRow],
    *,
    batch_simulator: BatchSimulator,
) -> dict[str, NDArray[Any]]:
    _report_stage(stage, "unbounded tree and midpoints")
    unbounded = _evaluate_unbounded(
        spec,
        settings,
        prepared,
        replicate,
        statuses,
        batch_simulator=batch_simulator,
    )
    if progress_bar is not None:
        progress_bar.update()
    if unbounded is None:
        return {}
    tree_rows.append(unbounded.tree_row)
    leaf_rows.extend(unbounded.leaf_rows)
    box_rows.extend(unbounded.box_rows)
    return unbounded.tree_state


def _prepare_replicate(
    spec: SystemSpec,
    settings: Settings,
    replicate: int,
    reference_scenarios: FloatArray,
    reference_batch: SimulationBatch,
    *,
    simulation_progress: SimulationProgress | None,
    raw_arrays: dict[str, NDArray[Any]],
    statuses: list[ExecutionStatusRow],
    batch_simulator: BatchSimulator,
) -> _PreparedReplicate | None:
    fitting_scenarios, assessment_scenarios = sample_replicate(
        spec,
        settings,
        replicate,
        reference_scenarios,
    )
    fitting = batch_simulator(
        spec,
        fitting_scenarios,
        settings.trajectory_samples,
        simulation_progress,
    )
    assessment = batch_simulator(
        spec,
        assessment_scenarios,
        settings.trajectory_samples,
        simulation_progress,
    )
    for stage, batch in (("fitting", fitting), ("assessment", assessment)):
        raw_arrays.update(_batch_arrays(batch, stage, "_"))
        statuses.append(_batch_status(spec, replicate, None, stage, batch))
    if not np.all(fitting.valid):
        statuses.append(
            ExecutionStatusRow(
                spec.name,
                replicate,
                None,
                "transform",
                "dependency_blocked",
                reason="complete fitting batch required",
            ),
        )
        return None
    try:
        transform = fit_target_transform(
            fitting.trajectories,
            settings.constant_scale_cutoff,
        )
    except (ValueError, ArithmeticError) as error:
        statuses.append(
            ExecutionStatusRow(
                spec.name,
                replicate,
                None,
                "transform",
                "fit_failed",
                reason=f"{type(error).__name__}: {error}",
            ),
        )
        return None
    statuses.append(
        ExecutionStatusRow(spec.name, replicate, None, "transform", "valid"),
    )
    fitting_targets = transform.transform(fitting.trajectories)
    return _PreparedReplicate(
        fitting_scenarios,
        assessment_scenarios,
        fitting.trajectories,
        assessment.trajectories,
        transform,
        fitting_targets,
        transform.transform(assessment.trajectories)
        if np.all(assessment.valid)
        else None,
        transform.transform(reference_batch.trajectories)
        if np.all(reference_batch.valid)
        else None,
        pairwise_coordinate_rms(fitting_targets),
    )


def _evaluate_capacity(
    spec: SystemSpec,
    settings: Settings,
    prepared: _PreparedReplicate,
    geometric_suite: Callable[[str, int, int], GeometricSuite],
    replicate: int,
    capacity: int,
    stage: StageCallback | None,
    statuses: list[ExecutionStatusRow],
    *,
    batch_simulator: BatchSimulator,
) -> _CapacityEvaluation | None:
    suite_arrays: dict[str, NDArray[Any]] = {}
    _report_stage(stage, f"capacity {capacity}: behavior tree and midpoint")
    bounds = np.asarray(spec.bounds, dtype=np.float64)
    try:
        behavior = fit_behavior_tree(
            prepared.fitting_scenarios,
            prepared.fitting_targets,
            capacity,
            settings.seed_sequence(
                "behavior_tree",
                spec.name,
                replicate=replicate,
                capacity=capacity,
            ),
        )
    except (ValueError, ArithmeticError) as error:
        statuses.append(
            ExecutionStatusRow(
                spec.name,
                replicate,
                capacity,
                "behavior_tree_fit",
                "fit_failed",
                reason=f"{type(error).__name__}: {error}",
            ),
        )
        return None
    realized_leaves = int(behavior.get_n_leaves())
    (
        tree_metric,
        leaf_rows,
        box_rows,
        behavior_suite,
        behavior_status,
        prototype_plot,
    ) = _behavior_midpoint_evaluation(
        spec,
        settings,
        prepared,
        behavior,
        bounds,
        replicate,
        capacity,
        suite_arrays,
        statuses,
        batch_simulator=batch_simulator,
    )
    input_suite, input_status = _input_suite(
        spec,
        settings,
        prepared,
        replicate,
        capacity,
        realized_leaves,
        suite_arrays,
        batch_simulator=batch_simulator,
    )
    archive_suite, archive_status = _archive_suite(
        spec,
        settings,
        prepared,
        replicate,
        capacity,
        realized_leaves,
        suite_arrays,
    )
    geometric_rows, geometric_statuses = _geometric_suite_rows(
        spec,
        settings,
        prepared,
        geometric_suite,
        replicate,
        capacity,
        realized_leaves,
        stage,
        suite_arrays,
    )
    return _CapacityEvaluation(
        tree_metric,
        leaf_rows,
        box_rows,
        tuple(
            row
            for row in (
                behavior_suite,
                input_suite,
                archive_suite,
                *geometric_rows,
            )
            if row is not None
        ),
        (behavior_status, input_status, archive_status, *geometric_statuses),
        prototype_plot,
        _serialized_tree_state(behavior, f"tree_capacity_{capacity:03d}"),
        suite_arrays,
    )


def _input_suite(
    spec: SystemSpec,
    settings: Settings,
    prepared: _PreparedReplicate,
    replicate: int,
    capacity: int,
    budget: int,
    arrays: dict[str, NDArray[Any]],
    *,
    batch_simulator: BatchSimulator,
) -> tuple[SuiteMetricRow | None, SuiteStatusRow]:
    bounds = np.asarray(spec.bounds, dtype=np.float64)
    try:
        tree = fit_input_only_tree(
            prepared.fitting_scenarios,
            bounds,
            budget,
            settings.seed_sequence(
                "input_tree",
                spec.name,
                replicate=replicate,
                capacity=budget,
            ),
        )
    except (ValueError, ArithmeticError) as error:
        return None, SuiteStatusRow(
            spec.name,
            replicate,
            capacity,
            "input_only_midpoint",
            None,
            budget,
            0,
            "fit_failed",
            reason=f"{type(error).__name__}: {error}",
        )
    leaves = int(tree.get_n_leaves())
    if leaves != budget:
        return None, SuiteStatusRow(
            spec.name,
            replicate,
            capacity,
            "input_only_midpoint",
            None,
            budget,
            leaves,
            "structurally_infeasible",
            reason="input tree leaf count differs from budget",
        )
    batch = batch_simulator(
        spec,
        tree_midpoints(tree, bounds),
        settings.trajectory_samples,
        None,
    )
    arrays.update(
        _batch_arrays(
            batch,
            f"suite_capacity_{capacity:03d}__input_only_midpoint",
        ),
    )
    return _score_suite(
        spec,
        settings,
        prepared,
        batch,
        replicate=replicate,
        capacity=capacity,
        budget=budget,
        method="input_only_midpoint",
        repetition=None,
        raw_arrays=arrays,
    )


def _archive_suite(
    spec: SystemSpec,
    settings: Settings,
    prepared: _PreparedReplicate,
    replicate: int,
    capacity: int,
    budget: int,
    arrays: dict[str, NDArray[Any]],
) -> tuple[SuiteMetricRow | None, SuiteStatusRow]:
    try:
        indices = list(
            pam_medoids(
                prepared.archive_distances,
                budget,
                settings.pam_tolerance,
            ),
        )
    except (ValueError, ArithmeticError) as error:
        return None, SuiteStatusRow(
            spec.name,
            replicate,
            capacity,
            "archive_pam",
            None,
            budget,
            0,
            "generation_failed",
            reason=f"{type(error).__name__}: {error}",
        )
    batch = SimulationBatch(
        prepared.fitting_scenarios[indices],
        prepared.fitting_trajectories[indices],
        np.ones(len(indices), dtype=np.bool_),
        ("",) * len(indices),
    )
    prefix = f"suite_capacity_{capacity:03d}__archive_pam"
    arrays.update(_batch_arrays(batch, prefix))
    arrays[f"{prefix}__fitting_indices"] = np.asarray(indices, dtype=np.int64)
    return _score_suite(
        spec,
        settings,
        prepared,
        batch,
        replicate=replicate,
        capacity=capacity,
        budget=budget,
        method="archive_pam",
        repetition=None,
        raw_arrays=arrays,
        reused=True,
    )


def _serialized_tree_state(
    tree: DecisionTreeRegressor,
    prefix: str,
) -> dict[str, NDArray[Any]]:
    state: Mapping[str, Any] = tree.tree_.__getstate__()
    random_state = tree.random_state
    if not isinstance(random_state, (int, np.integer)):
        message = "tree random state must be a fitted integer seed"
        raise TypeError(message)
    return {
        f"{prefix}__max_depth": np.asarray(state["max_depth"], dtype=np.int64),
        f"{prefix}__node_count": np.asarray(
            state["node_count"],
            dtype=np.int64,
        ),
        f"{prefix}__nodes": np.asarray(state["nodes"]).copy(),
        f"{prefix}__values": np.asarray(
            state["values"],
            dtype=np.float64,
        ),
        f"{prefix}__n_features_in": np.asarray(
            tree.n_features_in_,
            dtype=np.int64,
        ),
        f"{prefix}__n_outputs": np.asarray(
            tree.n_outputs_,
            dtype=np.int64,
        ),
        f"{prefix}__max_features": np.asarray(
            tree.max_features_,
            dtype=np.int64,
        ),
        f"{prefix}__random_state": np.asarray(
            random_state,
            dtype=np.int64,
        ),
    }


def _tree_diagnostics(
    spec: SystemSpec,
    settings: Settings,
    prepared: _PreparedReplicate,
    behavior: DecisionTreeRegressor,
    bounds: FloatArray,
    *,
    batch_simulator: BatchSimulator,
) -> _TreeDiagnostics:
    boxes = leaf_boxes(behavior, bounds)
    midpoint_scenarios = np.stack([box.midpoint for box in boxes])
    midpoint_batch = batch_simulator(
        spec,
        midpoint_scenarios,
        settings.trajectory_samples,
        None,
    )
    prediction = prediction_metrics(
        behavior,
        prepared.fitting_scenarios,
        prepared.fitting_targets,
        prepared.assessment_scenarios,
        prepared.assessment_targets,
    )
    realization = available_realization_metrics(
        behavior,
        prepared.assessment_scenarios,
        prepared.assessment_targets,
        midpoint_scenarios,
        tuple(
            prepared.transform.transform(
                midpoint_batch.trajectories[index : index + 1],
            )[0]
            if valid
            else None
            for index, valid in enumerate(midpoint_batch.valid)
        ),
        np.asarray(
            [box.relative_volume for box in boxes],
            dtype=np.float64,
        ),
    )
    if any(
        box.leaf_id != leaf.leaf_id
        for box, leaf in zip(boxes, realization.leaves, strict=True)
    ):
        message = "leaf realization diagnostics do not match path boxes"
        raise ValueError(message)
    return _TreeDiagnostics(
        boxes=boxes,
        midpoint_scenarios=midpoint_scenarios,
        midpoint_batch=midpoint_batch,
        prediction=prediction,
        realization=realization,
        fitting_assignments=np.asarray(
            behavior.apply(prepared.fitting_scenarios),
            dtype=np.int64,
        ),
    )


def _evaluate_unbounded(
    spec: SystemSpec,
    settings: Settings,
    prepared: _PreparedReplicate,
    replicate: int,
    statuses: list[ExecutionStatusRow],
    *,
    batch_simulator: BatchSimulator,
) -> _UnboundedEvaluation | None:
    bounds = np.asarray(spec.bounds, dtype=np.float64)
    try:
        behavior = fit_unbounded_behavior_tree(
            prepared.fitting_scenarios,
            prepared.fitting_targets,
            settings.seed_sequence(
                "unbounded_tree",
                spec.name,
                replicate=replicate,
            ),
        )
    except (ValueError, ArithmeticError) as error:
        statuses.append(
            ExecutionStatusRow(
                spec.name,
                replicate,
                None,
                "unbounded_tree_fit",
                "fit_failed",
                reason=f"{type(error).__name__}: {error}",
            ),
        )
        return None
    diagnostics = _tree_diagnostics(
        spec,
        settings,
        prepared,
        behavior,
        bounds,
        batch_simulator=batch_simulator,
    )
    statuses.append(
        _batch_status(
            spec,
            replicate,
            None,
            "unbounded_midpoints",
            diagnostics.midpoint_batch,
        ),
    )
    _record_undefined_ratios(spec, replicate, None, diagnostics, statuses)
    fitting_counts = np.asarray(
        [
            np.count_nonzero(diagnostics.fitting_assignments == box.leaf_id)
            for box in diagnostics.boxes
        ],
        dtype=np.int64,
    )
    prediction = diagnostics.prediction
    realization = diagnostics.realization
    tree_row = UnboundedTreeMetricRow(
        system=spec.name,
        replicate=replicate,
        realized_leaves=int(behavior.get_n_leaves()),
        depth=int(behavior.get_depth()),
        singleton_fraction=float(np.mean(fitting_counts == 1)),
        fitting_rmse=prediction.fitting_rmse,
        assessment_rmse=prediction.assessment_rmse,
        historical_mean_rmse=prediction.historical_mean_rmse,
        prediction_ratio=prediction.prediction_ratio,
        midpoint_realization_distance=realization.midpoint_realization_distance,
        held_leaf_distance=realization.held_leaf_distance,
        realization_ratio=realization.realization_ratio,
        assessment_unrepresented_volume=(
            realization.assessment_unrepresented_volume
        ),
    )
    leaf_rows = tuple(
        UnboundedLeafMetricRow(
            system=spec.name,
            replicate=replicate,
            leaf_id=box.leaf_id,
            depth=box.depth,
            relative_volume=box.relative_volume,
            fitting_members=int(fitting_members),
            assessment_members=leaf.assessment_members,
            midpoint_realization_distance=leaf.midpoint_realization_distance,
            held_leaf_distance=leaf.held_leaf_distance,
            realization_ratio=leaf.realization_ratio,
        )
        for box, leaf, fitting_members in zip(
            diagnostics.boxes,
            realization.leaves,
            fitting_counts,
            strict=True,
        )
    )
    box_rows = tuple(
        UnboundedLeafBoxRow(
            system=spec.name,
            replicate=replicate,
            leaf_id=box.leaf_id,
            parameter=parameter,
            lower=interval.lower,
            upper=interval.upper,
            lower_inclusive=interval.lower_inclusive,
            upper_inclusive=interval.upper_inclusive,
            midpoint=float(midpoint),
        )
        for box in diagnostics.boxes
        for parameter, interval, midpoint in zip(
            spec.feature_names,
            box.intervals,
            box.midpoint,
            strict=True,
        )
    )
    return _UnboundedEvaluation(
        tree_row=tree_row,
        leaf_rows=leaf_rows,
        box_rows=box_rows,
        tree_state={
            **_serialized_tree_state(behavior, "tree_unbounded"),
            **_batch_arrays(
                diagnostics.midpoint_batch,
                "unbounded_midpoints",
                "_",
            ),
        },
    )


def _behavior_midpoint_evaluation(
    spec: SystemSpec,
    settings: Settings,
    prepared: _PreparedReplicate,
    behavior: DecisionTreeRegressor,
    bounds: FloatArray,
    replicate: int,
    capacity: int,
    suite_arrays: dict[str, NDArray[Any]],
    statuses: list[ExecutionStatusRow],
    *,
    batch_simulator: BatchSimulator,
) -> tuple[
    TreeMetricRow,
    tuple[LeafMetricRow, ...],
    tuple[LeafBoxRow, ...],
    SuiteMetricRow | None,
    SuiteStatusRow,
    PrototypePlotData | None,
]:
    diagnostics = _tree_diagnostics(
        spec,
        settings,
        prepared,
        behavior,
        bounds,
        batch_simulator=batch_simulator,
    )
    boxes = diagnostics.boxes
    midpoint_trajectories = diagnostics.midpoint_batch.trajectories
    _record_undefined_ratios(spec, replicate, capacity, diagnostics, statuses)
    prediction = diagnostics.prediction
    realization = diagnostics.realization
    tree_metric = TreeMetricRow(
        system=spec.name,
        replicate=replicate,
        capacity=capacity,
        realized_leaves=int(behavior.get_n_leaves()),
        depth=int(behavior.get_depth()),
        fitting_rmse=prediction.fitting_rmse,
        assessment_rmse=prediction.assessment_rmse,
        historical_mean_rmse=prediction.historical_mean_rmse,
        prediction_ratio=prediction.prediction_ratio,
        midpoint_realization_distance=(
            realization.midpoint_realization_distance
        ),
        held_leaf_distance=realization.held_leaf_distance,
        realization_ratio=realization.realization_ratio,
        assessment_unrepresented_volume=(
            realization.assessment_unrepresented_volume
        ),
    )
    fitting_assignments = diagnostics.fitting_assignments
    leaf_rows = tuple(
        LeafMetricRow(
            system=spec.name,
            replicate=replicate,
            capacity=capacity,
            leaf_id=box.leaf_id,
            depth=box.depth,
            relative_volume=box.relative_volume,
            fitting_members=int(
                np.count_nonzero(fitting_assignments == box.leaf_id),
            ),
            assessment_members=leaf.assessment_members,
            midpoint_realization_distance=(leaf.midpoint_realization_distance),
            held_leaf_distance=leaf.held_leaf_distance,
            realization_ratio=leaf.realization_ratio,
        )
        for box, leaf in zip(boxes, realization.leaves, strict=True)
    )
    box_rows = tuple(
        LeafBoxRow(
            system=spec.name,
            replicate=replicate,
            capacity=capacity,
            leaf_id=box.leaf_id,
            parameter=parameter,
            lower=interval.lower,
            upper=interval.upper,
            lower_inclusive=interval.lower_inclusive,
            upper_inclusive=interval.upper_inclusive,
            midpoint=float(midpoint),
        )
        for box in boxes
        for parameter, interval, midpoint in zip(
            spec.feature_names,
            box.intervals,
            box.midpoint,
            strict=True,
        )
    )
    prototype_plot = None
    if np.all(diagnostics.midpoint_batch.valid):
        prototype_plot = _prototype_plot_data(
            spec,
            settings,
            prepared,
            behavior,
            boxes,
            fitting_assignments,
            midpoint_trajectories,
            replicate,
            capacity,
        )
    suite_arrays.update(
        _batch_arrays(
            diagnostics.midpoint_batch,
            f"suite_capacity_{capacity:03d}__behavior_midpoint",
        ),
    )
    behavior_suite, behavior_status = _score_suite(
        spec,
        settings,
        prepared,
        diagnostics.midpoint_batch,
        replicate=replicate,
        capacity=capacity,
        budget=int(behavior.get_n_leaves()),
        method="behavior_midpoint",
        repetition=None,
        raw_arrays=suite_arrays,
    )
    return (
        tree_metric,
        leaf_rows,
        box_rows,
        behavior_suite,
        behavior_status,
        prototype_plot,
    )


def _record_undefined_ratios(
    spec: SystemSpec,
    replicate: int,
    capacity: int | None,
    diagnostics: _TreeDiagnostics,
    statuses: list[ExecutionStatusRow],
) -> None:
    prefix = "unbounded" if capacity is None else "bounded"
    for stage, denominator in (
        ("prediction_ratio", diagnostics.prediction.historical_mean_rmse),
        ("realization_ratio", diagnostics.realization.held_leaf_distance),
    ):
        if denominator == 0:
            statuses.append(
                ExecutionStatusRow(
                    spec.name,
                    replicate,
                    capacity,
                    f"{prefix}_{stage}",
                    "undefined",
                    reason="zero denominator",
                ),
            )
    zero_leaves = [
        str(leaf.leaf_id)
        for leaf in diagnostics.realization.leaves
        if leaf.held_leaf_distance == 0
    ]
    if zero_leaves:
        statuses.append(
            ExecutionStatusRow(
                spec.name,
                replicate,
                capacity,
                f"{prefix}_leaf_realization_ratio",
                "undefined",
                reason="zero denominator in leaves: " + ",".join(zero_leaves),
            ),
        )


def _prototype_plot_data(
    spec: SystemSpec,
    settings: Settings,
    prepared: _PreparedReplicate,
    behavior: DecisionTreeRegressor,
    boxes: Sequence[LeafBox],
    fitting_assignments: NDArray[np.int64],
    midpoint_trajectories: FloatArray,
    replicate: int,
    capacity: int,
) -> PrototypePlotData | None:
    if (
        replicate != settings.prototype_plot_replicate
        or capacity != settings.prototype_plot_capacity
    ):
        return None
    leaf_ids = np.asarray([box.leaf_id for box in boxes], dtype=np.int64)
    prototypes = np.stack(
        [
            prepared.fitting_trajectories[fitting_assignments == leaf_id].mean(
                axis=0,
            )
            for leaf_id in leaf_ids
        ],
    )
    midpoint_scenarios = np.stack([box.midpoint for box in boxes])
    predicted = np.asarray(
        behavior.predict(midpoint_scenarios),
        dtype=np.float64,
    ).reshape(len(boxes), -1)
    if not np.allclose(
        prepared.transform.transform(prototypes),
        predicted,
        rtol=1e-12,
        atol=1e-12,
    ):
        message = "physical leaf prototypes disagree with tree predictions"
        raise ValueError(message)
    return PrototypePlotData(
        system=spec.name,
        replicate=replicate,
        capacity=capacity,
        state_names=spec.state_names,
        sample_times=spec.sample_times(settings.trajectory_samples),
        leaf_ids=leaf_ids,
        relative_volumes=np.asarray(
            [box.relative_volume for box in boxes],
            dtype=np.float64,
        ),
        fitting_assignments=fitting_assignments,
        fitting_trajectories=prepared.fitting_trajectories,
        leaf_prototypes=prototypes,
        midpoint_trajectories=midpoint_trajectories,
    )


def _report_stage(callback: StageCallback | None, description: str) -> None:
    if callback is not None:
        callback(description)


def _geometric_suite_rows(
    spec: SystemSpec,
    settings: Settings,
    prepared: _PreparedReplicate,
    geometric_suite: Callable[[str, int, int], GeometricSuite],
    replicate: int,
    capacity: int,
    budget: int,
    stage: StageCallback | None,
    suite_arrays: dict[str, NDArray[Any]],
) -> tuple[tuple[SuiteMetricRow, ...], tuple[SuiteStatusRow, ...]]:
    rows: list[SuiteMetricRow] = []
    statuses: list[SuiteStatusRow] = []
    methods = ("sobol", "random", "lhs")
    total = len(methods) * settings.geometric_repetitions
    completed = 0
    for method in methods:
        for repetition in range(settings.geometric_repetitions):
            completed += 1
            _report_stage(
                stage,
                f"capacity {capacity}, budget {budget}: geometric suite "
                f"{completed}/{total} ({method})",
            )
            suite = geometric_suite(method, budget, repetition)
            row, status = _score_suite(
                spec,
                settings,
                prepared,
                suite.batch,
                replicate=replicate,
                capacity=capacity,
                budget=budget,
                method=suite.method,
                repetition=suite.repetition,
                raw_arrays=suite_arrays,
            )
            if row is not None:
                rows.append(row)
            statuses.append(status)
    return tuple(rows), tuple(statuses)
