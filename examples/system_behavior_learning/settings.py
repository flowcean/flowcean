from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

SEED_ROLES = (
    "reference",
    "fitting",
    "assessment",
    "behavior_tree",
    "unbounded_tree",
    "input_tree",
    "sobol",
    "random",
    "lhs",
)
MIN_TREE_CAPACITY = 2


def _stable_code(value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(value.encode()).digest()[:4],
        "little",
    )


@dataclass(frozen=True)
class ParameterRange:
    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.name:
            message = "parameter ranges require a name"
            raise ValueError(message)
        if not math.isfinite(self.lower) or not math.isfinite(self.upper):
            message = f"parameter range must be finite: {self.name}"
            raise ValueError(message)
        if self.lower >= self.upper:
            message = f"parameter range must have lower < upper: {self.name}"
            raise ValueError(message)

    @property
    def bounds(self) -> tuple[float, float]:
        return (self.lower, self.upper)


@dataclass(frozen=True)
class SystemSettings:
    name: str
    scenario_parameters: tuple[ParameterRange, ...]
    fixed_parameters: tuple[tuple[str, float], ...]
    horizon: tuple[float, float]
    state_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            message = "systems require a name"
            raise ValueError(message)
        feature_names = self.feature_names
        fixed_names = tuple(name for name, _value in self.fixed_parameters)
        if not feature_names or len(set(feature_names)) != len(feature_names):
            message = f"{self.name} requires distinct scenario parameters"
            raise ValueError(message)
        if len(set(fixed_names)) != len(fixed_names):
            message = f"{self.name} requires distinct fixed parameters"
            raise ValueError(message)
        if set(feature_names) & set(fixed_names):
            message = f"{self.name} variable and fixed parameters overlap"
            raise ValueError(message)
        if any(not name for name in fixed_names) or any(
            not math.isfinite(value) for _name, value in self.fixed_parameters
        ):
            message = f"{self.name} fixed parameters must be named and finite"
            raise ValueError(message)
        start, end = self.horizon
        if not math.isfinite(start) or not math.isfinite(end) or start >= end:
            message = f"{self.name} requires a finite increasing horizon"
            raise ValueError(message)
        if not self.state_names or len(set(self.state_names)) != len(
            self.state_names,
        ):
            message = f"{self.name} requires distinct state names"
            raise ValueError(message)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.scenario_parameters)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            parameter.bounds for parameter in self.scenario_parameters
        )

    @property
    def fixed_parameter_values(self) -> dict[str, float]:
        return dict(self.fixed_parameters)


THERMOSTAT = SystemSettings(
    name="Thermostat",
    scenario_parameters=(
        ParameterRange(name="ambient", lower=17.0, upper=19.5),
        ParameterRange(name="heating_power", lower=3.5, upper=7.0),
        ParameterRange(name="cooling_rate", lower=0.15, upper=0.45),
        ParameterRange(name="hysteresis", lower=1.0, upper=3.0),
        ParameterRange(name="initial_temperature", lower=18.0, upper=22.0),
    ),
    fixed_parameters=(),
    horizon=(0.0, 10.0),
    state_names=("temperature",),
)
BOUNCING_BALL = SystemSettings(
    name="Bouncing Ball",
    scenario_parameters=(
        ParameterRange(name="gravity", lower=8.0, upper=12.0),
        ParameterRange(name="restitution", lower=0.85, upper=0.95),
        ParameterRange(name="initial_height", lower=0.5, upper=2.0),
        ParameterRange(name="initial_velocity", lower=-1.0, upper=1.0),
    ),
    fixed_parameters=(),
    horizon=(0.0, 3.0),
    state_names=("height", "velocity"),
)
HYBRID_OSCILLATOR = SystemSettings(
    name="Hybrid Oscillator",
    scenario_parameters=(
        ParameterRange(name="damping_left", lower=0.05, upper=0.4),
        ParameterRange(name="damping_right", lower=0.05, upper=0.4),
        ParameterRange(name="frequency", lower=1.5, upper=2.5),
        ParameterRange(name="initial_position", lower=-1.5, upper=-0.5),
        ParameterRange(name="initial_velocity", lower=-0.5, upper=0.5),
    ),
    fixed_parameters=(),
    horizon=(0.0, 15.0),
    state_names=("position", "velocity"),
)
TANK_VALVES = SystemSettings(
    name="Tank Valves",
    scenario_parameters=(
        ParameterRange(name="inflow", lower=0.015, upper=0.025),
        ParameterRange(name="outlet_area_2", lower=0.009, upper=0.015),
        ParameterRange(name="outlet_area_1", lower=0.0015, upper=0.0025),
        ParameterRange(name="valve_gain", lower=0.0075, upper=0.0125),
        ParameterRange(name="high_level", lower=1.0, upper=1.4),
        ParameterRange(name="low_level", lower=0.3, upper=0.5),
        ParameterRange(name="initial_level_1", lower=0.4, upper=0.8),
        ParameterRange(name="initial_level_2", lower=0.1, upper=0.4),
    ),
    fixed_parameters=(
        ("area_1", 1.0),
        ("area_2", 1.2),
        ("gravity", 9.81),
        ("simulation_rtol", 1e-6),
        ("simulation_atol", 1e-8),
        ("level_tolerance", 1e-6),
        ("wall_height", 1.5),
    ),
    horizon=(0.0, 300.0),
    state_names=("level_1", "level_2"),
)
PID_CONTROLLED_PLANT = SystemSettings(
    name="PID-Controlled Plant",
    scenario_parameters=(
        ParameterRange(name="kp", lower=4.0, upper=8.0),
        ParameterRange(name="ki", lower=1.0, upper=3.0),
        ParameterRange(name="kd", lower=0.5, upper=1.5),
        ParameterRange(name="stiffness", lower=2.0, upper=4.0),
        ParameterRange(name="damping", lower=0.4, upper=0.8),
        ParameterRange(name="setpoint_amp", lower=1.5, upper=2.5),
        ParameterRange(name="setpoint_freq", lower=0.8, upper=1.2),
        ParameterRange(name="actuator_limit", lower=0.75, upper=1.5),
        ParameterRange(name="initial_position", lower=-0.5, upper=0.5),
        ParameterRange(name="initial_velocity", lower=-0.25, upper=0.25),
        ParameterRange(name="initial_integral", lower=-0.25, upper=0.25),
    ),
    fixed_parameters=(),
    horizon=(0.0, 20.0),
    state_names=("position", "velocity", "integral_error"),
)
SYSTEMS = (
    THERMOSTAT,
    BOUNCING_BALL,
    HYBRID_OSCILLATOR,
    TANK_VALVES,
    PID_CONTROLLED_PLANT,
)


@dataclass(frozen=True)
class Settings:
    systems: tuple[SystemSettings, ...] = SYSTEMS
    root_seed: int = 2027
    replicates: int = 30
    fitting_size: int = 256
    assessment_size: int = 512
    reference_size: int = 4096
    capacities: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    geometric_repetitions: int = 31
    trajectory_samples: int = 128
    prototype_plot_replicate: int | None = None
    prototype_plot_capacity: int | None = None
    workers: int = 5
    output_dir: Path = Path("outputs")
    constant_scale_cutoff: float = 1e-12
    pam_tolerance: float = 1e-12
    statistics: InferenceSettings | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize the full configuration, including statistical settings."""
        parameters = asdict(self)
        parameters["output_dir"] = str(self.output_dir)
        return parameters

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Settings:
        """Restore the full current saved configuration, without defaults."""
        systems = tuple(
            SystemSettings(
                name=system["name"],
                scenario_parameters=tuple(
                    ParameterRange(**parameter)
                    for parameter in system["scenario_parameters"]
                ),
                fixed_parameters=tuple(
                    (name, number)
                    for name, number in system["fixed_parameters"]
                ),
                horizon=tuple(system["horizon"]),
                state_names=tuple(system["state_names"]),
            )
            for system in value["systems"]
        )
        statistics = value["statistics"]
        return cls(
            systems=systems,
            root_seed=value["root_seed"],
            replicates=value["replicates"],
            fitting_size=value["fitting_size"],
            assessment_size=value["assessment_size"],
            reference_size=value["reference_size"],
            capacities=tuple(value["capacities"]),
            geometric_repetitions=value["geometric_repetitions"],
            trajectory_samples=value["trajectory_samples"],
            prototype_plot_replicate=value["prototype_plot_replicate"],
            prototype_plot_capacity=value["prototype_plot_capacity"],
            workers=value["workers"],
            output_dir=Path(value["output_dir"]),
            constant_scale_cutoff=value["constant_scale_cutoff"],
            pam_tolerance=value["pam_tolerance"],
            statistics=None
            if statistics is None
            else InferenceSettings(
                root_seed=statistics["root_seed"],
                confidence_level=statistics["confidence_level"],
                bootstrap_draws=statistics["bootstrap_draws"],
                batch_size=statistics["batch_size"],
            ),
        )

    def __post_init__(self) -> None:
        if not self.systems or len(set(self.system_names)) != len(
            self.systems,
        ):
            message = "settings require distinctly named systems"
            raise ValueError(message)
        for name, value, minimum in (
            ("root_seed", self.root_seed, 0),
            ("replicates", self.replicates, 1),
            ("fitting_size", self.fitting_size, 1),
            ("assessment_size", self.assessment_size, 1),
            ("reference_size", self.reference_size, 1),
            ("geometric_repetitions", self.geometric_repetitions, 1),
            ("trajectory_samples", self.trajectory_samples, 2),
        ):
            if type(value) is not int or value < minimum:
                message = f"{name} must be a built-in integer >= {minimum}"
                raise ValueError(message)
        if type(self.workers) is not int or self.workers < 1:
            message = "workers must be a positive built-in integer"
            raise ValueError(message)
        if not self.capacities or any(
            type(value) is not int or value < MIN_TREE_CAPACITY
            for value in self.capacities
        ):
            message = "tree capacities must be integers of at least two"
            raise ValueError(message)
        if len(set(self.capacities)) != len(self.capacities):
            message = "tree capacities must be distinct"
            raise ValueError(message)
        for name, value in (
            ("constant_scale_cutoff", self.constant_scale_cutoff),
            ("pam_tolerance", self.pam_tolerance),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                message = f"{name} must be finite and nonnegative"
                raise ValueError(message)
        prototype_selection = (
            self.prototype_plot_replicate,
            self.prototype_plot_capacity,
        )
        if (prototype_selection[0] is None) != (
            prototype_selection[1] is None
        ):
            message = (
                "prototype plot replicate and capacity must be set together"
            )
            raise ValueError(message)
        if (
            self.prototype_plot_replicate is not None
            and not 0 <= self.prototype_plot_replicate < self.replicates
        ):
            message = (
                "prototype plot replicate is outside configured replicates"
            )
            raise ValueError(message)
        if (
            self.prototype_plot_capacity is not None
            and self.prototype_plot_capacity not in self.capacities
        ):
            message = "prototype plot capacity is not configured"
            raise ValueError(message)

    @property
    def system_names(self) -> tuple[str, ...]:
        return tuple(system.name for system in self.systems)

    def seed_sequence(
        self,
        role: str,
        system: str,
        *,
        replicate: int | None = None,
        capacity: int | None = None,
        repetition: int | None = None,
    ) -> np.random.SeedSequence:
        """Derive a seed without depending on call or iteration order."""
        if role not in SEED_ROLES:
            message = f"unknown seed role: {role}"
            raise ValueError(message)
        if system not in self.system_names:
            message = f"system is not enabled in settings: {system}"
            raise ValueError(message)
        for label, value in (
            ("replicate", replicate),
            ("capacity", capacity),
            ("repetition", repetition),
        ):
            if value is not None and value < 0:
                message = f"{label} cannot be negative"
                raise ValueError(message)
        spawn_key = (
            _stable_code(role),
            _stable_code(system),
            0 if replicate is None else replicate + 1,
            0 if capacity is None else capacity + 1,
            0 if repetition is None else repetition + 1,
        )
        return np.random.SeedSequence(self.root_seed, spawn_key=spawn_key)

    def geometric_seed_sequence(
        self,
        method: str,
        system: str,
        *,
        capacity: int,
        repetition: int,
    ) -> np.random.SeedSequence:
        """Derive a geometric-suite seed shared across data replicates."""
        if method not in {"sobol", "random", "lhs"}:
            message = f"not a geometric suite method: {method}"
            raise ValueError(message)
        return self.seed_sequence(
            method,
            system,
            capacity=capacity,
            repetition=repetition,
        )


INFERENCE_GEOMETRIC_METHODS = ("sobol", "random", "lhs")
INFERENCE_PRIMARY_COMPARATORS = (
    "sobol",
    "input_only_midpoint",
    "archive_pam",
)
INFERENCE_SUPPORTING_COMPARATORS = ("random", "lhs")
INFERENCE_METHODS = (
    "behavior_midpoint",
    *INFERENCE_PRIMARY_COMPARATORS,
    *INFERENCE_SUPPORTING_COMPARATORS,
)


@dataclass(frozen=True)
class InferenceSettings:
    """Statistical analysis configuration nested in experiment settings."""

    root_seed: int
    confidence_level: float = 0.95
    bootstrap_draws: int = 10_000
    batch_size: int = 32

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("root_seed", self.root_seed, 0),
            ("bootstrap_draws", self.bootstrap_draws, 1),
            ("batch_size", self.batch_size, 1),
        ):
            if type(value) is not int or value < minimum:
                message = f"{name} must be an integer >= {minimum}"
                raise ValueError(message)
        if (
            not isinstance(self.confidence_level, (int, float))
            or isinstance(self.confidence_level, bool)
            or not math.isfinite(self.confidence_level)
            or not 0 < self.confidence_level < 1
        ):
            message = "confidence_level must be finite and strictly in (0, 1)"
            raise ValueError(message)

    def seed_sequence(
        self,
        system: str,
        role: str,
        *,
        method: str | None = None,
        budget: int | None = None,
    ) -> np.random.SeedSequence:
        """Key streams by identity, not traversal or batch order."""
        if not isinstance(system, str) or not system.strip():
            message = "inference requires a nonempty system name"
            raise ValueError(message)
        if role not in {"history", "reference", "geometric"}:
            message = f"unknown inference seed role: {role}"
            raise ValueError(message)
        if role == "geometric":
            if (
                method not in INFERENCE_GEOMETRIC_METHODS
                or type(budget) is not int
                or budget < 1
            ):
                message = "geometric streams require a method and positive B"
                raise ValueError(message)
        elif method is not None or budget is not None:
            message = "only geometric streams accept method and budget"
            raise ValueError(message)
        return np.random.SeedSequence(
            self.root_seed,
            spawn_key=(
                _stable_code("coverage-inference-v1"),
                _stable_code(system),
                _stable_code(role),
                0 if method is None else _stable_code(method),
                0 if budget is None else budget,
            ),
        )


FULL = Settings(
    root_seed=2237213407657127,
    prototype_plot_replicate=0,
    prototype_plot_capacity=8,
    statistics=InferenceSettings(root_seed=2210151554901251),
)
DEVELOPMENT = replace(
    FULL,
    root_seed=2027,
    replicates=2,
    geometric_repetitions=3,
    statistics=InferenceSettings(root_seed=944701),
)


def inference_policy() -> dict[str, object]:
    """Fixed core policy, separate from physical seed roles and summaries."""
    return {
        "primary_comparators": list(INFERENCE_PRIMARY_COMPARATORS),
        "supporting_comparators": list(INFERENCE_SUPPORTING_COMPARATORS),
        "estimand": (
            "median across planned histories of paired "
            "log(behavior Q / comparator Q)"
        ),
        "geometric_q": (
            "median of all planned raw repetition Qs before logging"
        ),
        "resampling": (
            "shared paired history and reference indices; independent "
            "per-method realized-B repetition banks shared across histories "
            "and repeated-B capacities"
        ),
        "band": (
            "max-centered absolute deviation over the whole planned "
            "capacity curve"
        ),
        "missing_evidence": (
            "no survivor scoring or replacement; incomplete planned points "
            "withheld; zero/nonfinite Q makes logs unavailable; any invalid "
            "draw withholds the whole band; retain every complete "
            "observed point"
        ),
        "scope": (
            "per primary comparator curve across planned capacities within "
            "one system; not jointly simultaneous across systems or "
            "comparators; supporting points only"
        ),
        "descriptive_summary": (
            "summary.csv raw method medians are not the paired estimator"
        ),
    }
