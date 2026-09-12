"""Paired coverage inference on one *planned*, dense system grid.

Q is mean reference distance. Geometric Q is the median of all planned raw
repetition Qs, before logging. Capacity effects are medians of paired history
log effects, not ratios of separately aggregated method scores.

Only the three primary comparisons receive simultaneous bands, each covering
all supplied capacities within this system. There is no joint guarantee over
systems or comparisons. Missing evidence never induces complete-case analysis.
No archive reading, simulation, persistence, or confirmatory seed lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from settings import (
    INFERENCE_GEOMETRIC_METHODS,
    INFERENCE_METHODS,
    INFERENCE_PRIMARY_COMPARATORS,
    MIN_TREE_CAPACITY,
    InferenceSettings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from metrics import FloatArray

BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]
ReasonArray = NDArray[np.str_]


@dataclass(frozen=True)
class DistanceEvidence:
    """Distances (H,C,J_method,N), suite validity/reasons (H,C,J_method).

    Valid suites require finite nonnegative distances and an empty reason.
    Invalid suites require a nonempty reason; their entire vector is ignored,
    even if some entries are finite. J_method is one for nongeometric methods.
    """

    distances: FloatArray
    valid: BoolArray
    reasons: ReasonArray


@dataclass(frozen=True)
class CoverageInput:
    """Explicit planned counts, including failed histories and repetitions.

    All six methods must be present, using invalid masked slots for missing
    evidence. Capacities retain caller order. A realized budget of zero means
    unknown and is allowed only when every method's suite is unavailable there.
    Positive budgets are the observed behavior budgets, not nominal capacities.
    Arrays are borrowed read-only for the duration of inference, never mutated.
    """

    system: str
    history_count: int
    capacities: tuple[int, ...]
    reference_count: int
    geometric_repetitions: int
    realized_budgets: IntArray
    methods: Mapping[str, DistanceEvidence]


@dataclass(frozen=True)
class MethodScores:
    """Raw Qs (including zero/nonfinite); reasons identify missing evidence."""

    repetition_q: FloatArray
    repetition_reasons: ReasonArray
    history_q: FloatArray
    history_reasons: ReasonArray


@dataclass(frozen=True)
class PairedEffects:
    """H,C arrays; NaN is unavailable, with separate presentation reasons."""

    log_effect: FloatArray
    ratio: FloatArray
    reasons: ReasonArray
    ratio_reasons: ReasonArray


@dataclass(frozen=True)
class CapacityEffect:
    """Empty reason means a valid log estimate, independently of its ratio."""

    capacity: int
    planned_histories: int
    valid_pairs: int
    log_effect: float | None
    ratio: float | None
    reason: str
    ratio_reason: str


@dataclass(frozen=True)
class CurveBand:
    """Draws stay in original draw order, including undefined NaN entries.

    Missing observed points and supporting comparisons have zero attempted
    draws (shape 0,C). One undefined draw withholds the whole band: no retries,
    filtering, or NaN medians/quantiles. Counts refer to attempted draws.
    Finite log endpoints remain available if exponentiation over/underflows;
    only those ratio endpoints become NaN with a presentation reason.
    """

    status: Literal[
        "available",
        "incomplete_points",
        "supporting",
        "undefined_bootstrap",
    ]
    bootstrap_log_effects: FloatArray
    invalid_draw_count: int
    invalid_draw_counts_by_capacity: IntArray
    half_width: float | None = None
    log_lower: FloatArray | None = None
    log_upper: FloatArray | None = None
    ratio_lower: FloatArray | None = None
    ratio_upper: FloatArray | None = None
    ratio_lower_reasons: ReasonArray | None = None
    ratio_upper_reasons: ReasonArray | None = None
    constant_bootstrap_capacities: BoolArray | None = None
    zero_width: bool | None = None


@dataclass(frozen=True)
class ComparisonResult:
    comparator: str
    role: Literal["primary", "supporting"]
    pairs: PairedEffects
    points: tuple[CapacityEffect, ...]
    band: CurveBand


@dataclass(frozen=True)
class CoverageInference:
    system: str
    capacities: tuple[int, ...]
    history_count: int
    reference_count: int
    geometric_repetitions: int
    realized_budgets: IntArray
    settings: InferenceSettings
    methods: Mapping[str, MethodScores]
    comparisons: Mapping[str, ComparisonResult]


def _validate(data: CoverageInput) -> None:
    if not isinstance(data.system, str) or not data.system.strip():
        message = "a nonempty system name is required"
        raise ValueError(message)
    for count in (
        data.history_count,
        data.reference_count,
        data.geometric_repetitions,
    ):
        if type(count) is not int or count < 1:
            message = "planned counts must be positive integers"
            raise ValueError(message)
    if (
        not isinstance(data.capacities, tuple)
        or not data.capacities
        or any(
            type(c) is not int or c < MIN_TREE_CAPACITY
            for c in data.capacities
        )
        or len(set(data.capacities)) != len(data.capacities)
    ):
        message = "capacities must be a tuple of distinct integers >= 2"
        raise ValueError(message)
    shape = (data.history_count, len(data.capacities))
    budgets = data.realized_budgets
    if (
        budgets.shape != shape
        or budgets.dtype.kind not in "iu"
        or np.any(budgets < 0)
        or np.any(budgets > np.asarray(data.capacities)[None, :])
    ):
        message = "realized budgets need integer H,C shape and 0 <= B <= C"
        raise ValueError(message)
    if set(data.methods) != set(INFERENCE_METHODS):
        message = f"methods must be exactly {INFERENCE_METHODS}"
        raise ValueError(message)
    for method, evidence in data.methods.items():
        repetitions = (
            data.geometric_repetitions
            if method in INFERENCE_GEOMETRIC_METHODS
            else 1
        )
        suite_shape = (*shape, repetitions)
        _validate_evidence(evidence, suite_shape, data.reference_count)
        if np.any((budgets == 0) & np.any(evidence.valid, axis=2)):
            message = "unknown budget cannot have a valid suite"
            raise ValueError(message)
        if method in INFERENCE_GEOMETRIC_METHODS:
            _validate_reused_banks(evidence, budgets)


def _validate_evidence(
    evidence: DistanceEvidence,
    shape: tuple[int, int, int],
    reference_count: int,
) -> None:
    if (
        evidence.distances.shape != (*shape, reference_count)
        or evidence.distances.dtype.kind != "f"
        or evidence.valid.shape != shape
        or evidence.valid.dtype.kind != "b"
        or evidence.reasons.shape != shape
        or evidence.reasons.dtype.kind != "U"
    ):
        message = "distance, boolean validity, and string reason shapes/types"
        raise ValueError(message)
    if np.any(evidence.valid != (evidence.reasons == "")) or np.any(
        ~evidence.valid & (np.char.strip(evidence.reasons) == ""),
    ):
        message = (
            "valid suites need empty reasons; missing suites need reasons"
        )
        raise ValueError(message)
    # Validate per history to avoid copying an entire dense system array.
    for distances, valid in zip(
        evidence.distances,
        evidence.valid,
        strict=True,
    ):
        selected = distances[valid]
        if np.any(~np.isfinite(selected)) or np.any(selected < 0):
            message = "valid suite distances must be finite and nonnegative"
            raise ValueError(message)


def _validate_reused_banks(
    evidence: DistanceEvidence,
    budgets: IntArray,
) -> None:
    for h, row in enumerate(budgets):
        sources: dict[tuple[int, int], FloatArray] = {}
        for c, budget in enumerate(row):
            for j in np.flatnonzero(evidence.valid[h, c]):
                key = (int(budget), int(j))
                vector = evidence.distances[h, c, j]
                if key in sources and not np.array_equal(sources[key], vector):
                    message = "inconsistent repeated-B geometric bank vectors"
                    raise ValueError(message)
                sources[key] = vector


def _method_scores(evidence: DistanceEvidence) -> MethodScores:
    q = np.full(evidence.valid.shape, np.nan)
    # Never score any entries of an invalid suite, including finite survivors.
    for h in range(len(q)):
        with np.errstate(over="ignore", invalid="ignore"):
            q[h][evidence.valid[h]] = np.mean(
                evidence.distances[h][evidence.valid[h]],
                axis=-1,
            )
    history_q = np.full(q.shape[:2], np.nan)
    reasons: list[list[str]] = []
    for h, row in enumerate(q):
        history_reasons: list[str] = []
        for c, repetitions in enumerate(row):
            missing = [
                f"repetition {j}: {evidence.reasons[h, c, j]}"
                for j in range(len(repetitions))
                if not evidence.valid[h, c, j]
            ]
            if not missing:
                with np.errstate(over="ignore", invalid="ignore"):
                    history_q[h, c] = np.median(repetitions)
            history_reasons.append("; ".join(missing))
        reasons.append(history_reasons)
    return MethodScores(
        q,
        evidence.reasons.copy(),
        history_q,
        np.asarray(reasons),
    )


def _ratios(log_values: FloatArray) -> tuple[FloatArray, ReasonArray]:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        ratios = np.exp(log_values)
    available = np.isfinite(log_values)
    representable = np.isfinite(ratios) & (ratios > 0)
    reasons = np.where(
        ~available,
        "log_effect_unavailable",
        np.where(representable, "", "ratio_not_representable"),
    )
    ratios[~available | ~representable] = np.nan
    return ratios, reasons


def _log_effects(
    behavior_q: FloatArray,
    comparator_q: FloatArray,
) -> FloatArray:
    valid = (
        np.isfinite(behavior_q)
        & (behavior_q > 0)
        & np.isfinite(comparator_q)
        & (comparator_q > 0)
    )
    result = np.full(behavior_q.shape, np.nan)
    result[valid] = np.log(behavior_q[valid]) - np.log(comparator_q[valid])
    return result


def _pairs(behavior: MethodScores, comparator: MethodScores) -> PairedEffects:
    logs = _log_effects(behavior.history_q, comparator.history_q)
    reasons: list[list[str]] = []
    for h, row in enumerate(logs):
        history_reasons: list[str] = []
        for c, value in enumerate(row):
            missing = [
                f"{name}: {scores.history_reasons[h, c]}"
                for name, scores in (
                    ("behavior", behavior),
                    ("comparator", comparator),
                )
                if scores.history_reasons[h, c]
            ]
            invalid_q = [
                f"{name}_q_zero_or_nonfinite"
                for name, scores in (
                    ("behavior", behavior),
                    ("comparator", comparator),
                )
                if not scores.history_reasons[h, c]
                and (
                    not np.isfinite(scores.history_q[h, c])
                    or scores.history_q[h, c] <= 0
                )
            ]
            history_reasons.append(
                "" if np.isfinite(value) else "; ".join(missing + invalid_q),
            )
        reasons.append(history_reasons)
    ratios, ratio_reasons = _ratios(logs)
    return PairedEffects(logs, ratios, np.asarray(reasons), ratio_reasons)


def _points(
    pairs: PairedEffects,
    capacities: tuple[int, ...],
) -> tuple[CapacityEffect, ...]:
    points: list[CapacityEffect] = []
    for c, capacity in enumerate(capacities):
        values = pairs.log_effect[:, c]
        count = int(np.count_nonzero(np.isfinite(values)))
        complete = count == len(values)
        effect = float(np.median(values)) if complete else None
        ratios, reasons = _ratios(np.array([effect if complete else np.nan]))
        points.append(
            CapacityEffect(
                capacity,
                len(values),
                count,
                effect,
                float(ratios[0]) if np.isfinite(ratios[0]) else None,
                "" if complete else "incomplete_pairs",
                str(reasons[0]),
            ),
        )
    return tuple(points)


def _empty_band(
    status: Literal["incomplete_points", "supporting"],
    capacity_count: int,
) -> CurveBand:
    return CurveBand(
        status=status,
        bootstrap_log_effects=np.empty((0, capacity_count)),
        invalid_draw_count=0,
        invalid_draw_counts_by_capacity=np.zeros(
            capacity_count,
            dtype=np.int64,
        ),
    )


def _band(
    points: tuple[CapacityEffect, ...],
    draws: FloatArray,
    level: float,
) -> CurveBand:
    invalid = ~np.isfinite(draws)
    invalid_count = int(np.count_nonzero(np.any(invalid, axis=1)))
    invalid_counts = np.sum(invalid, axis=0, dtype=np.int64)
    if invalid_count:
        return CurveBand(
            status="undefined_bootstrap",
            bootstrap_log_effects=draws,
            invalid_draw_count=invalid_count,
            invalid_draw_counts_by_capacity=invalid_counts,
        )
    observed = np.asarray([point.log_effect for point in points], dtype=float)
    width = float(
        np.quantile(
            np.max(np.abs(draws - observed), axis=1),
            level,
            method="linear",
        ),
    )
    lower, upper = observed - width, observed + width
    ratio_lower, lower_reasons = _ratios(lower)
    ratio_upper, upper_reasons = _ratios(upper)
    return CurveBand(
        status="available",
        bootstrap_log_effects=draws,
        invalid_draw_count=0,
        invalid_draw_counts_by_capacity=invalid_counts,
        half_width=width,
        log_lower=lower,
        log_upper=upper,
        ratio_lower=ratio_lower,
        ratio_upper=ratio_upper,
        ratio_lower_reasons=lower_reasons,
        ratio_upper_reasons=upper_reasons,
        constant_bootstrap_capacities=np.all(draws == draws[0], axis=0),
        zero_width=width == 0,
    )


def _weighted_q(
    evidence: DistanceEvidence,
    weights: FloatArray,
) -> FloatArray:
    # Only invoked for whole-curve eligible methods: every suite is valid.
    # Largest intermediate is batch,H,C,J, never batch,H,C,J,N.
    shape = evidence.valid.shape
    with np.errstate(over="ignore", invalid="ignore"):
        return (
            weights @ evidence.distances.reshape(-1, weights.shape[1]).T
        ).reshape(len(weights), *shape)


def _resampled_method_q(
    q: FloatArray,
    budgets: IntArray,
    banks: Mapping[int, IntArray],
) -> FloatArray:
    if not banks:
        return q[..., 0]
    result = np.empty(q.shape[:3])
    for budget, indices in banks.items():
        rows, columns = np.where(budgets == budget)
        selected = np.take_along_axis(
            q[:, rows, columns, :],
            indices[:, None, :],
            axis=-1,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            result[:, rows, columns] = np.median(selected, axis=-1)
    return result


def _bootstrap(
    data: CoverageInput,
    settings: InferenceSettings,
    comparators: tuple[str, ...],
) -> dict[str, FloatArray]:
    samples = {
        method: np.empty((settings.bootstrap_draws, len(data.capacities)))
        for method in comparators
    }
    if not samples:
        return samples
    history_rng = np.random.default_rng(
        settings.seed_sequence(data.system, "history"),
    )
    reference_rng = np.random.default_rng(
        settings.seed_sequence(data.system, "reference"),
    )
    bank_rngs = {
        method: {
            int(b): np.random.default_rng(
                settings.seed_sequence(
                    data.system,
                    "geometric",
                    method=method,
                    budget=int(b),
                ),
            )
            for b in np.unique(data.realized_budgets)
        }
        for method in comparators
        if method in INFERENCE_GEOMETRIC_METHODS
    }
    for start in range(0, settings.bootstrap_draws, settings.batch_size):
        size = min(settings.batch_size, settings.bootstrap_draws - start)
        histories = history_rng.integers(
            data.history_count,
            size=(size, data.history_count),
        )
        references = reference_rng.integers(
            data.reference_count,
            size=(size, data.reference_count),
        )
        weights = (
            np.asarray(
                [
                    np.bincount(row, minlength=data.reference_count)
                    for row in references
                ],
                dtype=np.float64,
            )
            / data.reference_count
        )
        behavior_q = _weighted_q(data.methods["behavior_midpoint"], weights)[
            ...,
            0,
        ]
        selection = (np.arange(size)[:, None], histories, slice(None))
        sampled_behavior = behavior_q[selection]
        for method in comparators:
            banks = {
                b: rng.integers(
                    data.geometric_repetitions,
                    size=(size, data.geometric_repetitions),
                )
                for b, rng in bank_rngs.get(method, {}).items()
            }
            comparator_q = _resampled_method_q(
                _weighted_q(data.methods[method], weights),
                data.realized_budgets,
                banks,
            )
            # Original B and history-specific distances were retained above;
            # duplicate histories select the same already-resampled bank Q.
            paired = _log_effects(
                sampled_behavior,
                comparator_q[selection],
            )
            samples[method][start : start + size] = np.median(paired, axis=1)
    return samples


def infer_coverage(
    data: CoverageInput,
    settings: InferenceSettings,
) -> CoverageInference:
    """Validate and infer without changing inputs or dropping planned slots."""
    _validate(data)
    methods = {
        method: _method_scores(data.methods[method])
        for method in INFERENCE_METHODS
    }
    pairs = {
        method: _pairs(methods["behavior_midpoint"], methods[method])
        for method in INFERENCE_METHODS
        if method != "behavior_midpoint"
    }
    points = {
        method: _points(paired, data.capacities)
        for method, paired in pairs.items()
    }
    eligible = tuple(
        method
        for method in INFERENCE_PRIMARY_COMPARATORS
        if all(point.log_effect is not None for point in points[method])
    )
    samples = _bootstrap(data, settings, eligible)
    comparisons: dict[str, ComparisonResult] = {}
    for method, paired in pairs.items():
        primary = method in INFERENCE_PRIMARY_COMPARATORS
        band = (
            _band(points[method], samples[method], settings.confidence_level)
            if method in samples
            else _empty_band(
                "incomplete_points" if primary else "supporting",
                len(data.capacities),
            )
        )
        comparisons[method] = ComparisonResult(
            method,
            "primary" if primary else "supporting",
            paired,
            points[method],
            band,
        )
    return CoverageInference(
        data.system,
        data.capacities,
        data.history_count,
        data.reference_count,
        data.geometric_repetitions,
        data.realized_budgets.copy(),
        settings,
        methods,
        comparisons,
    )
