from __future__ import annotations

import math
from dataclasses import MISSING, FrozenInstanceError, fields, replace
from statistics import mean, median
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from inference import CoverageInput, DistanceEvidence, infer_coverage
from settings import (
    INFERENCE_GEOMETRIC_METHODS,
    INFERENCE_METHODS,
    INFERENCE_PRIMARY_COMPARATORS,
    InferenceSettings,
    Settings,
)
from threadpoolctl import threadpool_limits

if TYPE_CHECKING:
    from numpy.typing import NDArray

CAPACITIES = (2, 4, 8, 16, 32, 64)
SMALL = InferenceSettings(root_seed=613, bootstrap_draws=31, batch_size=7)
BEHAVIOR = "behavior_midpoint"


def fixture(
    *,
    histories: int = 4,
    repetitions: int = 4,
    references: int = 5,
) -> CoverageInput:
    budgets = np.array(
        [
            [2, 2, 4, 4, 8, 8],
            [1, 3, 3, 6, 6, 9],
            [2, 4, 4, 4, 7, 7],
            [1, 2, 5, 5, 5, 5],
        ],
        dtype=np.int64,
    )[:histories]
    methods: dict[str, DistanceEvidence] = {}
    for m, method in enumerate(INFERENCE_METHODS):
        geometric = method in INFERENCE_GEOMETRIC_METHODS
        j_count = repetitions if geometric else 1
        distances = np.empty((histories, 6, j_count, references))
        for h in range(histories):
            for c in range(6):
                for j in range(j_count):
                    for n in range(references):
                        # Literal dense data, including history-specific bank
                        # transforms and repeated realized budgets.
                        b = int(budgets[h, c]) if geometric else c + 1
                        distances[h, c, j, n] = (
                            1
                            + (m + 1) * (h + 1) ** 2
                            + b * (j + 1) ** 2
                            + (n + 1) * ((h + j + m) % 3 + 1)
                        ) / 10
        shape = distances.shape[:-1]
        methods[method] = DistanceEvidence(
            distances,
            np.ones(shape, dtype=bool),
            np.full(shape, "", dtype="<U80"),
        )
    return CoverageInput(
        "dense-test-system",
        histories,
        CAPACITIES,
        references,
        repetitions,
        budgets,
        methods,
    )


def constant(data: CoverageInput, value: float = 1.0) -> CoverageInput:
    return replace(
        data,
        methods={
            method: replace(
                evidence,
                distances=np.full_like(
                    evidence.distances,
                    value,
                ),
            )
            for method, evidence in data.methods.items()
        },
    )


def missing(
    data: CoverageInput,
    method: str,
    h: int,
    c: int,
    j: int = 0,
) -> CoverageInput:
    source = data.methods[method]
    evidence = DistanceEvidence(
        source.distances.copy(),
        source.valid.copy(),
        source.reasons.copy(),
    )
    evidence.distances[h, c, j] = np.nan
    evidence.valid[h, c, j] = False
    evidence.reasons[h, c, j] = "simulation_failed"
    return replace(data, methods={**data.methods, method: evidence})


def resamples(
    data: CoverageInput,
    settings: InferenceSettings,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    dict[int, NDArray[np.int64]],
]:
    h = np.random.default_rng(settings.seed_sequence(data.system, "history"))
    r = np.random.default_rng(settings.seed_sequence(data.system, "reference"))
    banks = {
        int(b): np.random.default_rng(
            settings.seed_sequence(
                data.system,
                "geometric",
                method="sobol",
                budget=int(b),
            ),
        ).integers(
            data.geometric_repetitions,
            size=(
                settings.bootstrap_draws,
                data.geometric_repetitions,
            ),
        )
        for b in np.unique(data.realized_budgets)
    }
    return (
        h.integers(
            data.history_count,
            size=(
                settings.bootstrap_draws,
                data.history_count,
            ),
        ),
        r.integers(
            data.reference_count,
            size=(
                settings.bootstrap_draws,
                data.reference_count,
            ),
        ),
        banks,
    )


def dense_oracle(
    data: CoverageInput,
    settings: InferenceSettings,
    method: str,
) -> NDArray[np.float64]:
    histories, references, banks = resamples(data, settings)
    effects = np.empty((settings.bootstrap_draws, len(data.capacities)))
    for d in range(settings.bootstrap_draws):
        for c in range(len(data.capacities)):
            paired: list[float] = []
            for h in histories[d]:
                behavior = mean(
                    float(
                        data.methods[BEHAVIOR].distances[
                            h,
                            c,
                            0,
                            n,
                        ],
                    )
                    for n in references[d]
                )
                repetitions = (
                    banks[int(data.realized_budgets[h, c])][d]
                    if method == "sobol"
                    else [0]
                )
                comparator = median(
                    [
                        mean(
                            float(data.methods[method].distances[h, c, j, n])
                            for n in references[d]
                        )
                        for j in repetitions
                    ],
                )
                paired.append(math.log(behavior) - math.log(comparator))
            effects[d, c] = median(paired)
    return effects


def test_results_are_identical_with_one_and_four_numerical_threads() -> None:
    data = fixture()
    with threadpool_limits(limits=1, user_api=None):
        serial = infer_coverage(data, SMALL)
    with threadpool_limits(limits=4, user_api=None):
        parallel = infer_coverage(data, SMALL)
    for method, expected in serial.comparisons.items():
        actual = parallel.comparisons[method]
        assert actual.points == expected.points
        for field in fields(expected.band):
            left = getattr(expected.band, field.name)
            right = getattr(actual.band, field.name)
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            else:
                assert left == right


def test_dense_reference_history_and_bank_bootstrap_and_bands() -> None:
    data = fixture()
    result = infer_coverage(data, SMALL)
    for method in INFERENCE_PRIMARY_COMPARATORS:
        comparison = result.comparisons[method]
        expected = dense_oracle(data, SMALL, method)
        np.testing.assert_allclose(
            comparison.band.bootstrap_log_effects,
            expected,
            atol=2e-15,
        )
        points = np.array([p.log_effect for p in comparison.points])
        maxima = sorted(max(abs(row - points)) for row in expected)
        # Independent linear .95 quantile calculation (31 draws => index 28.5).
        width = (maxima[28] + maxima[29]) / 2
        band = comparison.band
        assert band.status == "available"
        assert band.half_width == pytest.approx(width)
        assert band.log_lower is not None
        assert band.log_upper is not None
        assert band.ratio_lower is not None
        assert band.ratio_upper is not None
        np.testing.assert_allclose(band.log_lower, points - width)
        np.testing.assert_allclose(band.log_upper, points + width)
        np.testing.assert_allclose(band.ratio_lower, np.exp(points - width))
        np.testing.assert_allclose(band.ratio_upper, np.exp(points + width))
        assert band.invalid_draw_count == 0
        assert band.bootstrap_log_effects.shape == (31, 6)
    assert result.settings == SMALL
    assert result.capacities == CAPACITIES


def test_median_before_log_and_paired_history_median_order() -> None:
    data = constant(fixture(histories=2, repetitions=2))
    data.methods[BEHAVIOR].distances[0] = 1
    data.methods[BEHAVIOR].distances[1] = 9
    data.methods["sobol"].distances[:, :, 0] = 1
    data.methods["sobol"].distances[:, :, 1] = 9
    result = infer_coverage(data, SMALL)
    np.testing.assert_array_equal(result.methods["sobol"].history_q, 5)
    comparison = result.comparisons["sobol"]
    expected = (math.log(1 / 5) + math.log(9 / 5)) / 2
    assert comparison.points[0].log_effect == pytest.approx(expected)
    assert comparison.points[0].ratio == pytest.approx(3 / 5)
    assert expected != pytest.approx(math.log(median([1 / 5, 9 / 5])))
    assert expected != pytest.approx(math.log(median([1, 9]) / 5))
    # A second fixture distinguishes pairing from separate method medians
    # even with odd H (where log-median ratios alone would not distinguish).
    data = constant(fixture(histories=3))
    for h, (behavior, comparator) in enumerate(
        zip(
            [1, 10, 100],
            [10, 100, 1],
            strict=True,
        ),
    ):
        data.methods[BEHAVIOR].distances[h] = behavior
        data.methods["archive_pam"].distances[h] = comparator
    point = infer_coverage(data, SMALL).comparisons["archive_pam"].points[0]
    assert point.ratio == pytest.approx(0.1)
    assert point.ratio != median([1, 10, 100]) / median([10, 100, 1])


def test_reference_weights_direct_calculation_and_shared_cancellation() -> (
    None
):
    data = constant(fixture(histories=1, references=3))
    data.methods[BEHAVIOR].distances[:] = [1, 3, 8]
    histories, references, _ = resamples(data, SMALL)
    assert np.all(histories == 0)
    expected = [
        math.log((counts[0] + 3 * counts[1] + 8 * counts[2]) / 3)
        for counts in (np.bincount(row, minlength=3) for row in references)
    ]
    result = infer_coverage(data, SMALL)
    np.testing.assert_allclose(
        result.comparisons["archive_pam"].band.bootstrap_log_effects[:, 0],
        expected,
    )
    for evidence in data.methods.values():
        evidence.distances[:] = [1, 3, 8]
    result = infer_coverage(data, SMALL)
    for comparison in result.comparisons.values():
        np.testing.assert_array_equal(comparison.pairs.log_effect, 0)
        np.testing.assert_array_equal(comparison.pairs.ratio, 1)
        assert all(p.log_effect == 0 for p in comparison.points)
        if comparison.role == "primary":
            band = comparison.band
            np.testing.assert_array_equal(band.bootstrap_log_effects, 0)
            assert band.zero_width
            assert band.half_width == 0
            assert band.constant_bootstrap_capacities is not None
            assert band.ratio_lower is not None
            assert band.ratio_upper is not None
            assert np.all(band.constant_bootstrap_capacities)
            np.testing.assert_array_equal(band.ratio_lower, 1)
            np.testing.assert_array_equal(band.ratio_upper, 1)


def test_shared_banks_duplicate_histories_and_independent_budget_streams() -> (
    None
):
    data = fixture()
    histories, _, banks = resamples(data, SMALL)
    assert any(len(set(row)) < data.history_count for row in histories)
    assert len({indices.tobytes() for indices in banks.values()}) == len(banks)
    # Dense oracle uses original observed B, not resampled-row B or nominal C.
    np.testing.assert_allclose(
        infer_coverage(data, SMALL)
        .comparisons["sobol"]
        .band.bootstrap_log_effects,
        dense_oracle(data, SMALL, "sobol"),
    )
    data = constant(data)
    data.methods[BEHAVIOR].distances[:] = 2
    for h in range(data.history_count):
        data.methods["sobol"].distances[h] = np.arange(1, 5)[
            None,
            :,
            None,
        ] ** (h + 1)
    result = infer_coverage(data, SMALL).comparisons["sobol"]
    np.testing.assert_allclose(
        result.band.bootstrap_log_effects,
        dense_oracle(data, SMALL, "sobol"),
    )
    # Repeated B within a history shares source vectors; different histories
    # are explicitly allowed to have different transforms of the same bank.
    np.testing.assert_array_equal(
        data.methods["sobol"].distances[0, 0],
        data.methods["sobol"].distances[0, 1],
    )


def test_determinism_batching_order_and_missing_unrelated() -> None:
    data = fixture()
    baseline = infer_coverage(data, SMALL)
    permutation = np.array([5, 1, 3, 0, 4, 2])
    reordered = replace(
        data,
        capacities=tuple(data.capacities[c] for c in permutation),
        realized_budgets=data.realized_budgets[:, permutation],
        methods={
            method: DistanceEvidence(
                e.distances[:, permutation],
                e.valid[:, permutation],
                e.reasons[:, permutation],
            )
            for method, e in reversed(list(data.methods.items()))
        },
    )
    for changed, config, order in (
        (data, SMALL, np.arange(6)),
        (data, replace(SMALL, batch_size=1), np.arange(6)),
        (data, replace(SMALL, batch_size=31), np.arange(6)),
        (reordered, SMALL, permutation),
        (missing(data, "archive_pam", 0, 0), SMALL, np.arange(6)),
        (missing(data, "random", 0, 0), SMALL, np.arange(6)),
    ):
        actual = infer_coverage(changed, config)
        for method in ("sobol", "input_only_midpoint"):
            np.testing.assert_allclose(
                actual.comparisons[method].band.bootstrap_log_effects,
                baseline.comparisons[method].band.bootstrap_log_effects[
                    :,
                    order,
                ],
                atol=2e-15,
            )


@pytest.mark.parametrize(
    "method",
    [BEHAVIOR, "sobol", "archive_pam", "random"],
)
def test_strict_missing_repetitions_histories_and_whole_curve(
    method: str,
) -> None:
    data = missing(fixture(), method, 1, 2)
    # Finite survivors of a failed suite must not be scored.
    data.methods[method].distances[1, 2, 0, :3] = 0.001
    result = infer_coverage(data, SMALL)
    assert math.isnan(result.methods[method].repetition_q[1, 2, 0])
    assert math.isnan(result.methods[method].history_q[1, 2])
    assert "simulation_failed" in result.methods[method].history_reasons[1, 2]
    for name, comparison in result.comparisons.items():
        affected = method in {BEHAVIOR, name}
        assert comparison.points[2].valid_pairs == (3 if affected else 4)
        if affected:
            assert comparison.points[2].log_effect is None
            assert comparison.points[2].reason == "incomplete_pairs"
            assert "simulation_failed" in comparison.pairs.reasons[1, 2]
            assert np.isfinite(comparison.pairs.log_effect[[0, 2, 3], 2]).all()
        assert all(p.log_effect is not None for p in comparison.points[:2])
        expected_status = (
            "supporting"
            if comparison.role == "supporting"
            else "incomplete_points"
            if affected
            else "available"
        )
        assert comparison.band.status == expected_status
        if expected_status != "available":
            assert comparison.band.bootstrap_log_effects.shape == (0, 6)


def test_unknown_budget_preserves_failed_planned_slot() -> None:
    data = fixture()
    for method, evidence in data.methods.items():
        for j in range(evidence.valid.shape[-1]):
            data = missing(data, method, 1, 3, j)
    budgets = data.realized_budgets.copy()
    budgets[1, 3] = 0
    data = replace(data, realized_budgets=budgets)
    result = infer_coverage(data, SMALL)
    assert result.realized_budgets[1, 3] == 0
    for comparison in result.comparisons.values():
        assert comparison.points[3].valid_pairs == 3
        assert comparison.points[3].planned_histories == 4
        assert comparison.points[3].log_effect is None
        assert comparison.points[2].log_effect is not None
        assert "simulation_failed" in comparison.pairs.reasons[1, 3]
    data.methods[BEHAVIOR].valid[1, 3, 0] = True
    data.methods[BEHAVIOR].reasons[1, 3, 0] = ""
    data.methods[BEHAVIOR].distances[1, 3, 0] = 1
    with pytest.raises(ValueError, match="unknown budget"):
        infer_coverage(data, SMALL)


def test_zero_individual_geometric_q_and_undefined_draws_no_retry() -> None:
    data = constant(fixture(histories=1, repetitions=2, references=2))
    data.methods["sobol"].distances[:, :, 0] = 0
    result = infer_coverage(data, SMALL)
    comparison = result.comparisons["sobol"]
    np.testing.assert_array_equal(
        result.methods["sobol"].repetition_q[..., 0],
        0,
    )
    np.testing.assert_array_equal(result.methods["sobol"].history_q, 0.5)
    assert all(p.ratio == pytest.approx(2) for p in comparison.points)
    _, _, banks = resamples(data, SMALL)
    expected_invalid = np.column_stack(
        [np.all(banks[int(b)] == 0, axis=1) for b in data.realized_budgets[0]],
    )
    band = comparison.band
    assert band.status == "undefined_bootstrap"
    assert band.half_width is None
    assert band.bootstrap_log_effects.shape == (SMALL.bootstrap_draws, 6)
    np.testing.assert_array_equal(
        np.isnan(band.bootstrap_log_effects),
        expected_invalid,
    )
    np.testing.assert_array_equal(
        band.invalid_draw_counts_by_capacity,
        expected_invalid.sum(axis=0),
    )
    assert band.invalid_draw_count == np.any(expected_invalid, axis=1).sum()
    assert 0 < band.invalid_draw_count < SMALL.bootstrap_draws
    assert result.comparisons["archive_pam"].band.status == "available"


def test_reference_zero_resample_withholds_band_retains_observed_points() -> (
    None
):
    data = constant(fixture(histories=1, references=2))
    data.methods[BEHAVIOR].distances[:] = [0, 2]
    result = infer_coverage(data, SMALL)
    _, references, _ = resamples(data, SMALL)
    invalid = np.all(references == 0, axis=1)
    for method in INFERENCE_PRIMARY_COMPARATORS:
        comparison = result.comparisons[method]
        assert all(point.log_effect == 0 for point in comparison.points)
        assert comparison.band.status == "undefined_bootstrap"
        assert comparison.band.invalid_draw_count == invalid.sum()
        np.testing.assert_array_equal(
            np.isnan(comparison.band.bootstrap_log_effects[:, 0]),
            invalid,
        )


@pytest.mark.parametrize("method", [BEHAVIOR, "sobol"])
@pytest.mark.parametrize("value", [0.0, np.finfo(np.float64).max])
def test_zero_or_computationally_nonfinite_actual_q(
    method: str,
    value: float,
) -> None:
    data = constant(fixture(histories=1))
    data.methods[method].distances[:] = value
    result = infer_coverage(data, SMALL)
    expected_q = 0 if value == 0 else np.inf
    np.testing.assert_array_equal(result.methods[method].history_q, expected_q)
    comparison = result.comparisons["sobol"]
    assert comparison.points[0].log_effect is None
    assert "q_zero_or_nonfinite" in comparison.pairs.reasons[0, 0]
    assert comparison.band.status == "incomplete_points"


@pytest.mark.parametrize(
    ("behavior", "comparator"),
    [(1e300, 1e-300), (1e-300, 1e300)],
)
def test_unrepresentable_ratio_retains_finite_log_effect_and_band(
    behavior: float,
    comparator: float,
) -> None:
    data = constant(
        fixture(histories=1, references=1, repetitions=1),
        comparator,
    )
    data.methods[BEHAVIOR].distances[:] = behavior
    result = infer_coverage(data, SMALL)
    for comparison in result.comparisons.values():
        point = comparison.points[0]
        assert point.log_effect == pytest.approx(
            math.log(behavior) - math.log(comparator),
        )
        assert point.reason == ""
        assert point.ratio is None
        assert point.ratio_reason == "ratio_not_representable"
        assert comparison.pairs.reasons[0, 0] == ""
        assert (
            comparison.pairs.ratio_reasons[0, 0] == "ratio_not_representable"
        )
        if comparison.role == "primary":
            band = comparison.band
            assert band.status == "available"
            assert band.zero_width
            assert band.log_lower is not None
            assert band.ratio_lower is not None
            assert np.isfinite(band.log_lower).all()
            assert np.isnan(band.ratio_lower).all()
            assert np.all(
                band.ratio_lower_reasons == "ratio_not_representable",
            )
    assert np.all(result.methods[BEHAVIOR].history_q > 0)


def test_inputs_not_mutated_even_readonly_and_output_does_not_alias() -> None:
    data = fixture()
    arrays: list[NDArray[Any]] = [data.realized_budgets]
    for evidence in data.methods.values():
        arrays.extend([evidence.distances, evidence.valid, evidence.reasons])
    copies = [array.copy() for array in arrays]
    for array in arrays:
        array.setflags(write=False)
    result = infer_coverage(data, SMALL)
    for original, copy in zip(arrays, copies, strict=True):
        np.testing.assert_array_equal(original, copy)
    result.realized_budgets[:] = 0
    result.methods[BEHAVIOR].repetition_reasons[:] = "changed"
    for original, copy in zip(arrays, copies, strict=True):
        np.testing.assert_array_equal(original, copy)


def test_separate_settings_defaults_frozen_and_stable_namespaced_streams() -> (
    None
):
    settings = InferenceSettings(root_seed=19)
    assert settings.confidence_level == 0.95
    assert settings.bootstrap_draws == 10_000
    assert settings.batch_size == 32
    root_field = next(
        f for f in fields(InferenceSettings) if f.name == "root_seed"
    )
    assert root_field.default is MISSING
    assert root_field.default_factory is MISSING
    with pytest.raises(TypeError):
        InferenceSettings()  # type: ignore[call-arg]
    with pytest.raises(FrozenInstanceError):
        settings.batch_size = 1  # type: ignore[misc]
    assert "bootstrap_draws" not in {field.name for field in fields(Settings)}
    states = []
    for system in ("one", "two"):
        states.extend(
            settings.seed_sequence(system, role).generate_state(8)
            for role in ("history", "reference")
        )
        states.extend(
            settings.seed_sequence(
                system,
                "geometric",
                method=method,
                budget=budget,
            ).generate_state(8)
            for method in INFERENCE_GEOMETRIC_METHODS
            for budget in (1, 2)
        )
    assert len({state.tobytes() for state in states}) == len(states)
    np.testing.assert_array_equal(
        settings.seed_sequence("one", "history").generate_state(8),
        states[0],
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("root_seed", True),
        ("root_seed", -1),
        ("root_seed", 1.5),
        ("bootstrap_draws", True),
        ("bootstrap_draws", 0),
        ("batch_size", False),
        ("batch_size", -2),
        ("batch_size", 2.5),
        ("confidence_level", True),
        ("confidence_level", np.nan),
        ("confidence_level", np.inf),
        ("confidence_level", 0),
        ("confidence_level", 1),
        ("confidence_level", "0.95"),
    ],
)
def test_invalid_inference_settings(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        replace(SMALL, **{field: value})


@pytest.mark.parametrize(
    "metadata",
    [
        {"system": ""},
        {"history_count": True},
        {"history_count": 0},
        {"reference_count": 0},
        {"geometric_repetitions": 1.5},
        {"capacities": ()},
        {"capacities": (2, 2)},
        {"capacities": (True,)},
        {"capacities": (1,)},
        {"capacities": [2, 4, 8, 16, 32, 64]},
        {"realized_budgets": np.ones((4, 5), dtype=np.int64)},
        {"realized_budgets": np.ones((4, 6), dtype=bool)},
        {"realized_budgets": np.full((4, 6), -1, dtype=np.int64)},
        {"realized_budgets": np.full((4, 6), 65, dtype=np.int64)},
        {"realized_budgets": np.ones((4, 6), dtype=float)},
    ],
)
def test_invalid_planned_metadata(metadata: dict[str, Any]) -> None:
    with pytest.raises(
        ValueError,
        match=r"required|counts|capacities|budgets",
    ):
        infer_coverage(replace(fixture(), **metadata), SMALL)


@pytest.mark.parametrize(
    "change",
    [
        "unknown_method",
        "absent_method",
        "missing_history",
        "missing_capacity",
        "missing_repetition",
        "missing_reference",
        "mask_shape",
        "reason_shape",
        "mask_dtype",
        "reason_dtype",
        "negative",
        "nan",
        "inf",
        "valid_reason",
        "missing_reason",
        "whitespace_reason",
        "inconsistent_bank",
    ],
)
def test_invalid_evidence(change: str) -> None:
    data = fixture()
    evidence = data.methods["sobol"]
    methods = dict(data.methods)
    if change == "unknown_method":
        methods["unknown"] = evidence
    elif change == "absent_method":
        del methods["sobol"]
    elif change.startswith("missing_") and change != "missing_reason":
        axis = ["history", "capacity", "repetition", "reference"].index(
            change.removeprefix("missing_"),
        )
        methods["sobol"] = replace(
            evidence,
            distances=np.delete(evidence.distances, 0, axis=axis),
        )
    elif change in {
        "mask_shape",
        "reason_shape",
        "mask_dtype",
        "reason_dtype",
    }:
        methods["sobol"] = {
            "mask_shape": replace(evidence, valid=evidence.valid[:1]),
            "reason_shape": replace(evidence, reasons=evidence.reasons[:1]),
            "mask_dtype": replace(evidence, valid=evidence.valid.astype(int)),
            "reason_dtype": replace(
                evidence,
                reasons=evidence.reasons.astype(object),
            ),
        }[change]
    elif change in {"negative", "nan", "inf"}:
        evidence.distances[0, 0, 0, 0] = {
            "negative": -1,
            "nan": np.nan,
            "inf": np.inf,
        }[change]
    elif change == "valid_reason":
        evidence.reasons[0, 0, 0] = "failed"
    elif change in {"missing_reason", "whitespace_reason"}:
        evidence.valid[0, 0, 0] = False
        evidence.reasons[0, 0, 0] = (
            " " if change == "whitespace_reason" else ""
        )
    else:
        evidence.distances[0, 1, 0, 0] += 1
    with pytest.raises(
        ValueError,
        match=r"methods|shapes|distances|reasons|bank",
    ):
        infer_coverage(replace(data, methods=methods), SMALL)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"system": "", "role": "history"},
        {"system": "one", "role": "invalid"},
        {"system": "one", "role": "history", "method": "sobol", "budget": 2},
        {
            "system": "one",
            "role": "geometric",
            "method": "random",
            "budget": 0,
        },
        {
            "system": "one",
            "role": "geometric",
            "method": "sobol",
            "budget": True,
        },
        {
            "system": "one",
            "role": "geometric",
            "method": "archive_pam",
            "budget": 2,
        },
    ],
)
def test_invalid_stream_metadata(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"inference|streams"):
        SMALL.seed_sequence(**kwargs)


def test_whole_missing_history_is_not_dropped() -> None:
    data = fixture()
    for method, evidence in data.methods.items():
        for c in range(len(data.capacities)):
            for j in range(evidence.valid.shape[-1]):
                data = missing(data, method, 0, c, j)
    data.realized_budgets[0] = 0
    result = infer_coverage(data, SMALL)
    assert result.history_count == 4
    for comparison in result.comparisons.values():
        assert all(point.log_effect is None for point in comparison.points)
        assert all(point.valid_pairs == 3 for point in comparison.points)
        assert np.isfinite(comparison.pairs.log_effect[1:]).all()
        assert comparison.band.bootstrap_log_effects.shape == (0, 6)


def test_undefined_q_matters_only_for_histories_actually_resampled() -> None:
    data = constant(fixture(histories=2, references=2))
    data.methods[BEHAVIOR].distances[0] = [0, 2]
    histories, references, _ = resamples(data, SMALL)
    expected_invalid = np.all(references == 0, axis=1) & np.any(
        histories == 0,
        axis=1,
    )
    assert np.any(np.all(references == 0, axis=1) & ~expected_invalid)
    comparison = infer_coverage(data, SMALL).comparisons["archive_pam"]
    assert comparison.band.invalid_draw_count == expected_invalid.sum()
    np.testing.assert_array_equal(
        np.isnan(comparison.band.bootstrap_log_effects[:, 0]),
        expected_invalid,
    )
    assert all(point.log_effect == 0 for point in comparison.points)


def test_only_unrepresentable_band_endpoint_is_withheld() -> None:
    data = constant(fixture(histories=2, references=1, repetitions=1))
    data.methods[BEHAVIOR].distances[:, 0] = 1e304
    data.methods[BEHAVIOR].distances[0, 1] = 1e-300
    data.methods[BEHAVIOR].distances[1, 1] = 1e300
    comparison = infer_coverage(data, SMALL).comparisons["archive_pam"]
    assert comparison.points[0].ratio == pytest.approx(1e304)
    assert comparison.points[0].ratio_reason == ""
    band = comparison.band
    assert band.status == "available"
    assert band.log_upper is not None
    assert band.ratio_lower is not None
    assert band.ratio_upper is not None
    assert band.ratio_upper_reasons is not None
    assert np.isfinite(band.log_upper).all()
    assert math.isnan(band.ratio_upper[0])
    assert band.ratio_upper_reasons[0] == "ratio_not_representable"
    assert np.isfinite(band.ratio_lower).all()
    assert np.isfinite(band.ratio_upper[1:]).all()


def test_same_configuration_is_exactly_deterministic() -> None:
    data = fixture()
    first = infer_coverage(data, SMALL)
    second = infer_coverage(data, SMALL)
    for method in INFERENCE_PRIMARY_COMPARATORS:
        np.testing.assert_array_equal(
            first.comparisons[method].band.bootstrap_log_effects,
            second.comparisons[method].band.bootstrap_log_effects,
        )
        assert (
            first.comparisons[method].points
            == second.comparisons[method].points
        )
