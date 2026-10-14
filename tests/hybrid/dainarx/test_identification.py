from dataclasses import replace

import numpy as np
import polars as pl
import pytest
from sklearn.svm import SVC

from flowcean.hybrid.dainarx import fit_dainarx
from flowcean.hybrid.dainarx.learner import _cluster

from .conftest import sampled_system


def test_segmentation_clustering_guards_and_reset_phases(identified):
    fit, training = identified
    assert fit.eta == pytest.approx(
        1e-6 * 0.1 * max(frame["x"].abs().max() for frame, _ in training)
    )
    assert len(fit.model.locations) == 2
    assert len(fit.model.transitions) == 2
    for index, (frame, truth) in enumerate(training):
        segments = [s for s in fit.segments if s.trace_index == index]
        assert segments[0].start == 0
        assert segments[-1].stop == len(frame)
        assert [s.start for s in segments[1:]] == list(
            range(21, len(frame), 20)
        )
        for segment in segments:
            assert np.all(
                truth[segment.start : segment.stop] == segment.location_id
            )
    expected = [[0.5, -0.1, 0.2, 0.1], [-0.3, 0.15, 0.3, -0.2]]
    for model, coefficients in zip(fit.model.locations, expected, strict=True):
        np.testing.assert_allclose(
            model.coefficients[0], coefficients, atol=1e-12
        )
    for transition in fit.model.transitions:
        # Same output, different input: an output-only guard cannot do this.
        command = 2 if transition.source == 0 else -2
        assert transition.guard.predict(
            [[0.0, command], [0.0, 0.0]]
        ).tolist() == [True, False]
        sign = 1 if transition.destination == 1 else -1
        expected = [[0.7, 0.2, 0.1, sign], [0.4, -0.1, 0.1, 0.3 * sign]]
        for predictor, coefficients in zip(
            transition.reset_predictors, expected, strict=True
        ):
            np.testing.assert_allclose(
                predictor.coefficients[0], coefficients, atol=1e-12
            )


def test_rollout_is_recursive_and_uses_guard_at_current_row(identified):
    fit, _ = identified
    frame, truth = sampled_system(93)
    result = fit.model.rollout(
        frame.select("time", "u"),
        output_history=frame["x"][:2].to_numpy()[:, None],
        initial_location=0,
    )
    np.testing.assert_allclose(result["x"], frame["x"], atol=1e-11)
    assert result["location_id"][0] is None
    np.testing.assert_array_equal(result["location_id"][1:], truth[1:])
    # Unselected future outputs are not teacher forcing, even when present.
    poisoned = frame.with_columns(pl.lit(float("nan")).alias("x"))
    repeated = fit.model.rollout(
        poisoned,
        output_history=frame["x"][:2].to_numpy()[:, None],
        initial_location=0,
    )
    assert repeated.equals(result)
    changed = fit.model.rollout(
        frame, output_history=[[2.0], [-1.0]], initial_location=0
    )
    assert changed["x"][2] != result["x"][2]
    with pytest.raises(ValueError, match="exactly"):
        fit.model.rollout(frame, output_history=[[1.0]], initial_location=0)
    with pytest.raises(ValueError, match="finite"):
        fit.model.rollout(
            frame, output_history=[[np.nan], [1.0]], initial_location=0
        )


def test_ambiguous_guards_and_interrupted_reset_fail(identified):
    fit, _ = identified
    frame, _ = sampled_system(10)
    transition = fit.model.transitions[0]
    duplicated = replace(
        fit.model, transitions=(*fit.model.transitions, transition)
    )
    with pytest.raises(ValueError, match="Multiple enabled"):
        duplicated.rollout(
            frame, output_history=[[0.0], [0.0]], initial_location=0
        )
    always = SVC(kernel="linear", C=1e4).fit(
        [[0.0, 0.0], [0.0, 100.0]], [True, False]
    )
    transitions = tuple(
        replace(t, guard=always) for t in fit.model.transitions
    )
    interrupted = replace(fit.model, transitions=transitions)
    with pytest.raises(ValueError, match="interrupts reset"):
        interrupted.rollout(
            frame, output_history=[[0.0], [0.0]], initial_location=0
        )


def test_clustering_checks_full_union_and_reconsiders_rejected_pairs():
    def constant(value, size=10):
        return np.ones((size, 1, 1)), np.full((size, 1), value)

    # Adjacent pairs fit, but their connected component does not.
    assert _cluster([constant(-1), constant(0), constant(1)], 0.6) == [
        [0, 1],
        [2],
    ]
    # A previously rejected pair becomes eligible after pooling more evidence.
    assert _cluster(
        [constant(-1, 1), constant(1, 1), constant(-1, 100)], 0.8
    ) == [[0, 2, 1]]


def test_same_location_reset_fails_but_independent_traces_can_share_location():
    x = np.empty(80)
    x[0] = 0
    for t in range(1, len(x)):
        x[t] = -3 if t % 20 == 0 else x[t - 1] + 0.1
    frame = pl.DataFrame({"time": np.arange(len(x)) * 0.1, "x": x})
    kwargs = {
        "input_features": [],
        "output_features": ["x"],
        "time_column": "time",
        "order": 1,
        "window": 5,
        "guard_kernel": "linear",
        "guard_c": 1.0,
        "guard_gamma": "scale",
    }
    # Separate traces may share dynamics without any cross-trace reset.
    independent = [frame.slice(start, 20) for start in range(0, 80, 20)]
    fit = fit_dainarx(independent, **kwargs)
    assert len(fit.segments) == 4
    assert {segment.location_id for segment in fit.segments} == {0}
    assert not fit.model.transitions
    with pytest.raises(
        ValueError, match=r"Trace 0.*row 20.*unsupported same-location reset"
    ):
        fit_dainarx([frame], **kwargs)


def test_order_one_seed_only_rollout_and_empty_input_boundary():
    frame = pl.DataFrame({"time": np.arange(10) * 0.1, "x": np.arange(10)})
    model = fit_dainarx(
        [frame],
        input_features=[],
        output_features=["x"],
        time_column="time",
        order=1,
        window=5,
        guard_kernel="linear",
        guard_c=1.0,
        guard_gamma="scale",
    ).model
    seed = frame.select("time").head(1)
    result = model.rollout(seed, output_history=[[7.5]], initial_location=0)
    assert result.to_dict(as_series=False) == {
        "time": [0.0],
        "x": [7.5],
        "location_id": [0],
    }
    with pytest.raises(ValueError, match="timestamp"):
        model.rollout(seed.head(0), output_history=[[7.5]], initial_location=0)
    with pytest.raises(ValueError, match="at least two timestamps"):
        fit_dainarx(
            [frame.head(1)],
            input_features=[],
            output_features=["x"],
            time_column="time",
            order=1,
            window=5,
            guard_kernel="linear",
            guard_c=1.0,
            guard_gamma="scale",
        )


def test_threshold_controls_screening_and_full_segment_acceptance():
    frame = pl.DataFrame(
        {
            "time": np.arange(100) * 0.1,
            "x": 100 + 0.001 * np.arange(100) ** 2,
        }
    )
    kwargs = {
        "input_features": [],
        "output_features": ["x"],
        "time_column": "time",
        "order": 1,
        "window": 5,
        "guard_kernel": "linear",
        "guard_c": 1.0,
        "guard_gamma": "scale",
    }
    with pytest.raises(ValueError, match="screening RMSE"):
        fit_dainarx([frame], **kwargs)
    # Windows pass, but the full segment must still meet the same threshold.
    with pytest.raises(
        ValueError, match=r"segment \[0, 100\).*screening RMSE"
    ):
        fit_dainarx([frame], eta=0.01, **kwargs)
    fit = fit_dainarx([frame], eta=0.1, **kwargs)
    assert fit.eta == 0.1
    assert len(fit.segments) == len(fit.model.locations) == 1
    assert 0.01 < fit.model.locations[0].rmse <= fit.eta


@pytest.mark.parametrize("eta", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_threshold_fails(eta):
    frame, _ = sampled_system(0)
    with pytest.raises(ValueError, match="eta must be finite and nonnegative"):
        fit_dainarx(
            [frame],
            input_features=["u"],
            output_features=["x"],
            time_column="time",
            order=2,
            window=8,
            guard_kernel="linear",
            guard_c=1e4,
            guard_gamma="scale",
            eta=eta,
        )


def test_zero_threshold_is_accepted_and_not_replaced_by_default():
    frame, _ = sampled_system(0)
    kwargs = {
        "input_features": ["u"],
        "output_features": ["x"],
        "time_column": "time",
        "order": 2,
        "window": 8,
        "guard_kernel": "linear",
        "guard_c": 1e4,
        "guard_gamma": "scale",
    }
    zeros = frame.with_columns(pl.lit(0.0).alias("x"))
    for eta in (None, 0.0):
        fit = fit_dainarx([zeros], eta=eta, **kwargs)
        assert fit.eta == fit.model.locations[0].rmse == 0.0
        assert len(fit.model.locations) == 1
    # Nonzero floating-point residuals pass automatically but not at zero.
    fit_dainarx([frame], **kwargs)
    with pytest.raises(ValueError, match=r"screening RMSE.*exceeds eta 0"):
        fit_dainarx([frame], eta=0.0, **kwargs)


def test_short_terminal_segment_and_insufficient_reset_evidence_fail():
    frame, _ = sampled_system(0, length=42)
    kwargs = {
        "input_features": ["u"],
        "output_features": ["x"],
        "time_column": "time",
        "order": 2,
        "window": 8,
        "guard_kernel": "linear",
        "guard_c": 1e4,
        "guard_gamma": "scale",
    }
    with pytest.raises(ValueError, match="insufficient targets for screening"):
        fit_dainarx([frame], **kwargs)
    frame, _ = sampled_system(0, length=60)
    with pytest.raises(
        ValueError, match="reset phase 1: insufficient targets"
    ):
        fit_dainarx([frame], **kwargs)
    kwargs["window"] = 6
    with pytest.raises(ValueError, match="coefficient columns"):
        fit_dainarx([frame], **kwargs)
