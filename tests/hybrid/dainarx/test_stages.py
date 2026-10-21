from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from flowcean.hybrid import dainarx
from flowcean.hybrid.dainarx import (
    DainarxDynamics,
    DainarxModel,
    fit_dainarx,
    fit_dainarx_dynamics,
    fit_dainarx_transitions,
)

from .conftest import DynamicsSettings, GuardSettings, sampled_system

DYNAMICS: DynamicsSettings = {
    "input_features": ["u"],
    "output_features": ["x"],
    "time_column": "time",
    "order": 2,
    "window": 8,
}
GUARDS: GuardSettings = {
    "guard_kernel": "linear",
    "guard_c": 1e4,
    "guard_gamma": "scale",
}


@pytest.mark.parametrize("empty_column", ["u", "x", "time"])
def test_empty_selected_column_names_support_fit_and_rollout(empty_column):
    names = {
        name: "" if name == empty_column else name
        for name in ("u", "x", "time")
    }
    frame = pl.DataFrame(
        {
            "time": np.arange(20) * 0.1,
            "x": np.arange(20) * 0.2,
            "u": np.zeros(20),
        }
    ).rename({empty_column: ""})
    fit = fit_dainarx(
        [frame],
        input_features=[names["u"]],
        output_features=[names["x"]],
        time_column=names["time"],
        order=1,
        window=6,
        **GUARDS,
    )
    assert len(fit.model.locations) == 1
    assert not fit.model.transitions
    result = fit.model.rollout(
        frame, output_history=[[0.0]], initial_location=0
    )
    np.testing.assert_allclose(
        result[names["x"]], frame[names["x"]], atol=1e-12
    )
    assert result[names["time"]].equals(frame[names["time"]])


def test_staged_and_full_fit_agree_on_discovery_guards_phases_and_rollout():
    class SquaredInput:
        __hash__ = object.__hash__

        def __call__(self, history, current_input):
            return np.full(history.shape[1], current_input[0] ** 2)

        def __eq__(self, other):
            raise AssertionError(
                "Feature functions must not be compared by equality"
            )

    terms = (SquaredInput(),)
    traces = [sampled_system(seed)[0] for seed in range(4)]
    dynamics = dainarx.fit_dainarx_dynamics(
        traces, nonlinear_terms=terms, **DYNAMICS
    )
    transitions = fit_dainarx_transitions(traces, dynamics, **GUARDS)
    staged = DainarxModel(dynamics.locations, transitions)
    full = fit_dainarx(traces, nonlinear_terms=terms, **DYNAMICS, **GUARDS)
    assert isinstance(dynamics, DainarxDynamics)
    assert dynamics.segments == full.segments
    assert dynamics.eta == full.eta
    for left, right in zip(
        staged.locations, full.model.locations, strict=True
    ):
        np.testing.assert_array_equal(left.coefficients, right.coefficients)
        assert left.rmse == right.rmse
    for left, right in zip(transitions, full.model.transitions, strict=True):
        assert (left.source, left.destination) == (
            right.source,
            right.destination,
        )
        for a, b in zip(
            left.reset_predictors, right.reset_predictors, strict=True
        ):
            np.testing.assert_array_equal(a.coefficients, b.coefficients)
            assert a.rmse == b.rmse
        for trace in traces:
            rows = trace.select("x", "u").to_numpy()
            np.testing.assert_array_equal(
                left.guard.predict(rows), right.guard.predict(rows)
            )
    held_out, _ = sampled_system(93)
    history = held_out["x"][:2].to_numpy()[:, None]
    result = staged.rollout(
        held_out, output_history=history, initial_location=0
    )
    assert result.equals(
        full.model.rollout(
            held_out, output_history=history, initial_location=0
        )
    )
    np.testing.assert_allclose(result["x"], held_out["x"], atol=1e-11)


def test_projection_preserves_original_rows_columns_and_independent_trace_geometry():
    training = [sampled_system(2), sampled_system(3, length=180)]
    traces = [
        frame.with_columns(
            (pl.col("time") + index * 100).alias("time"),
            pl.lit(f"trace-{index}").alias("metadata"),
            pl.Series("count", np.arange(frame.height), dtype=pl.UInt32),
        ).select("metadata", "u", "count", "time", "x")
        for index, (frame, _) in enumerate(training)
    ]
    dynamics = fit_dainarx_dynamics(traces, **DYNAMICS)
    assert dynamics.trace_lengths == (200, 180)
    labels = dynamics.to_location_ids()
    frames = dynamics.to_labeled_frames(traces)
    for index, (frame, annotated, (_, truth)) in enumerate(
        zip(traces, frames, training, strict=True)
    ):
        assert annotated is not frame
        assert annotated.columns == [*frame.columns, "location_id"]
        assert annotated.select(frame.columns).equals(frame)
        assert annotated.schema == {**frame.schema, "location_id": pl.Int64}
        assert labels[index].dtype == np.int64
        np.testing.assert_array_equal(labels[index], truth)
        np.testing.assert_array_equal(annotated["location_id"], truth)
        for segment in dynamics.segments:
            if segment.trace_index == index:
                np.testing.assert_array_equal(
                    labels[index][segment.start : segment.start + 2],
                    np.full(2, segment.location_id),
                )
    labels[0][:] = -1
    frames[0].replace_column(0, pl.Series("metadata", ["changed"] * 200))
    np.testing.assert_array_equal(
        dynamics.to_location_ids()[0], training[0][1]
    )
    assert traces[0]["metadata"][0] == "trace-0"
    assert dynamics.to_labeled_frames(traces)[0]["metadata"][0] == "trace-0"

    for invalid in (
        traces[:1],
        [traces[0].head(199), traces[1]],
        [None, traces[1]],
    ):
        with pytest.raises(ValueError, match=r"original|Polars DataFrame"):
            # Deliberately includes a non-DataFrame to check runtime validation.
            dynamics.to_labeled_frames(invalid)  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match=r"original|Polars DataFrame"):
            # Deliberately includes a non-DataFrame to check runtime validation.
            fit_dainarx_transitions(invalid, dynamics, **GUARDS)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="reserved"):
        dynamics.to_labeled_frames(frames)
    for invalid, message in (
        ([traces[0].reverse(), traces[1]], "strictly increasing"),
        (
            [
                traces[0].with_columns((pl.col("time") * 2).alias("time")),
                traces[1],
            ],
            "uniform",
        ),
        (
            [
                traces[0].with_columns(pl.lit("not numeric").alias("u")),
                traces[1],
            ],
            "numeric",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            fit_dainarx_transitions(invalid, dynamics, **GUARDS)


def test_discovery_survives_insufficient_transition_evidence():
    frame, truth = sampled_system(0, length=60)
    dynamics = fit_dainarx_dynamics([frame], **DYNAMICS)
    assert len(dynamics.locations) == 2
    np.testing.assert_array_equal(dynamics.to_location_ids()[0], truth)
    with pytest.raises(
        ValueError, match="reset phase 1: insufficient targets"
    ):
        fit_dainarx_transitions([frame], dynamics, **GUARDS)
    assert dynamics.to_labeled_frames([frame])[0].height == frame.height


def test_discovery_retains_same_location_boundaries_that_transitions_reject():
    x = np.empty(80)
    x[0] = 0
    for t in range(1, len(x)):
        x[t] = -3 if t % 20 == 0 else x[t - 1] + 0.1
    frame = pl.DataFrame({"time": np.arange(len(x)) * 0.1, "x": x})
    dynamics = fit_dainarx_dynamics(
        [frame],
        input_features=[],
        output_features=["x"],
        time_column="time",
        order=1,
        window=5,
    )
    assert len(dynamics.locations) == 1
    assert [(s.start, s.stop) for s in dynamics.segments] == [
        (0, 20),
        (20, 40),
        (40, 60),
        (60, 80),
    ]
    np.testing.assert_array_equal(dynamics.to_location_ids()[0], np.zeros(80))
    with pytest.raises(
        ValueError, match=r"Trace 0.*row 20.*unsupported same-location reset"
    ):
        fit_dainarx_transitions([frame], dynamics, **GUARDS)


def test_discovery_invariants_reject_unusable_templates_and_incomplete_membership(
    identified,
):
    fit, training = identified
    dynamics = DainarxDynamics(
        fit.model.locations,
        fit.segments,
        tuple(len(frame) for frame, _ in training),
        fit.eta,
    )
    first = dynamics.segments[0]
    invalid_segments = (
        dynamics.segments[1:],
        (replace(first, start=1), *dynamics.segments[1:]),
        (replace(first, stop=first.stop + 1), *dynamics.segments[1:]),
        (*dynamics.segments[:-1], replace(dynamics.segments[-1], stop=201)),
        (replace(first, location_id=2), *dynamics.segments[1:]),
    )
    for segments in invalid_segments:
        with pytest.raises(ValueError, match="Segments"):
            replace(dynamics, segments=segments)
    for template in (
        replace(dynamics.locations[1], dt=0),
        replace(dynamics.locations[1], order=1),
        replace(dynamics.locations[1], input_features=("other",)),
        replace(dynamics.locations[1], coefficients=np.zeros((1, 3))),
        replace(dynamics.locations[1], nonlinear_terms=(None,)),
    ):
        with pytest.raises(ValueError, match="Location"):
            replace(dynamics, locations=(dynamics.locations[0], template))
    with pytest.raises(ValueError, match="eta"):
        replace(dynamics, eta=np.nan)


def test_reset_fits_keep_the_discovery_rmse_acceptance_gate():
    traces = []
    for seed in range(4):
        frame, truth = sampled_system(seed)
        x, u = frame["x"].to_numpy().copy(), frame["u"].to_numpy()
        for t in range(2, len(x)):
            sign = 1 if truth[t] == 1 else -1
            if t % 20 == 1:
                # Vary only the first post-switch predictor, then propagate
                # its outputs so ordinary location dynamics remain exact.
                x[t] = (
                    0.7 * x[t - 1]
                    + 0.2 * x[t - 2]
                    + 0.1 * u[t]
                    + sign
                    + 0.01 * (-1) ** (seed + t // 40)
                )
            elif t % 20 == 2 and t > 2:
                x[t] = (
                    0.4 * x[t - 1] - 0.1 * x[t - 2] + 0.1 * u[t] + 0.3 * sign
                )
            elif truth[t] == 0:
                x[t] = 0.5 * x[t - 1] - 0.1 * x[t - 2] + 0.2 * u[t] + 0.1
            else:
                x[t] = -0.3 * x[t - 1] + 0.15 * x[t - 2] + 0.3 * u[t] - 0.2
        traces.append(frame.with_columns(pl.Series("x", x)))
    dynamics = fit_dainarx_dynamics(traces, **DYNAMICS)
    assert all(model.rmse <= dynamics.eta for model in dynamics.locations)
    with pytest.raises(
        ValueError, match=r"reset phase 1: RMSE .* exceeds eta"
    ):
        fit_dainarx_transitions(traces, dynamics, **GUARDS)
    with pytest.raises(
        ValueError, match=r"reset phase 1: RMSE .* exceeds eta"
    ):
        fit_dainarx(traces, **DYNAMICS, **GUARDS)
