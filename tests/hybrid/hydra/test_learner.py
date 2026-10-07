"""Behavioral tests for independent, fresh-batch HyDRA discovery."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import override

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.core import Model, SupervisedLearner
from flowcean.hybrid.hydra import (
    HybridDecisionTreeLearner,
    HyDRACallback,
    HyDRAIdentificationError,
    HyDRALearner,
    HyDRAModel,
    LearnedFlow,
    LearnedFlows,
    SelectorFeatureConfig,
    TraceSegment,
)


class LinearFeatureModel(Model):
    def __init__(
        self, feature: str, output: str, slope: float, intercept: float
    ) -> None:
        self.feature = feature
        self.output = output
        self.slope = slope
        self.intercept = intercept

    def __deepcopy__(self, memo: dict[int, object]) -> LinearFeatureModel:
        raise TypeError("Backend models cannot be deep-copied")

    @override
    def _predict(
        self, input_features: pl.DataFrame | pl.LazyFrame
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        return pl.DataFrame(
            {
                self.output: self.slope * frame[self.feature].to_numpy()
                + self.intercept
            }
        ).lazy()


class LinearLearner(SupervisedLearner):
    """Batch least squares, recording each instance's single supplied fit."""

    def __init__(self, feature: str = "x") -> None:
        self.feature = feature
        self.fits: list[pl.DataFrame] = []
        self.model: LinearFeatureModel | None = None

    @override
    def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> Model:
        assert not self.fits, (
            "Each factory instance must be learned exactly once"
        )
        x, y = pl.collect_all([inputs, outputs])
        self.fits.append(x.hstack(y))
        design = np.column_stack(
            (x[self.feature].to_numpy(), np.ones(x.height))
        )
        slope, intercept = np.linalg.lstsq(
            design, y.to_series().to_numpy(), rcond=None
        )[0]
        self.model = LinearFeatureModel(
            self.feature, y.columns[0], float(slope), float(intercept)
        )
        return self.model


class RecordingFactory:
    def __init__(self, feature: str = "x") -> None:
        self.feature = feature
        self.instances: list[LinearLearner] = []

    def __call__(self) -> LinearLearner:
        instance = LinearLearner(self.feature)
        self.instances.append(instance)
        return instance

    @property
    def fits(self) -> list[pl.DataFrame]:
        return [instance.fits[0] for instance in self.instances]


class RecordingCallback(HyDRACallback):
    def __init__(self) -> None:
        self.candidates: list[tuple[TraceSegment, float]] = []
        self.finished: list[LearnedFlows] = []
        self.groupings: list[tuple[int, tuple[TraceSegment, ...], int]] = []
        self.finalized: list[tuple[int, LearnedFlow]] = []

    def grouping_evaluated(
        self,
        *,
        flow_id: int,
        accepted_segments: tuple[TraceSegment, ...],
        considered_count: int,
    ) -> None:
        self.groupings.append((flow_id, accepted_segments, considered_count))

    def flow_finalized(self, *, flow_id: int, flow: LearnedFlow) -> None:
        self.finalized.append((flow_id, flow))

    def candidate_window_evaluated(
        self, *, segment: TraceSegment, fit: float
    ) -> None:
        self.candidates.append((segment, fit))

    def finish(self, result: LearnedFlows) -> None:
        self.finished.append(result)


def discover(
    learner: HyDRALearner, frames: list[pl.DataFrame]
) -> LearnedFlows:
    return learner.learn(frames, input_features=["x"], output_features=["dx"])


def test_identification_error_exposes_failing_segment_and_message() -> None:
    segment = TraceSegment(2, 3, 7)
    error = HyDRAIdentificationError(segment, "Candidate accuracy failed.")
    assert isinstance(error, RuntimeError)
    assert error.segment is segment
    assert str(error) == "Trace 2 [3, 7): Candidate accuracy failed."


@pytest.mark.parametrize("threshold", [-0.1, np.nan, np.inf, -np.inf])
def test_learner_validates_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        HyDRALearner(LinearLearner, threshold=threshold)


@pytest.mark.parametrize("width", [0, -1, 1.5, True])
@pytest.mark.parametrize("name", ["start_width", "step_width"])
def test_learner_validates_widths(name: str, width: int) -> None:
    if name == "start_width":
        with pytest.raises(ValueError, match="start_width must be positive"):
            HyDRALearner(LinearLearner, threshold=0.1, start_width=width)
    else:
        with pytest.raises(ValueError, match="step_width must be positive"):
            HyDRALearner(LinearLearner, threshold=0.1, step_width=width)


def test_generic_single_flow_regression_preserves_feature_order() -> None:
    frame = pl.DataFrame(
        {
            "voltage": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "current": [-3.0, -1.0, 1.0, 3.0, 5.0],
            "temperature": [20.0, 21.0, 19.0, 23.0, 22.0],
        }
    )
    factory = RecordingFactory("voltage")
    result = HyDRALearner(factory, threshold=1e-10, start_width=3).learn(
        [frame],
        input_features=["temperature", "voltage"],
        output_features=["current"],
    )
    assert all(
        fit.columns == ["temperature", "voltage", "current"]
        for fit in factory.fits
    )
    assert result.input_features == ("temperature", "voltage")
    assert result.output_features == ("current",)
    assert result.trace_lengths == (frame.height,)
    assert result.to_flow_ids()[0].tolist() == [0] * frame.height
    assert len(result.flows) == 1
    assert result.flows[0].segments == (TraceSegment(0, 0, frame.height),)


def test_multitrace_reuse_refit_rows_and_selector_composition() -> None:
    factory = RecordingFactory()
    callback = RecordingCallback()
    frames = [
        pl.DataFrame(
            {
                "x": [0, 1, 2, 3],
                "dx": [0, 1, 102, 103],
                "metadata": ["a", "b", "c", "d"],
                "time": [3, 1, 1, -5],
            }
        ),
        pl.DataFrame(
            {
                "x": [10.0, 11.0, 12.0, 13.0],
                "dx": [110.0, 111.0, 12.0, 13.0],
                "metadata": [None, None, None, None],
                "other": [True] * 4,
            }
        ),
    ]
    result = discover(
        HyDRALearner(
            factory,
            threshold=0.01,
            start_width=2,
            step_width=2,
            callback=callback,
        ),
        frames,
    )
    assert [ids.tolist() for ids in result.to_flow_ids()] == [
        [0, 0, 1, 1],
        [1, 1, 0, 0],
    ]
    # Two pending rows in each active trace contribute to the second group.
    assert callback.groupings[1][2] == 4
    # The rejected expanding window is fitted independently, then excluded
    # from the final accepted-row fit. No window crosses a trace boundary.
    assert [fit["x"].to_list() for fit in factory.fits] == [
        [0, 1],
        [0, 1, 2, 3],
        [0, 1, 12, 13],
        [2, 3],
        [2, 3, 10, 11],
    ]
    flow_models = [flow.model for flow in result.flows]
    model = HyDRAModel(
        flow_models,
        input_features=result.input_features,
        output_features=result.output_features,
    )
    model.selector = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_features=("x",)),
        random_state=0,
    ).learn_from_traces(
        result.to_labeled_frames(frames),
        flow_models_by_id=dict(enumerate(flow_models)),
    )
    np.testing.assert_allclose(
        model.predict(frames[0].select("x")).collect()["dx"],
        frames[0]["dx"],
        atol=1e-12,
    )


def test_rejected_expansion_retains_noncopyable_candidate() -> None:
    factory = RecordingFactory()
    callback = RecordingCallback()
    learner = HyDRALearner(
        factory, threshold=0.1, start_width=2, step_width=2, callback=callback
    )
    candidate = learner._fit_candidate_flow(
        pl.DataFrame(
            {"x": [0.0, 1.0, 2.0, 3.0], "dx": [0.0, 1.0, 12.0, 13.0]}
        ),
        ["x"],
        ["dx"],
        trace_index=0,
        segment_start_index=7,
    )
    assert candidate is not None
    assert candidate is factory.instances[0].model
    assert [segment for segment, _ in callback.candidates] == [
        TraceSegment(0, 7, 9),
        TraceSegment(0, 7, 11),
    ]
    assert callback.candidates[1][1] > learner.threshold
    np.testing.assert_allclose(
        candidate.predict(pl.DataFrame({"x": [0.0, 1.0]})).collect()["dx"],
        [0.0, 1.0],
        atol=1e-12,
    )


def test_later_discovery_preserves_finalized_noncopyable_models() -> None:
    factory = RecordingFactory()
    learner = HyDRALearner(factory, threshold=0.1, start_width=10)
    first = discover(
        learner,
        [
            pl.DataFrame({"x": [0.0, 1.0], "dx": [0.0, 1.0]}),
            pl.DataFrame({"x": [2.0, 3.0], "dx": [12.0, 13.0]}),
        ],
    )
    assert first.flows[-1].model is factory.instances[-1].model
    probe = pl.DataFrame({"x": [2.0]})
    before = [
        flow.model.predict(probe).collect()["dx"] for flow in first.flows
    ]
    np.testing.assert_allclose(before, [[2.0], [12.0]], atol=1e-12)

    second = discover(
        learner, [pl.DataFrame({"x": [10.0, 11.0], "dx": [100.0, 110.0]})]
    )
    assert second.to_flow_ids()[0].tolist() == [0, 0]
    assert all(second.flows[0].model is not flow.model for flow in first.flows)
    np.testing.assert_allclose(
        second.flows[0].model.predict(probe).collect()["dx"],
        [20.0],
        atol=1e-12,
    )
    after = [flow.model.predict(probe).collect()["dx"] for flow in first.flows]
    np.testing.assert_allclose(after, before, atol=1e-12)


@pytest.mark.parametrize(
    "dtype",
    [pl.Int32, pl.Int64, pl.UInt32, pl.Float32, pl.Float64, pl.Decimal(20, 2)],
)
def test_selected_numeric_schemas_are_normalized_without_changing_traces(
    dtype: pl.DataType,
) -> None:
    frame = pl.DataFrame(
        [
            pl.Series("x", [0, 1, 2, 3]).cast(dtype),
            pl.Series("dx", [0, 2, 4, 6]).cast(dtype),
            pl.Series("phase", [0, 0, 1, 1], dtype=pl.Int8),
            pl.Series("metadata", ["a", "b", "c", "d"]),
        ]
    )
    original = frame.clone()
    factory = RecordingFactory()
    discover(HyDRALearner(factory, threshold=0.01, start_width=2), [frame])
    assert all(fit.dtypes == [pl.Float64, pl.Float64] for fit in factory.fits)
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("dtype", [pl.Decimal(20, 2), pl.Decimal(38, 18)])
@pytest.mark.parametrize("start_width", [2, 10])
@pytest.mark.parametrize("boundary", ["below", "exact", "above"])
def test_decimal_scoring_and_grouping_share_normalized_residuals(
    dtype: pl.DataType, start_width: int, boundary: str
) -> None:
    value = Decimal("100000000000000.01")
    predicted = Decimal("100000000000000.02")
    frame = pl.DataFrame(
        {
            "x": [0.0, 1.0],
            "dx": pl.Series([value, value], dtype=dtype),
        }
    )
    prediction = pl.DataFrame(
        {"dx": pl.Series([predicted, predicted], dtype=dtype)}
    )
    error = abs(
        frame["dx"].cast(pl.Float64)[0] - prediction["dx"].cast(pl.Float64)[0]
    )
    assert error > 0
    # Polars Float64 normalization defines scoring and grouping residuals.
    threshold = {
        "below": np.nextafter(error, -np.inf),
        "exact": error,
        "above": np.nextafter(error, np.inf),
    }[boundary]
    callback = RecordingCallback()

    learner = HyDRALearner(
        lambda: FixedLearner(FixedModel(lambda _: prediction)),
        threshold=threshold,
        start_width=start_width,
        callback=callback,
    )
    if boundary == "above":
        result = discover(learner, [frame])
        assert result.to_flow_ids()[0].tolist() == [0, 0]
        assert callback.groupings == [(0, (TraceSegment(0, 0, 2),), 2)]
    else:
        with pytest.raises(HyDRAIdentificationError) as caught:
            discover(learner, [frame])
        assert caught.value.segment == TraceSegment(0, 0, 2)
        if start_width == 2:
            assert "Candidate accuracy did not meet" in str(caught.value)
            assert not callback.groupings
        else:
            assert "grouping accepted no observations" in str(caught.value)
            assert callback.groupings == [(0, (), 2)]
    assert callback.candidates == (
        [(TraceSegment(0, 0, 2), error)] if start_width == 2 else []
    )


def test_short_windows_do_not_bridge_trace_boundaries() -> None:
    factory = RecordingFactory()
    result = discover(
        HyDRALearner(factory, threshold=0.01, start_width=10),
        [
            pl.DataFrame({"x": [0.0, 1.0], "dx": [0.0, 1.0]}),
            pl.DataFrame({"x": [2.0, 3.0], "dx": [2.0, 3.0]}),
        ],
    )
    assert len(result.flows) == 1
    assert [fit.height for fit in factory.fits] == [2, 4]


def test_later_trace_failure_after_prior_flow_finalization() -> None:
    callback = RecordingCallback()
    frames = [
        pl.DataFrame({"x": [0.0, 1.0, 2.0], "dx": [0.0, 1.0, 2.0]}),
        pl.DataFrame({"x": [3.0, 4.0, 5.0], "dx": [50.0, -50.0, 50.0]}),
        pl.DataFrame({"x": [6.0, 7.0, 8.0], "dx": [100.0, 100.0, 100.0]}),
    ]
    with pytest.raises(HyDRAIdentificationError) as caught:
        discover(
            HyDRALearner(
                LinearLearner,
                threshold=0.01,
                start_width=3,
                callback=callback,
            ),
            frames,
        )
    assert caught.value.segment == TraceSegment(1, 0, 3)
    assert "Trace 1 [0, 3)" in str(caught.value)
    assert "Candidate accuracy did not meet" in str(caught.value)
    assert len(callback.finalized) == 1
    assert callback.finalized[0][1].segments == (TraceSegment(0, 0, 3),)
    assert not callback.finished


@pytest.mark.parametrize("short", [False, True])
def test_zero_flow_discovery_raises(short: bool) -> None:
    callback = RecordingCallback()
    frame = pl.DataFrame({"x": [0.0, 1.0, 2.0], "dx": [1.0, 1.0, 1.0]})
    with pytest.raises(HyDRAIdentificationError) as caught:
        discover(
            HyDRALearner(
                LinearLearner,
                threshold=0,
                start_width=10 if short else 3,
                callback=callback,
            ),
            [frame],
        )
    assert caught.value.segment == TraceSegment(0, 0, 3)
    assert "Trace 0 [0, 3)" in str(caught.value)
    if short:
        assert "grouping accepted no observations" in str(caught.value)
        assert callback.groupings == [(0, (), 3)]
    else:
        assert "Candidate accuracy did not meet" in str(caught.value)
        assert callback.groupings == []
    assert not callback.finalized
    assert not callback.finished


@pytest.mark.parametrize(
    ("inputs", "outputs", "match"),
    [
        ([], ["dx"], "nonempty"),
        (["x", "x"], ["dx"], "unique"),
        (["x"], [], "nonempty"),
        (["x"], ["dx", "dx"], "unique"),
        (["x"], ["x"], "disjoint"),
        (["x"], ["dx", "other"], "single-output"),
        (["absent"], ["dx"], "missing column"),
        (["flow_id"], ["dx"], "reserved"),
    ],
)
def test_roles_are_validated(
    inputs: list[str], outputs: list[str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        HyDRALearner(LinearLearner, threshold=0.1).learn(
            [pl.DataFrame({"x": [0.0], "dx": [1.0]})],
            input_features=inputs,
            output_features=outputs,
        )


@pytest.mark.parametrize(
    ("frames", "match"),
    [
        ([], "nonempty trace"),
        (
            [pl.DataFrame(schema={"x": pl.Float64, "dx": pl.Float64})],
            "at least one row",
        ),
        (
            [pl.DataFrame({"x": [0.0], "dx": [1.0], "flow_id": [0]})],
            "reserved",
        ),
        ([pl.DataFrame({"x": ["bad"], "dx": [1.0]})], "numeric"),
        ([pl.DataFrame({"x": [True], "dx": [1.0]})], "numeric"),
        (
            [
                pl.DataFrame(
                    {"x": [None], "dx": [1.0]},
                    schema={"x": pl.Float64, "dx": pl.Float64},
                )
            ],
            "no nulls",
        ),
        ([pl.DataFrame({"x": [0.0], "dx": [np.nan]})], "finite"),
        ([pl.DataFrame({"x": [np.inf], "dx": [1.0]})], "finite"),
    ],
)
def test_frames_are_validated_before_backend(
    frames: list[pl.DataFrame], match: str
) -> None:
    factory = RecordingFactory()
    with pytest.raises(ValueError, match=match):
        discover(HyDRALearner(factory, threshold=0.1), frames)
    assert not factory.instances


class FixedModel(Model):
    def __init__(
        self, predict: Callable[[pl.DataFrame], pl.DataFrame]
    ) -> None:
        self.function = predict

    def _predict(
        self, input_features: pl.DataFrame | pl.LazyFrame
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        return self.function(frame).lazy()


class FixedLearner(SupervisedLearner):
    def __init__(self, model: Model) -> None:
        self.model = model

    def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> Model:
        return self.model


@pytest.mark.parametrize(
    ("prediction", "match"),
    [
        (pl.DataFrame({"dx": [1.0]}), "row count"),
        (pl.DataFrame({"wrong": [1.0, 1.0]}), "missing output"),
        (pl.DataFrame({"dx": ["bad", "bad"]}), "numeric"),
        (pl.DataFrame({"dx": [np.nan, 1.0]}), "finite"),
        (pl.DataFrame({"dx": [np.inf, 1.0]}), "finite"),
        (pl.DataFrame({"dx": [None, 1.0]}), "no nulls"),
    ],
)
@pytest.mark.parametrize("start_width", [2, 10])
def test_malformed_backend_predictions_raise(
    prediction: pl.DataFrame, match: str, start_width: int
) -> None:
    callback = RecordingCallback()
    with pytest.raises(ValueError, match=match):
        discover(
            HyDRALearner(
                lambda: FixedLearner(FixedModel(lambda _: prediction)),
                threshold=0.1,
                start_width=start_width,
                callback=callback,
            ),
            [pl.DataFrame({"x": [0.0, 1.0], "dx": [1.0, 1.0]})],
        )
    assert callback.finished == []


def test_malformed_grouping_prediction_raises_after_valid_candidate() -> None:
    calls = 0

    def factory() -> FixedLearner:
        nonlocal calls
        calls += 1
        if calls == 1:
            # Accurate on the first two-row candidate; malformed on grouping
            # over the whole trace after the next candidate is rejected.
            model = FixedModel(lambda _: pl.DataFrame({"dx": [1.0, 1.0]}))
        else:
            model = FixedModel(
                lambda frame: pl.DataFrame({"dx": [99.0] * frame.height})
            )
        return FixedLearner(model)

    with pytest.raises(ValueError, match="prediction row count"):
        discover(
            HyDRALearner(factory, threshold=0.1, start_width=2, step_width=2),
            [pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "dx": [1.0] * 4})],
        )


def test_final_fit_error_propagates_without_normal_finish() -> None:
    calls = 0
    callback = RecordingCallback()
    error = RuntimeError("final fit failed")

    class FailingFinalLearner(LinearLearner):
        def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> Model:
            raise error

    def factory() -> LinearLearner:
        nonlocal calls
        calls += 1
        return LinearLearner() if calls == 1 else FailingFinalLearner()

    with pytest.raises(RuntimeError) as caught:
        discover(
            HyDRALearner(
                factory, threshold=0.1, start_width=2, callback=callback
            ),
            [pl.DataFrame({"x": [0.0, 1.0], "dx": [0.0, 1.0]})],
        )
    assert caught.value is error
    assert calls == 2
    assert callback.finished == []
    assert callback.finalized == []
    assert len(callback.groupings) == 1


def test_complete_records_candidate_acceptance_not_final_refit_accuracy() -> (
    None
):
    calls = 0

    def factory() -> FixedLearner:
        nonlocal calls
        calls += 1
        value = 1.0 if calls == 1 else 99.0
        return FixedLearner(
            FixedModel(
                lambda frame: pl.DataFrame({"dx": [value] * frame.height})
            )
        )

    result = discover(
        HyDRALearner(factory, threshold=0.1, start_width=2),
        [pl.DataFrame({"x": [0.0, 1.0], "dx": [1.0, 1.0]})],
    )
    assert result.to_flow_ids()[0].tolist() == [0, 0]
    assert result.flows[0].segments == (TraceSegment(0, 0, 2),)
    assert (
        result.flows[0]
        .model.predict(pl.DataFrame({"x": [0.0]}))
        .collect()["dx"][0]
        == 99.0
    )


@pytest.mark.parametrize("stage", ["factory", "learn", "predict"])
def test_unexpected_errors_propagate(stage: str) -> None:
    error = RuntimeError("backend failure")

    class FailingLearner(LinearLearner):
        def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> Model:
            raise error

    def predict(_: pl.DataFrame) -> pl.DataFrame:
        raise error

    def factory() -> SupervisedLearner:
        if stage == "factory":
            raise error
        if stage == "learn":
            return FailingLearner()
        return FixedLearner(FixedModel(predict))

    callback = RecordingCallback()
    with pytest.raises(RuntimeError) as caught:
        discover(
            HyDRALearner(
                factory, threshold=0.1, start_width=2, callback=callback
            ),
            [pl.DataFrame({"x": [0.0, 1.0], "dx": [0.0, 1.0]})],
        )
    assert caught.value is error
    assert not callback.finalized
    assert not callback.finished


def test_fragmented_grouping_final_fit_rows_and_prediction_batches() -> None:
    fits: list[list[float]] = []
    predictions: list[list[float]] = []

    class ConstantLearner(SupervisedLearner):
        def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> Model:
            fits.append(inputs.collect()["x"].to_list())
            value = outputs.collect()["dx"][0]

            def predict(frame: pl.DataFrame) -> pl.DataFrame:
                predictions.append(frame["x"].to_list())
                return pl.DataFrame({"dx": [value] * frame.height})

            return FixedModel(predict)

    frames = [
        pl.DataFrame({"x": [0, 1], "dx": [0, 0]}),
        pl.DataFrame({"x": [2, 3, 4, 5], "dx": [0, 100, 0, 100]}),
    ]
    result = discover(
        HyDRALearner(ConstantLearner, threshold=0.1, start_width=10),
        frames,
    )
    assert [ids.tolist() for ids in result.to_flow_ids()] == [
        [0, 0],
        [0, 1, 0, 1],
    ]
    assert fits == [[0, 1], [0, 1, 2, 4], [3], [3, 5]]
    # Grouping skips completed trace 0 but retains the whole partial trace 1.
    # Neither short-segment candidate needs a window check.
    assert predictions == [[0, 1], [2, 3, 4, 5], [2, 3, 4, 5]]
