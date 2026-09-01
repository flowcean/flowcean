"""Behavioral tests for HyDRA prediction, routing, and simulation."""

from collections.abc import Callable
from typing import override

import numpy as np
import polars as pl
import pytest

from flowcean.core import Model
from flowcean.hybrid import (
    HybridDecisionTreeLearner,
    HyDRAModel,
    HyDRATraceSchema,
    SelectorFeatureConfig,
    Trace,
    compare_state_traces,
)


class DerivativeModel(Model):
    """Test flow model backed by a deterministic row expression."""

    def __init__(
        self,
        function: Callable[[pl.DataFrame], np.ndarray],
        output: str = "dx",
    ) -> None:
        self.function = function
        self.output = output

    @override
    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        return pl.DataFrame({self.output: self.function(frame)}).lazy()


def _schema(*, with_input: bool = False) -> HyDRATraceSchema:
    return HyDRATraceSchema(
        time="time",
        state=("x",),
        derivative=("dx",),
        inputs=("u",) if with_input else (),
    )


def _single_mode_model(
    flow: DerivativeModel,
    *,
    with_input: bool = False,
) -> HyDRAModel:
    schema = _schema(with_input=with_input)
    return HyDRAModel(
        [flow],
        input_features=list(schema.input_features),
        output_features=["dx"],
        trace_schema=schema,
    )


def test_single_mode_batch_prediction_and_diagnostics() -> None:
    model = _single_mode_model(
        DerivativeModel(lambda frame: -2.0 * frame["x"].to_numpy()),
    )
    inputs = pl.DataFrame({"time": [0.0, 0.5, 1.0], "x": [1.0, -2.0, 3.0]})

    diagnostics = model.predict_with_diagnostics(inputs.lazy())

    np.testing.assert_allclose(diagnostics.outputs["dx"], [-2.0, 4.0, -6.0])
    assert diagnostics.row_indices == [0, 1, 2]
    assert [result.mode_id for result in diagnostics.selector_results] == [
        0,
        0,
        0,
    ]
    np.testing.assert_allclose(
        model.predict(inputs).collect()["dx"],
        [-2.0, 4.0, -6.0],
    )


def test_predict_next_state_and_simulate_match_exponential_solution() -> None:
    rate = -0.8
    model = _single_mode_model(
        DerivativeModel(lambda frame: rate * frame["x"].to_numpy()),
    )

    next_state = model.predict_next_state([1.5], t=0.25, dt=0.75)
    np.testing.assert_allclose(
        next_state,
        [1.5 * np.exp(rate * 0.75)],
        rtol=2e-7,
        atol=1e-9,
    )

    times = np.linspace(0.0, 2.0, 9)
    trace = model.simulate((0.0, 2.0), [1.5], sample_times=times)
    np.testing.assert_allclose(trace.t, times)
    np.testing.assert_allclose(
        trace.x[:, 0],
        1.5 * np.exp(rate * times),
        rtol=3e-7,
        atol=1e-9,
    )
    assert trace.location.tolist() == ["mode_0"] * times.size
    assert trace.events == ()
    assert trace.u is None


def test_simulation_uses_and_captures_external_inputs() -> None:
    model = _single_mode_model(
        DerivativeModel(lambda frame: frame["u"].to_numpy()),
        with_input=True,
    )

    def input_stream(time: float) -> np.ndarray:
        return np.array([2.0 + 0.0 * time])

    trace = model.simulate(
        (0.0, 1.0),
        [1.0],
        input_stream=input_stream,
        sample_dt=0.25,
    )

    np.testing.assert_allclose(trace.x[:, 0], 1.0 + 2.0 * trace.t, atol=1e-8)
    assert trace.u is not None
    np.testing.assert_allclose(trace.u, np.full((5, 1), 2.0))


def test_model_reports_relevant_configuration_and_simulation_errors() -> None:
    flow = DerivativeModel(lambda frame: np.zeros(frame.height))
    without_schema = HyDRAModel(
        [flow],
        input_features=["time", "x"],
        output_features=["dx"],
    )
    with pytest.raises(ValueError, match="requires trace_schema"):
        without_schema.predict_next_state([1.0], t=0.0, dt=0.1)

    empty = HyDRAModel(
        [],
        input_features=["time", "x"],
        output_features=["dx"],
        trace_schema=_schema(),
    )
    with pytest.raises(ValueError, match="no learned modes"):
        empty.predict(pl.DataFrame({"time": [0.0], "x": [1.0]})).collect()

    model = _single_mode_model(flow)
    with pytest.raises(ValueError, match="state dimension"):
        model.predict_next_state([1.0, 2.0], t=0.0, dt=0.1)
    with pytest.raises(ValueError, match="greater than zero"):
        model.predict_next_state([1.0], t=0.0, dt=0.0)
    with pytest.raises(ValueError, match="Exactly one"):
        model.simulate((0.0, 1.0), [1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        model.simulate(
            (0.0, 1.0),
            [1.0],
            sample_times=[0.0, 0.5, 0.5, 1.0],
        )

    input_model = _single_mode_model(flow, with_input=True)
    with pytest.raises(ValueError, match="input_stream is required"):
        input_model.predict_next_state([1.0], t=0.0, dt=0.1)


def test_multi_mode_batch_prediction_routes_rows_with_decision_tree() -> None:
    negative_flow = DerivativeModel(lambda frame: np.full(frame.height, -1.0))
    positive_flow = DerivativeModel(lambda frame: np.full(frame.height, 2.0))
    selector_learner = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_features=("x",)),
        max_depth=1,
        random_state=0,
    )
    selector = selector_learner.learn_from_traces(
        [
            pl.DataFrame(
                {
                    "x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
                    "mode": [0, 0, 0, 1, 1, 1],
                },
            ),
        ],
        mode_to_flow={0: negative_flow, 1: positive_flow},
    )
    model = HyDRAModel(
        [negative_flow, positive_flow],
        input_features=["time", "x"],
        output_features=["dx"],
        selector=selector,
        trace_schema=_schema(),
    )
    inputs = pl.DataFrame(
        {"time": [0.0, 0.0, 0.0, 0.0], "x": [2.0, -2.0, 3.0, -1.0]},
    )

    diagnostics = model.predict_with_diagnostics(inputs)

    assert [result.mode_id for result in diagnostics.selector_results] == [
        1,
        0,
        1,
        0,
    ]
    np.testing.assert_allclose(
        diagnostics.outputs["dx"],
        [2.0, -1.0, 2.0, -1.0],
    )
    assert diagnostics.row_indices == [0, 1, 2, 3]


def _trace(times: list[float], states: list[list[float]]) -> Trace:
    return Trace(
        t=np.asarray(times),
        x=np.asarray(states),
        location=np.asarray(["mode"] * len(times), dtype=object),
        events=(),
    )


def test_compare_state_traces_calculates_elementwise_and_summary_metrics() -> (
    None
):
    reference = _trace([0.0, 1.0], [[1.0, 2.0], [3.0, 4.0]])
    predicted = _trace([0.0, 1.0], [[0.0, 4.0], [3.0, 2.0]])

    comparison = compare_state_traces(reference, predicted)

    np.testing.assert_allclose(
        comparison.absolute_error,
        [[1.0, 2.0], [0.0, 2.0]],
    )
    assert comparison.mae == pytest.approx(1.25)
    assert comparison.rmse == pytest.approx(1.5)
    assert comparison.max_error == pytest.approx(2.0)


def test_compare_state_traces_validates_grid_and_shape() -> None:
    reference = _trace([0.0, 1.0], [[1.0], [2.0]])

    with pytest.raises(ValueError, match="time grids must match"):
        compare_state_traces(reference, _trace([0.0, 1.1], [[1.0], [2.0]]))
    with pytest.raises(ValueError, match="state shapes must match"):
        compare_state_traces(
            reference,
            _trace([0.0, 1.0], [[1.0, 2.0], [2.0, 3.0]]),
        )
