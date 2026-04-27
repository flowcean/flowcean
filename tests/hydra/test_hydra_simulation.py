from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import flowcean.hydra.model as hydra_model_module
from flowcean.core import Model, SupervisedIncrementalLearner
from flowcean.hydra.learner import HyDRALearner
from flowcean.hydra.model import HyDRAModel
from flowcean.hydra.schema import HyDRATraceSchema
from flowcean.ode import Trace


class NamedZeroModel(Model):
    def __init__(self, output_name: str) -> None:
        self.output_name = output_name

    def _predict(
        self,
        input_features: pl.LazyFrame | pl.DataFrame,
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        return pl.DataFrame({self.output_name: [0.0] * frame.height}).lazy()


class AlwaysDxZeroLearner(SupervisedIncrementalLearner):
    def learn_incremental(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> NamedZeroModel:
        del inputs
        output_columns = outputs.collect_schema().names()
        return NamedZeroModel(output_columns[0])


class LinearDerivativeModel(Model):
    def __init__(self, output_columns: tuple[str, ...]) -> None:
        self.output_columns = output_columns

    def _predict(
        self,
        input_features: pl.LazyFrame | pl.DataFrame,
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        values: dict[str, list[float]] = {}
        if self.output_columns == ("dx",):
            values["dx"] = [2.0] * frame.height
        else:
            values["dx"] = [1.0] * frame.height
            values["dv"] = [-0.5] * frame.height
        return pl.DataFrame(values).lazy()


class ThresholdSelector:
    def __init__(self) -> None:
        from flowcean.hydra.selector import SelectorFeatureConfig

        self.feature_config = SelectorFeatureConfig(state_columns=("x",))

    def predict_details(self, input_features: pl.DataFrame) -> list[object]:
        from flowcean.hydra.selector import ModePredictionResult

        mode_id = 1 if input_features["x"][0] >= 0.5 else 0
        return [ModePredictionResult(ready=True, mode_id=mode_id)]


class AlternatingSelector:
    def __init__(self) -> None:
        from flowcean.hydra.selector import SelectorFeatureConfig

        self.feature_config = SelectorFeatureConfig(state_columns=("x",))
        self.call_count = 0

    def predict_details(self, input_features: pl.DataFrame) -> list[object]:
        from flowcean.hydra.selector import ModePredictionResult

        del input_features
        mode_id = self.call_count % 2
        self.call_count += 1
        return [ModePredictionResult(ready=True, mode_id=mode_id)]


class ConstantDerivativeModel(Model):
    def __init__(self, value: float) -> None:
        self.value = value

    def _predict(
        self,
        input_features: pl.LazyFrame | pl.DataFrame,
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        return pl.DataFrame({"dx": [self.value] * frame.height}).lazy()


def test_hydra_trace_schema_rejects_duplicate_columns() -> None:
    with pytest.raises(ValueError, match="schema columns must be disjoint"):
        HyDRATraceSchema(
            time="t",
            state=("x",),
            derivative=("dx",),
            inputs=("x",),
        )


def test_hydra_trace_schema_validates_input_feature_coverage() -> None:
    schema = HyDRATraceSchema(
        time="t",
        state=("x",),
        derivative=("dx",),
        inputs=("u",),
    )

    schema.validate_input_features(["x", "t", "u"])

    with pytest.raises(
        ValueError,
        match="input_features must match trace schema",
    ):
        schema.validate_input_features(["x", "t"])

    with pytest.raises(
        ValueError,
        match="input_features must match trace schema",
    ):
        schema.validate_input_features(["x", "t", "u", "u"])


def test_hydra_trace_schema_validates_output_feature_order() -> None:
    schema = HyDRATraceSchema(
        time="t",
        state=("x", "v"),
        derivative=("dx", "dv"),
    )

    schema.validate_output_features(["dx", "dv"])

    with pytest.raises(
        ValueError,
        match="output_features must match schema\\.derivative",
    ):
        schema.validate_output_features(["dv", "dx"])


def test_hydra_learner_passes_trace_schema_to_model() -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    learner = HyDRALearner(
        regressor_factory=AlwaysDxZeroLearner,
        threshold=1.0,
        start_width=1,
        step_width=1,
        trace_schema=schema,
    )

    model = learner.learn(
        inputs=pl.DataFrame({"t": [0.0, 1.0], "x": [0.0, 1.0]}).lazy(),
        outputs=pl.DataFrame({"dx": [0.0, 0.0]}).lazy(),
    )

    assert model.trace_schema == schema


def test_hydra_learner_validates_schema_input_coverage() -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    learner = HyDRALearner(
        regressor_factory=AlwaysDxZeroLearner,
        threshold=1.0,
        trace_schema=schema,
    )

    with pytest.raises(
        ValueError,
        match="input_features must match trace schema",
    ):
        learner.learn(
            inputs=pl.DataFrame({"x": [0.0, 1.0]}).lazy(),
            outputs=pl.DataFrame({"dx": [0.0, 0.0]}).lazy(),
        )


def test_hydra_learner_rejects_multi_dimensional_schema_for_now() -> None:
    schema = HyDRATraceSchema(
        time="t",
        state=("x", "v"),
        derivative=("dx", "dv"),
    )
    learner = HyDRALearner(
        regressor_factory=AlwaysDxZeroLearner,
        threshold=1.0,
        trace_schema=schema,
    )

    with pytest.raises(ValueError, match="single-output"):
        learner.learn(
            inputs=pl.DataFrame(
                {"t": [0.0, 1.0], "x": [0.0, 1.0], "v": [1.0, 1.0]},
            ).lazy(),
            outputs=pl.DataFrame(
                {"dx": [0.0, 0.0], "dv": [1.0, 1.0]},
            ).lazy(),
        )


def test_hydra_model_predict_next_state_integrates_single_mode_derivative() -> (  # noqa: E501
    None
):
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    model = HyDRAModel(
        [LinearDerivativeModel(("dx",))],
        input_features=["t", "x"],
        output_features=["dx"],
        trace_schema=schema,
    )

    next_state = model.predict_next_state([1.0], t=0.0, dt=0.5)

    assert np.allclose(next_state, [2.0])


def test_hydra_model_predict_next_state_requires_schema() -> None:
    model = HyDRAModel(
        [LinearDerivativeModel(("dx",))],
        input_features=["t", "x"],
        output_features=["dx"],
    )

    with pytest.raises(ValueError, match="trace_schema"):
        model.predict_next_state([1.0], t=0.0, dt=0.5)


def test_hydra_model_predict_next_state_validates_state_width() -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    model = HyDRAModel(
        [LinearDerivativeModel(("dx",))],
        input_features=["t", "x"],
        output_features=["dx"],
        trace_schema=schema,
    )

    with pytest.raises(ValueError, match="state dimension"):
        model.predict_next_state([1.0, 2.0], t=0.0, dt=0.5)


def test_hydra_model_predict_next_state_forwards_solver_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    model = HyDRAModel(
        [LinearDerivativeModel(("dx",))],
        input_features=["t", "x"],
        output_features=["dx"],
        trace_schema=schema,
    )
    recorded_options: dict[str, object] = {}

    def fake_solve_ivp(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        recorded_options.update(kwargs)
        return SimpleNamespace(success=True, y=np.array([[2.0]]))

    monkeypatch.setattr(hydra_model_module, "solve_ivp", fake_solve_ivp)

    model.predict_next_state(
        [1.0],
        t=0.0,
        dt=0.5,
        rtol=1e-5,
        atol=1e-6,
        max_step=0.1,
    )

    assert recorded_options["rtol"] == 1e-5
    assert recorded_options["atol"] == 1e-6
    assert recorded_options["max_step"] == 0.1


def test_hydra_model_predict_next_state_rejects_non_1d_input_stream() -> None:
    schema = HyDRATraceSchema(
        time="t",
        state=("x",),
        derivative=("dx",),
        inputs=("u",),
    )
    model = HyDRAModel(
        [LinearDerivativeModel(("dx",))],
        input_features=["t", "x", "u"],
        output_features=["dx"],
        trace_schema=schema,
    )

    with pytest.raises(ValueError, match="1D array"):
        model.predict_next_state(
            [1.0],
            t=0.0,
            dt=0.5,
            input_stream=lambda _t: np.array([[1.0]]),
        )


def test_hydra_model_simulate_returns_trace_on_sample_grid() -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    model = HyDRAModel(
        [LinearDerivativeModel(("dx",))],
        input_features=["t", "x"],
        output_features=["dx"],
        trace_schema=schema,
    )

    trace = model.simulate((0.0, 1.0), [0.0], sample_dt=0.5)

    assert np.allclose(trace.t, [0.0, 0.5, 1.0])
    assert np.allclose(trace.x[:, 0], [0.0, 1.0, 2.0])
    assert trace.location.tolist() == ["mode_0", "mode_0", "mode_0"]
    assert trace.events == ()
    assert trace.dx is None


def test_manual_multi_output_hydra_model_simulates_multi_dimensional_state() -> (  # noqa: E501
    None
):
    schema = HyDRATraceSchema(
        time="t",
        state=("x", "v"),
        derivative=("dx", "dv"),
    )
    model = HyDRAModel(
        [LinearDerivativeModel(("dx", "dv"))],
        input_features=["t", "x", "v"],
        output_features=["dx", "dv"],
        trace_schema=schema,
    )

    trace = model.simulate((0.0, 1.0), [0.0, 2.0], sample_dt=0.5)

    assert np.allclose(trace.x[-1], [1.0, 1.5])


def test_hydra_simulation_selects_mode_once_per_interval() -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    model = HyDRAModel(
        [ConstantDerivativeModel(1.0), ConstantDerivativeModel(3.0)],
        input_features=["t", "x"],
        output_features=["dx"],
        selector=ThresholdSelector(),  # type: ignore[arg-type]
        trace_schema=schema,
    )

    trace = model.simulate((0.0, 1.0), [0.0], sample_dt=0.5)

    assert np.allclose(trace.x[:, 0], [0.0, 0.5, 2.0])
    assert trace.location.tolist() == ["mode_0", "mode_0", "mode_1"]


def test_hydra_simulation_uses_selected_mode_for_interval_dynamics() -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    selector = AlternatingSelector()
    model = HyDRAModel(
        [ConstantDerivativeModel(1.0), ConstantDerivativeModel(3.0)],
        input_features=["t", "x"],
        output_features=["dx"],
        selector=selector,  # type: ignore[arg-type]
        trace_schema=schema,
    )

    trace = model.simulate((0.0, 1.0), [0.0], sample_dt=0.5)

    assert np.allclose(trace.x[:, 0], [0.0, 0.5, 2.0])
    assert trace.location.tolist() == ["mode_0", "mode_0", "mode_1"]
    assert selector.call_count == 2


def test_hydra_model_simulate_captures_inputs_when_requested() -> None:
    schema = HyDRATraceSchema(
        time="t",
        state=("x",),
        derivative=("dx",),
        inputs=("u",),
    )
    model = HyDRAModel(
        [LinearDerivativeModel(("dx",))],
        input_features=["t", "x", "u"],
        output_features=["dx"],
        trace_schema=schema,
    )

    trace = model.simulate(
        (0.0, 1.0),
        [0.0],
        sample_times=[0.0, 0.5, 1.0],
        input_stream=lambda t: np.array([t + 1.0], dtype=float),
    )

    assert trace.u is not None
    assert np.allclose(trace.u[:, 0], [1.0, 1.5, 2.0])


def test_hydra_model_simulate_rejects_missing_sample_grid() -> None:
    schema = HyDRATraceSchema(time="t", state=("x",), derivative=("dx",))
    model = HyDRAModel(
        [LinearDerivativeModel(("dx",))],
        input_features=["t", "x"],
        output_features=["dx"],
        trace_schema=schema,
    )

    with pytest.raises(ValueError, match="sample_times or sample_dt"):
        model.simulate((0.0, 1.0), [0.0])


def test_compare_state_traces_reports_error_metrics() -> None:
    from flowcean.hydra import compare_state_traces

    reference = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[0.0], [2.0]], dtype=float),
        location=np.array(["a", "a"], dtype=object),
        events=(),
    )
    predicted = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[0.5], [1.0]], dtype=float),
        location=np.array(["mode_0", "mode_0"], dtype=object),
        events=(),
    )

    comparison = compare_state_traces(reference, predicted)

    assert np.allclose(comparison.absolute_error[:, 0], [0.5, 1.0])
    assert comparison.mae == pytest.approx(0.75)
    assert comparison.rmse == pytest.approx(np.sqrt((0.25 + 1.0) / 2.0))
    assert comparison.max_error == pytest.approx(1.0)


def test_compare_state_traces_allows_tiny_time_grid_drift() -> None:
    from flowcean.hydra import compare_state_traces

    reference = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[0.0], [1.0]], dtype=float),
        location=np.array(["a", "a"], dtype=object),
        events=(),
    )
    predicted = Trace(
        t=np.array([0.0, 1.0 + 5e-13], dtype=float),
        x=np.array([[0.0], [1.0]], dtype=float),
        location=np.array(["mode_0", "mode_0"], dtype=object),
        events=(),
    )

    comparison = compare_state_traces(reference, predicted)

    assert comparison.max_error == pytest.approx(0.0)


def test_compare_state_traces_rejects_mismatched_time_grid() -> None:
    from flowcean.hydra import compare_state_traces

    reference = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[0.0], [1.0]], dtype=float),
        location=np.array(["a", "a"], dtype=object),
        events=(),
    )
    predicted = Trace(
        t=np.array([0.0, 1.1], dtype=float),
        x=np.array([[0.0], [1.0]], dtype=float),
        location=np.array(["mode_0", "mode_0"], dtype=object),
        events=(),
    )

    with pytest.raises(ValueError, match="time grids"):
        compare_state_traces(reference, predicted)


def test_hydra_package_exports_simulation_api() -> None:
    from flowcean import hydra

    assert "HyDRATraceSchema" in hydra.__all__
    assert "StateTraceComparison" in hydra.__all__
    assert "compare_state_traces" in hydra.__all__
