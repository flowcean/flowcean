"""Behavioral tests for HyDRA mode discovery."""

from __future__ import annotations

from typing import override

import numpy as np
import polars as pl
import pytest

from flowcean.core import Model, SupervisedIncrementalLearner
from flowcean.hybrid import HyDRALearner, HyDRATraceSchema


class LinearFeatureModel(Model):
    """Single-feature affine model fitted by the test learner."""

    def __init__(
        self,
        *,
        feature: str,
        output: str,
        slope: float,
        intercept: float,
    ) -> None:
        self.feature = feature
        self.output = output
        self.slope = slope
        self.intercept = intercept

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
        values = self.slope * frame[self.feature].to_numpy() + self.intercept
        return pl.DataFrame({self.output: values}).lazy()


class IncrementalLinearLearner(SupervisedIncrementalLearner):
    """Deterministic least-squares learner that retains incremental rows."""

    def __init__(self, feature: str = "x") -> None:
        self.feature = feature
        self._inputs: list[pl.DataFrame] = []
        self._outputs: list[pl.DataFrame] = []

    @override
    def learn_incremental(
        self,
        inputs: pl.DataFrame | pl.LazyFrame,
        outputs: pl.DataFrame | pl.LazyFrame,
    ) -> LinearFeatureModel:
        input_frame = (
            inputs.collect() if isinstance(inputs, pl.LazyFrame) else inputs
        )
        output_frame = (
            outputs.collect() if isinstance(outputs, pl.LazyFrame) else outputs
        )
        self._inputs.append(input_frame)
        self._outputs.append(output_frame)
        all_inputs = pl.concat(self._inputs)
        all_outputs = pl.concat(self._outputs)
        design = np.column_stack(
            (all_inputs[self.feature].to_numpy(), np.ones(all_inputs.height)),
        )
        slope, intercept = np.linalg.lstsq(
            design,
            all_outputs.to_series().to_numpy(),
            rcond=None,
        )[0]
        return LinearFeatureModel(
            feature=self.feature,
            output=all_outputs.columns[0],
            slope=float(slope),
            intercept=float(intercept),
        )


def test_learner_validates_configuration() -> None:
    factory = IncrementalLinearLearner
    with pytest.raises(ValueError, match="threshold must be non-negative"):
        HyDRALearner(factory, threshold=-0.1)
    with pytest.raises(ValueError, match="start_width must be positive"):
        HyDRALearner(factory, threshold=0.1, start_width=0)
    with pytest.raises(ValueError, match="step_width must be positive"):
        HyDRALearner(factory, threshold=0.1, step_width=0)


def test_learner_requires_rows_and_a_single_output() -> None:
    learner = HyDRALearner(IncrementalLinearLearner, threshold=0.01)

    with pytest.raises(ValueError, match="at least one row"):
        learner.learn(
            pl.DataFrame(schema={"time": pl.Float64, "x": pl.Float64}).lazy(),
            pl.DataFrame(schema={"dx": pl.Float64}).lazy(),
        )
    with pytest.raises(ValueError, match="single-output"):
        learner.learn(
            pl.DataFrame({"time": [0.0], "x": [1.0]}).lazy(),
            pl.DataFrame({"dx": [1.0], "dy": [2.0]}).lazy(),
        )


def test_single_mode_is_discovered_and_learned_from_real_rows() -> None:
    schema = HyDRATraceSchema(
        time="time",
        state=("x",),
        derivative=("dx",),
    )
    x = np.array([-2.0, 0.5, 3.0, -1.0, 4.5, 2.0])
    inputs = pl.DataFrame({"time": np.arange(x.size, dtype=float), "x": x})
    outputs = pl.DataFrame({"dx": 2.5 * x - 0.75})
    learner = HyDRALearner(
        IncrementalLinearLearner,
        threshold=1e-10,
        start_width=3,
        step_width=2,
        trace_schema=schema,
    )

    model = learner.learn(inputs.lazy(), outputs.lazy())
    prediction_inputs = pl.DataFrame(
        {"time": [10.0, 11.0, 12.0], "x": [-3.0, 0.0, 5.0]},
    )
    predictions = model.predict(prediction_inputs).collect()["dx"].to_numpy()

    assert len(model.modes) == 1
    np.testing.assert_allclose(
        predictions,
        2.5 * prediction_inputs["x"].to_numpy() - 0.75,
        rtol=1e-12,
        atol=1e-12,
    )


def test_schema_mismatch_is_rejected_before_learning() -> None:
    schema = HyDRATraceSchema(
        time="time",
        state=("x",),
        derivative=("dx",),
    )
    learner = HyDRALearner(
        IncrementalLinearLearner,
        threshold=0.1,
        trace_schema=schema,
    )

    with pytest.raises(ValueError, match="input_features must match"):
        learner.learn(
            pl.DataFrame({"time": [0.0], "wrong": [1.0]}).lazy(),
            pl.DataFrame({"dx": [2.0]}).lazy(),
        )
