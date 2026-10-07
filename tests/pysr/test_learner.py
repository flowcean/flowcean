"""Mocked batch-adapter checks; these do not exercise PySR or Julia."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal


class FakeRegressor:
    def __init__(self) -> None:
        self.warm_start = False
        self.fits: list[tuple[pl.DataFrame, pl.DataFrame]] = []

    def fit(self, inputs: pl.DataFrame, outputs: pl.DataFrame) -> None:
        self.fits.append((inputs, outputs))

    def predict(self, inputs: pl.DataFrame) -> np.ndarray:
        return 2.0 * inputs.to_series().to_numpy()

    def sympy(self) -> str:
        return "2*x"


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    stub = ModuleType("pysr")
    stub.PySRRegressor = FakeRegressor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pysr", stub)
    path = Path(__file__).parents[2] / "src/flowcean/pysr/learner.py"
    spec = importlib.util.spec_from_file_location(
        "_pysr_batch_adapter_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_fit_prediction_equation_and_no_forced_warm_start(
    adapter: Any,
) -> None:
    regressor = FakeRegressor()
    learner = adapter.PySRLearner(regressor)
    assert regressor.warm_start is False
    inputs = pl.DataFrame({"x": [1.0, 2.0]})
    outputs = pl.DataFrame({"y": [2.0, 4.0]})
    model = learner.learn(inputs.lazy(), outputs.lazy())
    assert len(regressor.fits) == 1
    assert_frame_equal(regressor.fits[0][0], inputs)
    assert_frame_equal(regressor.fits[0][1], outputs)
    assert_frame_equal(model.predict(inputs).collect(), outputs)
    assert model.flow_summary() == "2*x"
    assert regressor.warm_start is False


def test_supplied_rows_are_not_accumulated_by_adapter(adapter: Any) -> None:
    regressor = FakeRegressor()
    learner = adapter.PySRLearner(regressor)
    learner.learn(
        pl.DataFrame({"x": [1.0, 2.0]}).lazy(),
        pl.DataFrame({"y": [2.0, 4.0]}).lazy(),
    )
    learner.learn(
        pl.DataFrame({"x": [3.0]}).lazy(), pl.DataFrame({"y": [6.0]}).lazy()
    )
    assert [inputs["x"].to_list() for inputs, _ in regressor.fits] == [
        [1.0, 2.0],
        [3.0],
    ]


def test_single_output_validation_precedes_fit(adapter: Any) -> None:
    regressor = FakeRegressor()
    with pytest.raises(ValueError, match="single-output"):
        adapter.PySRLearner(regressor).learn(
            pl.DataFrame({"x": [1.0]}).lazy(),
            pl.DataFrame({"y": [2.0], "z": [3.0]}).lazy(),
        )
    assert not regressor.fits


def test_backend_error_propagates(adapter: Any) -> None:
    class FailingRegressor(FakeRegressor):
        def fit(self, inputs: pl.DataFrame, outputs: pl.DataFrame) -> None:
            raise RuntimeError("search failed")

    with pytest.raises(RuntimeError, match="search failed"):
        adapter.PySRLearner(FailingRegressor()).learn(
            pl.DataFrame({"x": [1.0]}).lazy(),
            pl.DataFrame({"y": [2.0]}).lazy(),
        )
