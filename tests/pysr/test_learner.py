from typing import Any, cast

import polars as pl

from flowcean.pysr.learner import PySRModel


class FakeEquationModel:
    def sympy(self, index: int | list[int] | None = None) -> str:
        del index
        return "x0 + x1"

    def predict(self, inputs: pl.DataFrame) -> list[list[float]]:
        return [[0.0] for _ in range(inputs.height)]


class FakeBestEquationModel:
    def sympy(self, index: int | list[int] | None = None) -> str:
        del index
        msg = "sympy unavailable"
        raise AttributeError(msg)

    def get_best(self, index: int | list[int] | None = None) -> dict[str, str]:
        del index
        return {"equation": "best_equation(x0)"}

    def predict(self, inputs: pl.DataFrame) -> list[list[float]]:
        return [[0.0] for _ in range(inputs.height)]


class FakeValueErrorEquationModel:
    def sympy(self, index: int | list[int] | None = None) -> str:
        del index
        msg = "no selected equation"
        raise ValueError(msg)

    def get_best(self, index: int | list[int] | None = None) -> dict[str, str]:
        del index
        return {"sympy_format": "fallback_from_get_best"}

    def predict(self, inputs: pl.DataFrame) -> list[list[float]]:
        return [[0.0] for _ in range(inputs.height)]


def test_pysr_model_flow_summary_prefers_equation_text() -> None:
    model = PySRModel(cast("Any", FakeEquationModel()), "y")

    assert model.flow_summary() == "x0 + x1"


def test_pysr_model_flow_summary_falls_back_to_get_best() -> None:
    model = PySRModel(cast("Any", FakeBestEquationModel()), "y")

    assert model.flow_summary() == "best_equation(x0)"


def test_pysr_model_flow_summary_recovers_from_sympy_value_error() -> None:
    model = PySRModel(cast("Any", FakeValueErrorEquationModel()), "y")

    assert model.flow_summary() == "fallback_from_get_best"
