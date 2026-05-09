import logging
from typing import Any, override

import polars as pl

from flowcean._optional import raise_for_missing_optional_dependency

try:
    from pysr import PySRRegressor
except ModuleNotFoundError as error:
    raise_for_missing_optional_dependency(
        error,
        extra="pysr",
        module="flowcean.pysr.learner",
        missing_dependencies={"pysr"},
    )

from flowcean.core import Model, SupervisedIncrementalLearner

logger = logging.getLogger(__name__)


class PySRModel(Model):
    def __init__(self, model: PySRRegressor, output_column: str) -> None:
        self.model = model
        self.output_column = output_column

    def flow_summary(self) -> str:
        try:
            return _stringify_equation(self.model.sympy())
        except (AttributeError, ValueError):
            pass

        get_best = getattr(self.model, "get_best", None)
        if callable(get_best):
            best_equation = _equation_summary_from_record(get_best())
            if best_equation is not None:
                return best_equation

        equations = getattr(self.model, "equations_", None)
        if equations is not None:
            equation_summary = _equation_summary_from_table(equations)
            if equation_summary is not None:
                return equation_summary

        return type(self.model).__name__

    @override
    def _predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        df = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        predictions = self.model.predict(df)
        return pl.from_numpy(predictions, schema=[self.output_column]).lazy()


class PySRLearner(SupervisedIncrementalLearner):
    """Wrapper for PySR symbolic regression learner."""

    def __init__(self, model: PySRRegressor) -> None:
        self.model = model
        model.warm_start = True

    def learn_incremental(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> Model:
        output_columns = outputs.collect_schema().names()
        if len(output_columns) != 1:
            message = (
                "PySRLearner currently supports single-output training only."
            )
            raise ValueError(message)

        dfs = pl.collect_all([inputs, outputs])
        collected_inputs = dfs[0]
        collected_outputs = dfs[1]
        self.model.fit(collected_inputs, collected_outputs)

        # Return the trained PySRModel
        return PySRModel(self.model, output_columns[0])


def _stringify_equation(equation: Any) -> str:
    if isinstance(equation, list):
        if not equation:
            return "[]"
        return _stringify_equation(equation[0])
    return str(equation)


def _equation_summary_from_table(equations: Any) -> str | None:
    for candidate in _equation_candidates(equations):
        if candidate is not None:
            return str(candidate)
    return None


def _equation_summary_from_record(equation: Any) -> str | None:
    if isinstance(equation, list):
        if not equation:
            return None
        return _equation_summary_from_record(equation[0])

    for key in ("sympy_format", "equation", "sympy"):
        value = _lookup_equation_value(equation, key)
        if value is not None:
            return str(value)

    if isinstance(equation, str):
        return equation

    return None


def _equation_candidates(equations: Any) -> list[Any]:
    candidates: list[Any] = []
    for column_name in ("sympy_format", "equation", "sympy"):
        if hasattr(equations, "columns") and column_name in equations.columns:
            column = equations[column_name]
            if len(column) > 0:
                candidates.append(
                    column.iloc[-1] if hasattr(column, "iloc") else column[-1],
                )

    if hasattr(equations, "iloc"):
        try:
            last_row = equations.iloc[-1]
        except (IndexError, KeyError, TypeError):
            last_row = None
        if last_row is not None:
            for key in ("sympy_format", "equation", "sympy"):
                try:
                    value = last_row[key]
                except (KeyError, TypeError):
                    continue
                candidates.append(value)

    return candidates


def _lookup_equation_value(equation: Any, key: str) -> Any:
    getter = getattr(equation, "get", None)
    if callable(getter):
        value = getter(key)
        if value is not None:
            return value

    try:
        return equation[key]
    except (KeyError, TypeError, IndexError):
        return None
