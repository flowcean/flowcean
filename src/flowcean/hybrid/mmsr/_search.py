"""Fresh weighted PySR searches and method-local expression exports."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
import sympy as sp
from numpy.typing import NDArray
from scipy.special import expit

from flowcean._optional import raise_for_missing_optional_dependency

if TYPE_CHECKING:
    from pysr import PySRRegressor

# Private search helpers shared by mapping and transition identification.
__all__ = ["_aic", "_inputs", "_search"]

type _Array = NDArray[np.float64]
type _SearchFactory = Callable[[], "PySRRegressor"]


@dataclass(frozen=True)
class _Expression:
    expression: sp.Expr
    complexity: int
    input_features: tuple[str, ...]
    _evaluate: Callable[..., Any] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
            raise ValueError("Expression contains nonfinite constants")
        symbols = tuple(sp.Symbol(name) for name in self.input_features)
        if not self.expression.free_symbols <= set(symbols):
            raise ValueError("Expression contains unknown input symbols")
        object.__setattr__(
            self,
            "_evaluate",
            sp.lambdify(
                symbols, self.expression, modules="numpy", dummify=True
            ),
        )

    def predict(self, inputs: _Array) -> _Array:
        with np.errstate(all="ignore"):
            values = np.asarray(self._evaluate(*inputs.T))
        if np.iscomplexobj(values):
            raise ValueError("Expression produced complex values")
        values = np.broadcast_to(values, (len(inputs),)).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Expression produced nonfinite values")
        return values


@dataclass(frozen=True)
class _Candidate:
    function: _Expression
    loss: float


def _inputs(frame: pl.DataFrame, features: Sequence[str]) -> _Array:
    if not features or len(set(features)) != len(features):
        raise ValueError("input_features must be nonempty and unique")
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    selected = frame.select(features)
    if not all(dtype.is_numeric() for dtype in selected.dtypes):
        raise ValueError("MMSR columns must be numeric")
    values = selected.cast(pl.Float64).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("MMSR observations must be finite")
    return values


def _aic(complexity: int, n_rows: int, error: float) -> float:
    if not np.isfinite(error) or error < 0:
        raise ValueError("MMSR objective must be finite and nonnegative")
    return float(2 * complexity + n_rows * np.log(max(error, 1e-300)))


def _pareto(candidates: list[_Candidate]) -> list[_Candidate]:
    frontier = []
    best_loss = np.inf
    for candidate in sorted(
        candidates, key=lambda c: (c.function.complexity, c.loss)
    ):
        if candidate.loss < best_loss:
            frontier.append(candidate)
            best_loss = candidate.loss
    return frontier


def _search(
    inputs: _Array,
    targets: _Array,
    weights: _Array,
    input_features: tuple[str, ...],
    factory: _SearchFactory,
    *,
    objective: Literal["csr", "transition"],
) -> list[_Candidate]:
    try:
        from pysr import PySRRegressor
    except ModuleNotFoundError as error:
        raise_for_missing_optional_dependency(
            error,
            extra="pysr",
            module="flowcean.hybrid.mmsr",
            missing_dependencies={"pysr"},
        )
    model = factory()
    if not isinstance(model, PySRRegressor):
        raise TypeError("search_factory must return a fresh PySRRegressor")
    # These alter the method's objective, observations, tree complexity, or
    # random-population initialization, rather than its search budget.
    for setting in (
        "elementwise_loss",
        "loss_function",
        "loss_function_expression",
        "expression_spec",
        "type_spec",
        "select_k_features",
        "guesses",
        "complexity_of_operators",
        "complexity_of_constants",
        "complexity_of_variables",
        "complexity_mapping",
    ):
        if getattr(model, setting) is not None:
            raise ValueError(f"MMSR controls {setting}; leave it unset")
    if model.denoise or model.warm_start or hasattr(model, "equations_"):
        raise ValueError("MMSR requires a fresh, non-denoising PySR search")
    loss = (
        "mmsr_loss(x, y, w) = w * log1p(abs(x - y))"
        if objective == "csr"
        else "mmsr_loss(x, y, w) = w * "
        "((x >= 0 ? inv(1 + exp(-x)) : exp(x)/(1 + exp(x))) - y)^2"
    )
    # Backend-safe names decouple expression export from Polars column naming.
    names = [f"x{i}" for i in range(inputs.shape[1])]
    symbols = {
        sp.Symbol(name): sp.Symbol(feature)
        for name, feature in zip(names, input_features, strict=True)
    }
    candidates = []
    with TemporaryDirectory(prefix="flowcean-mmsr-") as directory:
        model.set_params(
            elementwise_loss=loss,
            output_directory=directory,
            temp_equation_file=False,
        )
        model.fit(inputs, targets, weights=weights, variable_names=names)
        equations = model.equations_
        if equations is None or isinstance(equations, list):
            raise ValueError("MMSR requires a scalar PySR candidate table")
        for index in range(len(equations)):
            record = equations.iloc[index]
            if not np.isfinite(record["loss"]):
                continue
            try:
                exported = model.sympy(index=index)
                if not isinstance(exported, sp.Expr):
                    continue
                expression = exported.xreplace(symbols)
                function = _Expression(
                    expression, int(record["complexity"]), input_features
                )
                prediction = function.predict(inputs)
                backend_prediction = np.asarray(
                    model.predict(inputs, index=index), dtype=np.float64
                )
                if not np.allclose(
                    prediction, backend_prediction, rtol=1e-7, atol=1e-9
                ):
                    continue
                residual = (
                    prediction - targets
                    if objective == "csr"
                    else expit(prediction) - targets
                )
                losses = (
                    np.log1p(np.abs(residual))
                    if objective == "csr"
                    else np.square(residual)
                )
                local_loss = float(np.dot(weights, losses) / weights.sum())
                if np.isfinite(local_loss):
                    candidates.append(_Candidate(function, local_loss))
            except (
                ArithmeticError,
                ValueError,
                TypeError,
                NameError,
                NotImplementedError,
            ):
                # A bad export or numerical expression is a bad proposal, not
                # a failed search. Backend fit errors above still propagate.
                continue
    return _pareto(candidates)
