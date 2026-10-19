"""Clustered symbolic regression for aligned scalar observations."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl
import sympy as sp
from scipy.special import logsumexp

from ._search import (
    _aic,
    _Array,
    _Candidate,
    _Expression,
    _inputs,
    _search,
    _SearchFactory,
)


@dataclass(frozen=True)
class CSRDiagnostics:
    """Stopping information, not a certificate of distinct location recovery.

    ``updates`` counts committed sequential replacements, excluding
    initialization. ``error`` is the final responsibility-weighted sum of
    absolute residuals; ``aic`` uses the total original tree complexity.
    ``failed_location`` is set only for an entirely unusable update frontier.
    """

    completed_sweeps: int
    updates: int
    stop_reason: Literal["max_sweeps", "error_stable", "no_usable_candidate"]
    error: float
    aic: float
    failed_location: int | None = None


@dataclass(frozen=True)
class MMSRCSR:
    """Scalar symbolic mappings, residual variances, and soft memberships.

    Locations are positional integers in ``expressions`` and ``variances``.
    ``memberships[i]`` has shape ``(trace_i.height, n_locations)`` in the
    original row order. No observations are dropped. Variances are in squared
    output units. Expressions alone do not classify locations: posterior
    membership also requires the observed target.
    """

    input_features: tuple[str, ...]
    output_feature: str
    variances: _Array
    memberships: tuple[_Array, ...]
    diagnostics: CSRDiagnostics
    _functions: tuple[_Expression, ...] = field(repr=False)

    @property
    def expressions(self) -> tuple[sp.Expr, ...]:
        """Selected SymPy mappings, one per location."""
        return tuple(function.expression for function in self._functions)

    @property
    def complexities(self) -> tuple[int, ...]:
        """Original search-tree node counts, before SymPy simplification."""
        return tuple(function.complexity for function in self._functions)

    def predict(self, inputs: pl.DataFrame) -> _Array:
        """Evaluate every mapping; return shape ``(n_rows, n_locations)``.

        Only ``input_features`` are read. Nonfinite expression outputs raise
        ``ValueError``, including rational singularities outside training.
        """
        values = _inputs(inputs, self.input_features)
        return np.column_stack([f.predict(values) for f in self._functions])

    def membership(self, observations: pl.DataFrame) -> _Array:
        """Evaluate posterior memberships from regressors AND observed targets.

        Returns shape ``(n_rows, n_locations)``. This is not an input-only
        location classifier and does not perform temporal inference.
        """
        targets = _inputs(observations, [self.output_feature])[:, 0]
        return _responsibilities(
            targets, self.predict(observations), self.variances
        )


def _observations(
    traces: Sequence[pl.DataFrame],
    input_features: tuple[str, ...],
    output_feature: str,
) -> tuple[_Array, _Array]:
    if not traces or any(trace.height == 0 for trace in traces):
        raise ValueError("MMSR requires nonempty independent traces")
    inputs = np.concatenate([_inputs(t, input_features) for t in traces])
    targets = np.concatenate(
        [_inputs(t, [output_feature])[:, 0] for t in traces]
    )
    return inputs, targets


def _responsibilities(
    targets: _Array, predictions: _Array, variances: _Array
) -> _Array:
    if not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError("Invalid residual variances")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        log_density = -0.5 * (
            np.log(variances)
            + np.square(targets[:, None] - predictions) / variances
        )
        normalizer = np.asarray(logsumexp(log_density, axis=1))
        gamma = np.exp(log_density - normalizer[:, None])
    if not np.isfinite(gamma).all():
        raise ValueError("Invalid Gaussian responsibilities")
    return gamma


def _support(weights: _Array) -> tuple[float, float]:
    mass = float(weights.sum())
    squares = float(np.dot(weights, weights))
    denominator = mass * mass - squares
    if (
        mass <= 1e-12
        or denominator <= 1e-12
        or mass * mass / squares <= 1 + 1e-10
    ):
        raise ValueError("Insufficient effective membership support")
    return mass, denominator


def _variance(weights: _Array, residual: _Array, floor: float) -> float:
    mass, denominator = _support(weights)
    with np.errstate(over="ignore", invalid="ignore"):
        variance = float(mass / denominator * np.dot(weights, residual**2))
    if not np.isfinite(variance):
        raise ValueError("Nonfinite weighted residual variance")
    return max(variance, floor)


def _state(
    targets: _Array, predictions: _Array, variances: _Array
) -> tuple[_Array, float]:
    gamma = _responsibilities(targets, predictions, variances)
    for location in range(predictions.shape[1]):
        try:
            _support(gamma[:, location])
        except ValueError as error:
            raise ValueError(
                f"Invalid committed MMSR mixture at location {location}: {error}"
            ) from error
    error = float(np.sum(gamma * np.abs(targets[:, None] - predictions)))
    if not np.isfinite(error):
        raise ValueError("Nonfinite committed MMSR mixture error")
    return gamma, error


def _choose_global(
    candidates: list[_Candidate],
    *,
    location: int,
    inputs: _Array,
    targets: _Array,
    functions: list[_Expression],
    predictions: _Array,
    variances: _Array,
    variance_floor: float,
) -> tuple[_Expression, float] | None:
    best = None
    best_aic = np.inf
    other_complexity = (
        sum(f.complexity for f in functions) - functions[location].complexity
    )
    for candidate in candidates:
        try:
            temporary = predictions.copy()
            temporary[:, location] = candidate.function.predict(inputs)
            # Algorithm 2's temporary responsibilities use OLD variances. Only
            # the candidate's variance is updated, once, after this step.
            gamma = _responsibilities(targets, temporary, variances)
            variance = _variance(
                gamma[:, location],
                targets - temporary[:, location],
                variance_floor,
            )
            prospective_variances = variances.copy()
            prospective_variances[location] = variance
            # Check eligibility after the variance update without changing the
            # OLD-variance responsibilities used by the selection criterion.
            _state(targets, temporary, prospective_variances)
            error = float(np.sum(gamma * np.abs(targets[:, None] - temporary)))
            score = _aic(
                other_complexity + candidate.function.complexity,
                len(targets),
                error,
            )
        except (ValueError, ArithmeticError):
            continue
        if score < best_aic:
            best = candidate.function, variance
            best_aic = score
    return best


def fit_mmsr_csr(
    traces: Sequence[pl.DataFrame],
    *,
    input_features: Sequence[str],
    output_feature: str,
    n_locations: int,
    search_factory: _SearchFactory,
    max_sweeps: int = 10,
    random_state: int = 0,
    variance_floor: float = 1e-12,
) -> MMSRCSR:
    """Jointly fit symbolic mappings, memberships, and residual variances.

    Args:
        traces: Independent eager frames of aligned scalar regression rows.
            Inputs may be measured states. No time, derivative, lag, or sampling
            period is inferred. Ordering matters only for subsequent TM.
        input_features: Ordered regressor columns.
        output_feature: One observed scalar target column.
        n_locations: Known positive number of locations.
        search_factory: Zero-argument factory returning a fresh configured
            ``pysr.PySRRegressor``. Requires ``flowcean[pysr]``. The factory owns
            operators, budgets, and seeds. MMSR owns weighted loss, ordinary
            scalar expressions, tree-node complexity, candidate selection, and
            temporary output storage. Leave custom loss, expression/type specs,
            complexity overrides, denoising, feature selection, guesses, and
            warm starts unset. Backend errors propagate.
        max_sweeps: Finite cap on complete sequential update sweeps.
        random_state: Seed for Dirichlet(1, ..., 1) initial memberships, separate
            from the factory's search seeds.
        variance_floor: Positive floor in squared output units; no rescaling.

    Returns:
        Independently usable CSR artifact. A wholly unusable proposal frontier
        preserves the last valid mixture and reports ``no_usable_candidate``;
        invalid initialization or committed mixtures raise ``ValueError``.

    Notes:
        Follows Ly and Lipson's MMSR (JMLR 2012), with PySR replacing Eureqa.
        Local search minimizes weighted mean ``log1p(abs(residual))``.
        Initialization interprets the paper's unspecified local AIC as
        ``2*c + N*log(local_mean_loss)``. Updates select from the fresh frontier
        by complete-mixture AIC with total tree complexity and the sum of
        responsibility-weighted absolute residuals, not by local loss.
        Gaussian memberships have variance prefactors and no frequency priors;
        variances use the paper's unbiased weighted estimator, not MLE.

        Finite searches can worsen the incumbent: no incumbent injection,
        monotonicity gate, restart, or best-ever return is used. Five complete
        sweeps with relative error change below 2% stop as ``error_stable``;
        collapsed or inaccurate mappings can also be stable. Numerical support
        guards are 1e-12 for mass and variance denominator, effective sample
        size must exceed 1 + 1e-10, and logarithms floor error at 1e-300.
    """
    if not isinstance(n_locations, int) or n_locations < 1:
        raise ValueError("n_locations must be a positive integer")
    if not isinstance(max_sweeps, int) or max_sweeps < 1:
        raise ValueError("max_sweeps must be a positive integer")
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be finite and positive")
    features = tuple(input_features)
    inputs, targets = _observations(traces, features, output_feature)
    gamma = np.random.default_rng(random_state).dirichlet(
        np.ones(n_locations), size=len(targets)
    )
    functions = []
    variances = np.empty(n_locations)
    for location in range(n_locations):
        frontier = _search(
            inputs,
            targets,
            gamma[:, location],
            features,
            search_factory,
            objective="csr",
        )
        if not frontier:
            raise ValueError(
                f"MMSR initialization at location {location}: no usable candidate"
            )
        chosen = min(
            frontier,
            key=lambda c: _aic(c.function.complexity, len(targets), c.loss),
        ).function
        try:
            variances[location] = _variance(
                gamma[:, location],
                targets - chosen.predict(inputs),
                variance_floor,
            )
        except ValueError as error:
            raise ValueError(
                f"MMSR initialization at location {location}: {error}"
            ) from error
        functions.append(chosen)
    predictions = np.column_stack([f.predict(inputs) for f in functions])
    gamma, error = _state(targets, predictions, variances)
    completed_sweeps = updates = stable_sweeps = 0
    stop_reason = "max_sweeps"
    failed_location = None
    for _ in range(max_sweeps):
        previous_error = error
        for location in range(n_locations):
            gamma, _ = _state(targets, predictions, variances)
            frontier = _search(
                inputs,
                targets,
                gamma[:, location],
                features,
                search_factory,
                objective="csr",
            )
            selected = _choose_global(
                frontier,
                location=location,
                inputs=inputs,
                targets=targets,
                functions=functions,
                predictions=predictions,
                variances=variances,
                variance_floor=variance_floor,
            )
            if selected is None:
                stop_reason = "no_usable_candidate"
                failed_location = location
                break
            functions[location], variances[location] = selected
            predictions[:, location] = functions[location].predict(inputs)
            updates += 1
        gamma, error = _state(targets, predictions, variances)
        if failed_location is not None:
            break
        completed_sweeps += 1
        relative_change = abs(error - previous_error) / max(
            previous_error, 1e-300
        )
        stable_sweeps = stable_sweeps + 1 if relative_change < 0.02 else 0
        if stable_sweeps == 5:
            stop_reason = "error_stable"
            break
    boundaries = np.cumsum([trace.height for trace in traces])[:-1]
    diagnostics = CSRDiagnostics(
        completed_sweeps,
        updates,
        stop_reason,
        error,
        _aic(sum(f.complexity for f in functions), len(targets), error),
        failed_location,
    )
    return MMSRCSR(
        features,
        output_feature,
        variances,
        tuple(np.split(gamma, boundaries)),
        diagnostics,
        tuple(functions),
    )
