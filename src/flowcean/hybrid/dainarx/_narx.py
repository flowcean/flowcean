"""NARX regression features, least-squares fitting, and prediction."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

type FloatArray = NDArray[np.float64]
type NonlinearTerm = Callable[[FloatArray, FloatArray], ArrayLike]
"""A nonlinear regression feature called as ``term(history, current_input)``.

``history`` has shape ``(order, n_outputs)``, oldest observation first;
``current_input`` has shape ``(n_inputs,)`` and is aligned with the output
being predicted. Here ``n_outputs`` and ``n_inputs`` count the selected
output and input features. Columns follow their selected feature order.
Return a ``(n_outputs,)`` vector in output-feature order, with one
feature value for each output's regression.
"""


def finite_array(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def columns(frame: pl.DataFrame, names: Sequence[str]) -> FloatArray:
    for name in names:
        if name not in frame.columns or not frame.schema[name].is_numeric():
            raise ValueError(
                f"Selected column {name!r} must exist and be numeric"
            )
    if not names:
        return np.empty((frame.height, 0))
    return finite_array(frame.select(names).to_numpy(), "Selected columns")


def sampling_period(frame: pl.DataFrame, name: str, dt: float | None) -> float:
    """Check timestamp spacing and return the sampling interval.

    Infer the interval when ``dt`` is absent; otherwise check against it.
    This validates observation timing without resampling the trace.
    """
    time = columns(frame, (name,))[:, 0]
    if len(time) == 0:
        raise ValueError("A sampled trace needs at least one timestamp")
    if len(time) < 2 and dt is None:
        raise ValueError(
            "Inferring the sampling period needs at least two timestamps"
        )
    differences = np.diff(time)
    period = float(differences[0]) if dt is None else dt
    if not np.isfinite(period) or period <= 0 or not np.all(differences > 0):
        raise ValueError("Timestamps must be strictly increasing")
    if not np.allclose(differences, period, rtol=1e-7, atol=period * 1e-9):
        raise ValueError("Sampled traces must share a uniform timestamp grid")
    return period


def selections(
    input_features: Sequence[str],
    output_features: Sequence[str],
    time_column: str,
    order: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    inputs, outputs = tuple(input_features), tuple(output_features)
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    names = (*inputs, *outputs, time_column)
    if not outputs or len(set(names)) != len(names) or "location_id" in names:
        raise ValueError(
            "Select at least one output and distinct input/output/time names; "
            "'location_id' is reserved for rollout location IDs"
        )
    return inputs, outputs


def design_row(
    history: FloatArray,
    current_input: FloatArray,
    terms: tuple[NonlinearTerm, ...],
) -> FloatArray:
    """Build one ``(n_outputs, n_coefficients)`` regression row.

    History is ``(order, n_outputs)``, oldest first; ``current_input`` is
    ``(n_inputs,)`` for the output being predicted. Coefficient columns are
    nonlinear terms, all outputs at lags 1 through order, current inputs,
    then bias, so ``n_coefficients = len(terms) + order * n_outputs +
    n_inputs + 1``. Output and input axes follow selected feature order.
    """
    n_outputs = history.shape[1]
    linear = np.concatenate((history[::-1].ravel(), current_input, [1.0]))
    row = np.empty((n_outputs, len(terms) + len(linear)))
    for index, term in enumerate(terms):
        values = finite_array(term(history, current_input), "Nonlinear term")
        if values.shape != (n_outputs,):
            raise ValueError(
                f"Nonlinear terms must return shape ({n_outputs},)"
            )
        row[:, index] = values
    row[:, len(terms) :] = linear
    return row


def design(
    outputs: FloatArray,
    inputs: FloatArray,
    order: int,
    terms: tuple[NonlinearTerm, ...],
) -> FloatArray:
    """Build regression rows aligned with ``outputs[order:]``.

    Outputs and inputs have shapes ``(n_observations, n_outputs)`` and
    ``(n_observations, n_inputs)``, in selected feature order. Each target
    uses the preceding ``order`` outputs and its same-row input. Return
    ``(n_observations - order, n_outputs, n_coefficients)`` rows with the
    coefficient-column order defined by ``design_row``.
    """
    return np.stack(
        [
            design_row(outputs[t - order : t], inputs[t], terms)
            for t in range(order, len(outputs))
        ]
    )


def solve(rows: FloatArray, targets: FloatArray) -> tuple[FloatArray, float]:
    """Return least-squares coefficients and maximum per-output training RMSE.

    ``rows`` has axes ``(n_targets, n_outputs, n_coefficients)``;
    ``targets`` has shape ``(n_targets, n_outputs)``. Coefficients have
    shape ``(n_outputs, n_coefficients)``. Compute RMSE directly from
    residuals, including for rank-deficient fits.
    """
    coefficients = np.empty((targets.shape[1], rows.shape[2]))
    errors = []
    for output in range(targets.shape[1]):
        matrix = rows[:, output, :]
        # Column scaling keeps differently sized nonlinear terms resolvable.
        scale = np.linalg.norm(matrix, axis=0)
        scale[scale == 0] = 1
        # Explicit NumPy cutoff: machine epsilon times max(matrix.shape).
        solution = np.linalg.lstsq(
            matrix / scale, targets[:, output], rcond=None
        )[0]
        coefficients[output] = solution / scale
        residual = targets[:, output] - matrix @ coefficients[output]
        # Deliberate adaptation of the paper's unnormalized L2 acceptance.
        errors.append(float(np.sqrt(np.mean(residual**2))))
    finite_array(coefficients, "Fitted coefficients")
    finite_array(errors, "Fitting RMSE")
    return coefficients, max(errors)


@dataclass(frozen=True)
class NarxModel:
    """A nonlinear autoregressive model with exogenous inputs (NARX).

    Predict the next output from previous outputs and the input at the row
    being predicted. Let ``n_outputs`` and ``n_inputs`` count the selected
    output and input features; ``n_coefficients = len(nonlinear_terms) +
    order * n_outputs + n_inputs + 1``.

    Attributes:
        order: Number of previous output observations used for prediction.
        input_features: Selected input names, in input-vector order.
        output_features: Selected output names, in output-vector order.
        time_column: Timestamp column name.
        dt: Sampling period in timestamp units.
        nonlinear_terms: Ordered nonlinear regression features; see
            [NonlinearTerm][flowcean.hybrid.dainarx.NonlinearTerm].
        coefficients: Array of shape ``(n_outputs, n_coefficients)``.
            Columns are nonlinear terms, all outputs at lag 1 through
            ``order`` (newest lag first), current inputs, then bias.
        rmse: Largest root-mean-square error (RMSE) among the outputs,
            measured on training targets in output units. This is not a
            guarantee of future prediction error.
    """

    order: int
    input_features: tuple[str, ...]
    output_features: tuple[str, ...]
    time_column: str
    dt: float
    nonlinear_terms: tuple[NonlinearTerm, ...]
    coefficients: FloatArray
    rmse: float

    def predict(
        self, history: ArrayLike, current_input: ArrayLike
    ) -> FloatArray:
        """Predict the next output from output history and its aligned input.

        ``n_outputs`` and ``n_inputs`` count the selected output and input
        features, respectively.

        Args:
            history: Output history of shape ``(order, n_outputs)``, oldest
                observation first, with columns in ``output_features`` order.
            current_input: A ``(n_inputs,)`` vector in ``input_features`` order,
                aligned with the output being predicted, not lagged.

        Returns:
            A ``(n_outputs,)`` vector in ``output_features`` order.
        """
        past = finite_array(history, "Output history")
        inputs = finite_array(current_input, "Current input")
        if past.shape != (self.order, len(self.output_features)):
            raise ValueError("Output history must have shape (order, outputs)")
        if inputs.shape != (len(self.input_features),):
            raise ValueError("Current input must have shape (inputs,)")
        row = design_row(past, inputs, self.nonlinear_terms)
        return finite_array(
            np.sum(row * self.coefficients, axis=1), "Prediction"
        )


def fit_narx(
    traces: Sequence[pl.DataFrame],
    *,
    input_features: Sequence[str],
    output_features: Sequence[str],
    time_column: str,
    order: int,
    nonlinear_terms: Sequence[NonlinearTerm] = (),
) -> NarxModel:
    """Fit one NARX model to predict the next output from history and inputs.

    Use ordinary least squares without crossing trace histories. Each trace
    needs more than ``order`` observations; pooled targets must number at
    least ``n_coefficients = len(nonlinear_terms) + order * n_outputs +
    n_inputs + 1``, where ``n_outputs`` and ``n_inputs`` count selected
    features. Rank-deficient fits are allowed.

    Args:
        traces: Independent Polars frames sampled at the same interval.
            Start times and lengths may differ; only selected columns are used.
        input_features: Input column names, in input-vector order.
        output_features: Output column names, in output-vector order.
        time_column: Timestamp column, increasing within each trace.
        order: Number of previous output observations used for prediction.
        nonlinear_terms: Ordered nonlinear regression features following the
            [NonlinearTerm][flowcean.hybrid.dainarx.NonlinearTerm] contract.

    Returns:
        A [NarxModel][flowcean.hybrid.dainarx.NarxModel] with fitted
            coefficients and maximum per-output training RMSE.
    """
    inputs, outputs = selections(
        input_features, output_features, time_column, order
    )
    terms = tuple(nonlinear_terms)
    rows, targets = [], []
    dt = None
    for index, trace in enumerate(traces):
        dt = sampling_period(trace, time_column, dt)
        x, u = columns(trace, outputs), columns(trace, inputs)
        if len(x) <= order:
            raise ValueError(f"Trace {index} has no targets after its history")
        rows.append(design(x, u, order, terms))
        targets.append(x[order:])
    if not rows or dt is None:
        raise ValueError("Supply at least one sampled trace")
    pooled = np.concatenate(rows)
    if len(pooled) < pooled.shape[2]:
        raise ValueError(
            "Insufficient targets for the NARX coefficient columns"
        )
    coefficients, rmse = solve(pooled, np.concatenate(targets))
    return NarxModel(
        order, inputs, outputs, time_column, dt, terms, coefficients, rmse
    )
