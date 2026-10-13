"""Noise-free Dainarx segmentation, pooled clustering, and transition learning."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import numpy as np
import polars as pl
from sklearn.svm import SVC

from ._narx import (
    FloatArray,
    NarxModel,
    NonlinearTerm,
    columns,
    design,
    sampling_period,
    selections,
    solve,
)
from .model import DainarxModel, DainarxTransition


@dataclass(frozen=True)
class DainarxSegment:
    """A half-open sampled-trace interval and its learned location ID.

    Attributes:
        trace_index: Index into the input trace collection.
        start: Inclusive row index in the original trace frame.
        stop: Exclusive row index in the original trace frame.
        location_id: Learned location ID shared across traces, indexing
            ``DainarxModel.locations``.
    """

    trace_index: int
    start: int
    stop: int
    location_id: int


@dataclass(frozen=True)
class DainarxFit:
    """The executable model, complete segmentation, and acceptance threshold.

    Attributes:
        model: Learned locations, guards, and post-switch predictors.
        segments: Half-open original-frame intervals ``[start, stop)``,
            ordered by trace and row, covering all input observations.
        eta: Fitting acceptance threshold in output units, not a rollout
            error bound.
    """

    model: DainarxModel
    segments: tuple[DainarxSegment, ...]
    eta: float


def _segment(
    rows: FloatArray,
    targets: FloatArray,
    *,
    order: int,
    window: int,
    eta: float,
    trace_index: int,
) -> list[tuple[int, int]]:
    """Return screened half-open observation-row bounds for one trace.

    ``rows`` and ``targets`` are aligned regression data beginning at
    observation ``order``. Returned ``(start, stop)`` indices refer to the
    original observations, not regression rows, and cover the whole trace.
    """
    length = len(targets) + order
    boundaries = [0]
    left = 0
    while left <= length - window:
        _, error = solve(
            rows[left : left + window - order],
            targets[left : left + window - order],
        )
        if error > eta:
            boundaries.append(left + window - 1)
            left += window
        else:
            left += 1
    # Property 5 supplies the terminal endpoint omitted from Algorithm 1.
    boundaries.append(length)
    segments = list(pairwise(boundaries))
    for start, stop in segments:
        context = f"Trace {trace_index} segment [{start}, {stop})"
        if stop - start - order <= rows.shape[2]:
            raise ValueError(f"{context}: insufficient targets for screening")
        _, error = solve(
            rows[start : stop - order], targets[start : stop - order]
        )
        if error > eta:
            raise ValueError(
                f"{context}: screening RMSE {error:g} exceeds eta {eta:g}"
            )
    return segments


def _cluster(
    segments: list[tuple[FloatArray, FloatArray]], eta: float
) -> list[list[int]]:
    """Return groups of indices into the supplied regression-data segments.

    A merge is eligible only if the full pooled fit has maximum per-output
    RMSE <= ``eta``. Check pairs deterministically and restart after each
    eligible merge until no pair can merge.
    """
    groups = [[index] for index in range(len(segments))]
    while True:
        merged = False
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                candidate = groups[left] + groups[right]
                rows = np.concatenate([segments[i][0] for i in candidate])
                targets = np.concatenate([segments[i][1] for i in candidate])
                if solve(rows, targets)[1] <= eta:
                    groups[left] = candidate
                    del groups[right]
                    merged = True
                    break
            if merged:
                break
        if not merged:
            return groups


def fit_dainarx(
    traces: Sequence[pl.DataFrame],
    *,
    input_features: Sequence[str],
    output_features: Sequence[str],
    time_column: str,
    order: int,
    window: int,
    guard_kernel: Literal["linear", "poly", "rbf", "sigmoid"],
    guard_c: float,
    guard_gamma: Literal["scale", "auto"] | float,
    nonlinear_terms: Sequence[NonlinearTerm] = (),
    guard_degree: int = 3,
    guard_coef0: float = 0.0,
) -> DainarxFit:
    """Infer sampled NARX locations, SVM guards, and post-switch predictors.

    Fitting acceptance uses the largest per-output root-mean-square error
    (RMSE), rather than the paper's unnormalized L2 residual. The threshold
    is ``eta = 1e-6 * dt * max(abs(training_outputs))``, where ``dt`` is the
    shared sampling period and ``training_outputs`` includes all selected
    output values across the traces. See the
    [Dainarx guide](../examples/dainarx.md) for the noise-free method and usage.

    Args:
        traces: Independent ordered Polars frames with finite numeric selected
            columns and a shared uniform sampling period. Start times and
            lengths may differ; unselected columns are ignored.
        input_features: Input column names, in input-vector order.
        output_features: Nonempty output column names, in output-vector order.
        time_column: Numeric timestamp column, strictly increasing per trace.
        order: Number of previous output observations used for prediction.
        window: Number of observations per screening window, not regression
            targets. Each trace needs at least this many observations, and
            ``window - order`` must exceed the per-output coefficient count.
        guard_kernel: SVM kernel: ``"linear"``, ``"poly"``, ``"rbf"``, or
            ``"sigmoid"``.
        guard_c: Positive inverse regularization strength; larger values
            penalize training misclassification more.
        guard_gamma: Nonnegative scale coefficient for ``"rbf"``, ``"poly"``,
            and ``"sigmoid"``; ignored for ``"linear"``. For RBF, larger values
            narrow each training observation's influence. ``"scale"`` chooses
            a coefficient from the guard feature count and training-data
            variance; ``"auto"`` uses the reciprocal of the feature count.
            Guard features are selected outputs followed by selected inputs.
        nonlinear_terms: Ordered nonlinear regression features following the
            [NonlinearTerm][flowcean.hybrid.dainarx.NonlinearTerm] contract.
            Linear terms already include all outputs at every lag, current
            inputs, and bias.
        guard_degree: Nonnegative polynomial degree, used only for ``"poly"``.
        guard_coef0: Additive constant in the ``"poly"`` and ``"sigmoid"``
            kernels; ignored for other kernels.

    Returns:
        The learned model, complete segmentation, and fitting threshold in a
            [DainarxFit][flowcean.hybrid.dainarx.DainarxFit].

    Whole segments need more targets than the per-output coefficient count,
    ``len(nonlinear_terms) + order * len(output_features) +
    len(input_features) + 1``. Each reset phase needs at least that many
    transition occurrences. Insufficient or rejected segment/reset/guard
    evidence, or a detected same-location boundary within a trace, raises
    ``ValueError`` rather than returning a partial model. Rank deficiency
    alone is allowed.
    """
    inputs, outputs = selections(
        input_features, output_features, time_column, order
    )
    terms = tuple(nonlinear_terms)
    coefficient_count = len(terms) + order * len(outputs) + len(inputs) + 1
    if (
        isinstance(window, bool)
        or not isinstance(window, int)
        or window - order <= coefficient_count
    ):
        raise ValueError(
            f"window - order must exceed {coefficient_count} coefficient columns"
        )
    xs, us, designs = [], [], []
    dt = None
    for index, trace in enumerate(traces):
        dt = sampling_period(trace, time_column, dt)
        x, u = columns(trace, outputs), columns(trace, inputs)
        if len(x) < window:
            raise ValueError(f"Trace {index} is shorter than window")
        xs.append(x)
        us.append(u)
        designs.append(design(x, u, order, terms))
    if not xs or dt is None:
        raise ValueError("Supply at least one sampled trace")
    eta = 1e-6 * dt * max(float(np.max(np.abs(x))) for x in xs)
    if not np.isfinite(eta):
        raise ValueError("Training output scale produces a non-finite eta")
    bounds = []
    segment_rows = []
    for index, (x, rows) in enumerate(zip(xs, designs, strict=True)):
        for start, stop in _segment(
            rows,
            x[order:],
            order=order,
            window=window,
            eta=eta,
            trace_index=index,
        ):
            bounds.append((index, start, stop))
            segment_rows.append(
                (rows[start : stop - order], x[start + order : stop])
            )
    groups = _cluster(segment_rows, eta)

    def fitted(
        rows: FloatArray, targets: FloatArray, context: str
    ) -> NarxModel:
        if len(targets) < coefficient_count:
            raise ValueError(
                f"{context}: insufficient targets (need at least {coefficient_count})"
            )
        coefficients, error = solve(rows, targets)
        if error > eta:
            raise ValueError(f"{context}: RMSE {error:g} exceeds eta {eta:g}")
        return NarxModel(
            order, inputs, outputs, time_column, dt, terms, coefficients, error
        )

    locations, assignments = [], {}
    for location, group in enumerate(groups):
        locations.append(
            fitted(
                np.concatenate([segment_rows[i][0] for i in group]),
                np.concatenate([segment_rows[i][1] for i in group]),
                f"Location {location}",
            )
        )
        assignments.update(dict.fromkeys(group, location))
    segments = tuple(
        DainarxSegment(index, start, stop, assignments[i])
        for i, (index, start, stop) in enumerate(bounds)
    )
    for previous, current in pairwise(segments):
        if (
            previous.trace_index == current.trace_index
            and previous.location_id == current.location_id
        ):
            raise ValueError(
                f"Trace {current.trace_index} boundary at row {current.start}: "
                f"unsupported same-location reset in location {current.location_id}"
            )
    labels = [np.full(len(x), -1, dtype=np.int64) for x in xs]
    for segment in segments:
        labels[segment.trace_index][segment.start : segment.stop] = (
            segment.location_id
        )
    edges = sorted(
        {
            (int(label[t]), int(label[t + 1]))
            for label in labels
            for t in range(len(label) - 1)
            if label[t] != label[t + 1]
        }
    )
    transitions = []
    for source, destination in edges:
        guard_rows, guard_targets = [], []
        phase_rows: list[list[FloatArray]] = [[] for _ in range(order)]
        phase_targets: list[list[FloatArray]] = [[] for _ in range(order)]
        for x, u, rows, label in zip(xs, us, designs, labels, strict=True):
            source_rows = np.flatnonzero(label[:-1] == source)
            guard_rows.append(
                np.concatenate((x[source_rows], u[source_rows]), axis=1)
            )
            guard_targets.append(label[source_rows + 1] == destination)
            for t in source_rows[label[source_rows + 1] == destination]:
                boundary = int(t) + 1
                if (
                    boundary < order
                    or boundary + order > len(x)
                    or not np.all(
                        label[boundary : boundary + order] == destination
                    )
                ):
                    raise ValueError(
                        f"Transition {source}->{destination}: incomplete reset context"
                    )
                for phase in range(order):
                    target = boundary + phase
                    phase_rows[phase].append(rows[target - order])
                    phase_targets[phase].append(x[target])
        positive = np.concatenate(guard_targets)
        if np.all(positive) or not np.any(positive):
            raise ValueError(
                f"Transition {source}->{destination} needs positive and negative guard evidence"
            )
        guard = SVC(
            kernel=guard_kernel,
            C=guard_c,
            # sklearn's inferred annotation omits documented numeric gamma.
            gamma=guard_gamma,  # pyright: ignore[reportArgumentType]
            degree=guard_degree,
            coef0=guard_coef0,
            class_weight=None,
        ).fit(np.concatenate(guard_rows), positive)
        if guard.fit_status_ != 0:
            raise ValueError(
                f"Transition {source}->{destination}: SVM did not converge"
            )
        predictors = tuple(
            fitted(
                np.stack(rows),
                np.stack(targets),
                f"Transition {source}->{destination} reset phase {phase + 1}",
            )
            for phase, (rows, targets) in enumerate(
                zip(phase_rows, phase_targets, strict=True)
            )
        )
        transitions.append(
            DainarxTransition(source, destination, guard, predictors)
        )
    return DainarxFit(
        DainarxModel(tuple(locations), tuple(transitions)), segments, eta
    )
