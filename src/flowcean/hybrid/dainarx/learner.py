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
    design_row,
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
            ``DainarxDynamics.locations`` or ``DainarxModel.locations``.
    """

    trace_index: int
    start: int
    stop: int
    location_id: int


@dataclass(frozen=True)
class DainarxDynamics:
    """Complete dynamics discovery and original-observation membership.

    This is not an executable hybrid model: guards and post-switch predictors
    still require transition fitting, which can fail independently. Membership
    is discovered, not ground truth or a per-row RMSE certificate. The first
    ``order`` rows of each segment are labeled even though ordinary location
    fitting excludes them. Adjacent segments may share a location ID; retain
    their detected boundaries even when projected labels do not change.

    Attributes:
        locations: Ordinary NARX models indexed by contiguous location IDs.
            All share history order, selected feature order, time column,
            sampling period, and ordered nonlinear feature functions.
        segments: Half-open original-frame intervals ``[start, stop)``, ordered
            by independent trace and row, covering every observation once.
        trace_lengths: Original observation counts in input trace order.
        eta: Finite nonnegative maximum per-output training RMSE threshold,
            in output units, not a rollout error bound.
    """

    locations: tuple[NarxModel, ...]
    segments: tuple[DainarxSegment, ...]
    trace_lengths: tuple[int, ...]
    eta: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.eta) or self.eta < 0:
            raise ValueError("eta must be finite and nonnegative")
        if (
            not isinstance(self.locations, tuple)
            or not self.locations
            or not all(
                isinstance(model, NarxModel) for model in self.locations
            )
        ):
            raise ValueError("Supply a nonempty tuple of NARX locations")
        template = self.locations[0]
        for model in self.locations:
            names = (
                *model.input_features,
                *model.output_features,
                model.time_column,
            )
            if (
                not isinstance(model.input_features, tuple)
                or not isinstance(model.output_features, tuple)
                or not all(isinstance(name, str) for name in names)
            ):
                raise ValueError(
                    "Location features must be tuples of column names"
                )
            selections(
                model.input_features,
                model.output_features,
                model.time_column,
                model.order,
            )
            if not np.isfinite(model.dt) or model.dt <= 0:
                raise ValueError(
                    "Location sampling period must be finite and positive"
                )
            if not isinstance(model.nonlinear_terms, tuple) or not all(
                callable(term) for term in model.nonlinear_terms
            ):
                raise ValueError(
                    "Location nonlinear terms must be a tuple of callables"
                )
            if (
                model.order != template.order
                or model.input_features != template.input_features
                or model.output_features != template.output_features
                or model.time_column != template.time_column
                or model.dt != template.dt
                or len(model.nonlinear_terms) != len(template.nonlinear_terms)
                or any(
                    term is not other
                    for term, other in zip(
                        model.nonlinear_terms,
                        template.nonlinear_terms,
                        strict=True,
                    )
                )
            ):
                raise ValueError(
                    "Locations must share a NARX feature template"
                )
            count = (
                len(model.nonlinear_terms)
                + model.order * len(model.output_features)
                + len(model.input_features)
                + 1
            )
            if (
                not isinstance(model.coefficients, np.ndarray)
                or model.coefficients.shape
                != (len(model.output_features), count)
                or not np.all(np.isfinite(model.coefficients))
                or not np.isfinite(model.rmse)
                or model.rmse < 0
            ):
                raise ValueError(
                    "Location coefficients and RMSE must be finite and usable"
                )
        if (
            not isinstance(self.trace_lengths, tuple)
            or not self.trace_lengths
            or any(
                isinstance(length, bool)
                or not isinstance(length, int)
                or length <= template.order
                for length in self.trace_lengths
            )
        ):
            raise ValueError(
                "Trace lengths must be integers exceeding the history order"
            )
        if not isinstance(self.segments, tuple):
            raise ValueError("Segments must be an ordered tuple")
        stops = [0] * len(self.trace_lengths)
        last_trace = 0
        used = set()
        for segment in self.segments:
            if not isinstance(segment, DainarxSegment) or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (
                    segment.trace_index,
                    segment.start,
                    segment.stop,
                    segment.location_id,
                )
            ):
                raise ValueError(
                    "Segments must have integer trace, row, and location IDs"
                )
            index = segment.trace_index
            if (
                not last_trace <= index < len(stops)
                or segment.start != stops[index]
                or not segment.start
                < segment.stop
                <= self.trace_lengths[index]
                or not 0 <= segment.location_id < len(self.locations)
            ):
                raise ValueError(
                    "Segments must cover traces in order with valid location IDs"
                )
            stops[index] = segment.stop
            last_trace = index
            used.add(segment.location_id)
        if tuple(stops) != self.trace_lengths or used != set(
            range(len(self.locations))
        ):
            raise ValueError(
                "Segments must cover every observation and learned location"
            )

    def to_location_ids(self) -> list[np.ndarray]:
        """Project discovered membership onto all original observation rows.

        Returns:
            Fresh, fully populated int64 arrays of shape ``(trace_length,)``,
                one per independent trace in original order. Segment warmup
                rows are included; equal adjacent labels do not erase the
                boundaries retained in ``segments``.
        """
        labels = [
            np.empty(length, dtype=np.int64) for length in self.trace_lengths
        ]
        for segment in self.segments:
            labels[segment.trace_index][segment.start : segment.stop] = (
                segment.location_id
            )
        return labels

    def _validate_frames(self, traces: Sequence[pl.DataFrame]) -> None:
        if len(traces) != len(self.trace_lengths):
            raise ValueError(
                "Supply the original number of independent traces"
            )
        for index, (trace, length) in enumerate(
            zip(traces, self.trace_lengths, strict=True)
        ):
            if not isinstance(trace, pl.DataFrame):
                raise ValueError(f"Trace {index} must be a Polars DataFrame")
            if trace.height != length:
                raise ValueError(
                    f"Trace {index} must have its original observation count"
                )

    def to_labeled_frames(
        self, traces: Sequence[pl.DataFrame]
    ) -> list[pl.DataFrame]:
        """Add discovered ``location_id`` membership to original observations.

        Args:
            traces: Original Polars frames in the same independent-trace order
                and row correspondence as discovery. Count, types, heights,
                and reserved-column collisions are checked, but same-length
                reordered or changed observations cannot be detected.

        Returns:
            Independent frames preserving every original column, column order,
                and dtype, followed by a non-null Int64 ``location_id`` column.
                Labels describe membership, including segment warmup rows,
                not ground truth or a final-fit error certificate.
        """
        self._validate_frames(traces)
        for trace in traces:
            if "location_id" in trace.columns:
                raise ValueError(
                    "'location_id' is reserved for discovered membership"
                )
        return [
            trace.with_columns(
                pl.Series("location_id", labels, dtype=pl.Int64)
            )
            for trace, labels in zip(
                traces, self.to_location_ids(), strict=True
            )
        ]


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
    eta: float | None = None,
    guard_degree: int = 3,
    guard_coef0: float = 0.0,
) -> DainarxFit:
    """Infer sampled NARX locations, SVM guards, and post-switch predictors.

    Compose ``fit_dainarx_dynamics`` and ``fit_dainarx_transitions`` into a
    complete executable model. Use those stages separately to inspect dynamics
    and annotate observations even when transition fitting later fails.

    Fitting acceptance uses the largest per-output root-mean-square error
    (RMSE), rather than the paper's unnormalized L2 residual. By default, the
    threshold is scaled from the shared sampling period and the largest
    absolute selected training output across all traces. See the
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
        eta: Finite nonnegative acceptance threshold in output units for the
            maximum per-output training RMSE. Applies to screening, clustering,
            location fits, and reset fits; it is not a rollout error guarantee.
            ``None`` uses the automatic threshold scaled from the sampling
            period and training output magnitude; zero is accepted explicitly.
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
    dynamics = fit_dainarx_dynamics(
        traces,
        input_features=input_features,
        output_features=output_features,
        time_column=time_column,
        order=order,
        window=window,
        nonlinear_terms=nonlinear_terms,
        eta=eta,
    )
    transitions = fit_dainarx_transitions(
        traces,
        dynamics,
        guard_kernel=guard_kernel,
        guard_c=guard_c,
        guard_gamma=guard_gamma,
        guard_degree=guard_degree,
        guard_coef0=guard_coef0,
    )
    return DainarxFit(
        DainarxModel(dynamics.locations, transitions),
        dynamics.segments,
        dynamics.eta,
    )


def _fitted(
    rows: FloatArray,
    targets: FloatArray,
    context: str,
    *,
    order: int,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    time_column: str,
    dt: float,
    terms: tuple[NonlinearTerm, ...],
    eta: float,
) -> NarxModel:
    """Apply the same evidence and RMSE gates to ordinary and reset fits."""
    coefficient_count = len(terms) + order * len(outputs) + len(inputs) + 1
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


def fit_dainarx_dynamics(
    traces: Sequence[pl.DataFrame],
    *,
    input_features: Sequence[str],
    output_features: Sequence[str],
    time_column: str,
    order: int,
    window: int,
    nonlinear_terms: Sequence[NonlinearTerm] = (),
    eta: float | None = None,
) -> DainarxDynamics:
    """Discover ordinary NARX locations and complete observation membership.

    Screen segments, cluster pooled regression data, and fit ordinary location
    models using maximum per-output training RMSE acceptance. Each segment's
    first ``order`` rows are excluded from ordinary fitting, but included in
    annotations. Same-location boundaries are retained, not coalesced or
    rejected here: transition fitting determines whether resets are supported.

    Args:
        traces: Independent ordered Polars frames with finite numeric selected
            columns and a shared uniform sampling period. Start times and
            lengths may differ; unselected columns are ignored.
        input_features: Input column names, in input-vector order.
        output_features: Nonempty output column names, in output-vector order.
        time_column: Numeric timestamp column, strictly increasing per trace.
        order: Number of preceding output observations used for prediction.
        window: Screening window observation count. Each trace needs at least
            this many rows; ``window - order`` must exceed the per-output
            coefficient count, ``len(nonlinear_terms) + order *
            len(output_features) + len(input_features) + 1``. Whole segments
            also need more targets than that count; rank deficiency is allowed.
        nonlinear_terms: Ordered regression feature functions following the
            [NonlinearTerm][flowcean.hybrid.dainarx.NonlinearTerm] contract.
        eta: Finite nonnegative maximum per-output training RMSE threshold in
            output units. ``None`` uses the automatic threshold scaled from
            the sampling period and largest absolute selected training output
            across traces; explicit zero is honored. Applies
            to screening, clustering, and final location fitting, not rollout.

    Returns:
        Complete [DainarxDynamics][flowcean.hybrid.dainarx.DainarxDynamics],
            not a partial executable hybrid model. Discovery raises
            ``ValueError`` on insufficient or rejected segment evidence;
            transition learning may fail separately after discovery succeeds.
    """
    if eta is not None and (not np.isfinite(eta) or eta < 0):
        raise ValueError("eta must be finite and nonnegative")
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
    xs, designs = [], []
    dt = None
    for index, trace in enumerate(traces):
        dt = sampling_period(trace, time_column, dt)
        x, u = columns(trace, outputs), columns(trace, inputs)
        if len(x) < window:
            raise ValueError(f"Trace {index} is shorter than window")
        xs.append(x)
        designs.append(design(x, u, order, terms))
    if not xs or dt is None:
        raise ValueError("Supply at least one sampled trace")
    if eta is None:
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

    locations, assignments = [], {}
    for location, group in enumerate(groups):
        locations.append(
            _fitted(
                np.concatenate([segment_rows[i][0] for i in group]),
                np.concatenate([segment_rows[i][1] for i in group]),
                f"Location {location}",
                order=order,
                inputs=inputs,
                outputs=outputs,
                time_column=time_column,
                dt=dt,
                terms=terms,
                eta=eta,
            )
        )
        assignments.update(dict.fromkeys(group, location))
    segments = tuple(
        DainarxSegment(index, start, stop, assignments[i])
        for i, (index, start, stop) in enumerate(bounds)
    )
    return DainarxDynamics(
        tuple(locations), segments, tuple(len(x) for x in xs), eta
    )


def fit_dainarx_transitions(
    traces: Sequence[pl.DataFrame],
    dynamics: DainarxDynamics,
    *,
    guard_kernel: Literal["linear", "poly", "rbf", "sigmoid"],
    guard_c: float,
    guard_gamma: Literal["scale", "auto"] | float,
    guard_degree: int = 3,
    guard_coef0: float = 0.0,
) -> tuple[DainarxTransition, ...]:
    """Learn guards and post-switch predictors without repeating discovery.

    Guards use current selected outputs followed by current selected inputs.
    Each source-conditioned binary target asks whether the next observation
    belongs to the edge's destination. Reset predictors use preceding output
    history and the input aligned with the target row, one per post-switch
    phase. They share ordinary fitting's coefficient-count and RMSE gates.

    Args:
        traces: The same original observations used for dynamics discovery,
            in the same independent-trace order and row correspondence, not
            arbitrary frames of matching length. Frame types, counts, heights,
            finite numeric selected columns, and the shared uniform time grid
            are checked. Changed observations or reordered same-length traces
            cannot be detected without retaining the original data.
        dynamics: Complete discovery result supplying locations, segment
            provenance, trace lengths, feature template, and acceptance eta.
        guard_kernel: SVM kernel: ``"linear"``, ``"poly"``, ``"rbf"``, or
            ``"sigmoid"``.
        guard_c: Positive inverse regularization strength.
        guard_gamma: Nonnegative kernel scale coefficient or ``"scale"`` or
            ``"auto"``; ignored for the linear kernel.
        guard_degree: Nonnegative polynomial degree, used only for ``"poly"``.
        guard_coef0: Additive constant for ``"poly"`` and ``"sigmoid"`` kernels.

    Returns:
        Learned transitions, ready for ``DainarxModel(dynamics.locations,
            transitions)``. No segmentation, clustering, or ordinary fitting
            is repeated. Nonlinear features are evaluated only for required
            post-switch targets, not all ordinary regression rows again.

    Same-location boundaries, insufficient guard/reset evidence, incomplete
    reset contexts, rejected reset RMSE, and guard fitting errors still fail
    rather than inventing transitions or returning a partial fit.
    """
    if not isinstance(dynamics, DainarxDynamics):
        raise ValueError("Supply a DainarxDynamics discovery result")
    dynamics._validate_frames(traces)
    template = dynamics.locations[0]
    order = template.order
    xs, us = [], []
    for trace in traces:
        sampling_period(trace, template.time_column, template.dt)
        xs.append(columns(trace, template.output_features))
        us.append(columns(trace, template.input_features))
    for previous, current in pairwise(dynamics.segments):
        if (
            previous.trace_index == current.trace_index
            and previous.location_id == current.location_id
        ):
            raise ValueError(
                f"Trace {current.trace_index} boundary at row {current.start}: "
                f"unsupported same-location reset in location {current.location_id}"
            )
    labels = dynamics.to_location_ids()
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
        for x, u, label in zip(xs, us, labels, strict=True):
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
                    phase_rows[phase].append(
                        design_row(
                            x[target - order : target],
                            u[target],
                            template.nonlinear_terms,
                        )
                    )
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
            _fitted(
                np.stack(rows),
                np.stack(targets),
                f"Transition {source}->{destination} reset phase {phase + 1}",
                order=order,
                inputs=template.input_features,
                outputs=template.output_features,
                time_column=template.time_column,
                dt=template.dt,
                terms=template.nonlinear_terms,
                eta=dynamics.eta,
            )
            for phase, (rows, targets) in enumerate(
                zip(phase_rows, phase_targets, strict=True)
            )
        )
        transitions.append(
            DainarxTransition(source, destination, guard, predictors)
        )
    return tuple(transitions)
