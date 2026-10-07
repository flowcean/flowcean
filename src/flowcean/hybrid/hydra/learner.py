from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from flowcean.core.learner import SupervisedLearner

from .callbacks import HyDRACallback

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from flowcean.core.model import Model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceSegment:
    """Consecutive observations within a trace: start inclusive, stop exclusive.

    Args:
        trace_index: Index in the supplied trace collection.
        start: Index of the first included observation.
        stop: Index just past the last observation.
    """

    trace_index: int
    start: int
    stop: int


class HyDRAIdentificationError(RuntimeError):
    """Discovery could not assign the supplied observations in a segment.

    Attributes:
        segment: The failing trace segment, with half-open row bounds.
    """

    segment: TraceSegment

    def __init__(self, segment: TraceSegment, message: str) -> None:
        self.segment = segment
        super().__init__(
            f"Trace {segment.trace_index} [{segment.start}, {segment.stop}): "
            f"{message}"
        )


@dataclass(frozen=True)
class LearnedFlow:
    """A fitted model and the observation segments accepted by its candidate.

    Segments use half-open row bounds and are ordered by trace, then row.
    Membership is recorded before the final model refit.
    """

    model: Model
    segments: tuple[TraceSegment, ...]


@dataclass(frozen=True)
class LearnedFlows:
    """Discovered flows and their membership in the supplied observations.

    Args:
        flows: Fitted models with accepted segments. A flow's position is its
            shared flow ID across all traces.
        trace_lengths: Row counts in the original trace order.
        input_features: Ordered input column names used for fitting.
        output_features: Output column names used for fitting.

    Accepted segments cover every supplied observation exactly once.
    Use ``to_labeled_frames(original_traces)`` to train a selector, or
    ``to_flow_ids()`` for row-aligned NumPy assignments.
    """

    flows: tuple[LearnedFlow, ...]
    trace_lengths: tuple[int, ...]
    input_features: tuple[str, ...]
    output_features: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.trace_lengths or any(
            isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
            for length in self.trace_lengths
        ):
            raise ValueError(
                "Trace lengths must be nonempty positive integers."
            )
        intervals: list[list[TraceSegment]] = [[] for _ in self.trace_lengths]
        for flow in self.flows:
            if not flow.segments:
                raise ValueError(
                    "Each learned flow requires accepted segments."
                )
            for segment in flow.segments:
                if (
                    any(
                        isinstance(value, bool) or not isinstance(value, int)
                        for value in (
                            segment.trace_index,
                            segment.start,
                            segment.stop,
                        )
                    )
                    or not 0 <= segment.trace_index < len(self.trace_lengths)
                    or not 0
                    <= segment.start
                    < segment.stop
                    <= self.trace_lengths[segment.trace_index]
                ):
                    raise ValueError(
                        "Accepted segment has invalid trace bounds."
                    )
                intervals[segment.trace_index].append(segment)
        # Sweep sparse interval endpoints across all flows within each trace.
        for segments, length in zip(
            intervals, self.trace_lengths, strict=True
        ):
            previous_stop = 0
            for segment in sorted(segments, key=lambda item: item.start):
                if segment.start < previous_stop:
                    raise ValueError("Accepted segments must not overlap.")
                if segment.start > previous_stop:
                    raise ValueError("Accepted segments must cover every row.")
                previous_stop = segment.stop
            if previous_stop != length:
                raise ValueError("Accepted segments must cover every row.")

    def to_flow_ids(self) -> list[np.ndarray]:
        """Return fresh, fully populated int64 arrays of row assignments."""
        flow_ids = [
            np.empty(length, dtype=np.int64) for length in self.trace_lengths
        ]
        for flow_id, flow in enumerate(self.flows):
            for segment in flow.segments:
                flow_ids[segment.trace_index][segment.start : segment.stop] = (
                    flow_id
                )
        return flow_ids

    def to_labeled_frames(
        self, traces: Sequence[pl.DataFrame]
    ) -> list[pl.DataFrame]:
        """Add an Int64 ``flow_id`` column to each supplied frame.

        Frames must match the original trace count, row counts, and positional
        order. All existing columns are preserved; every row has a flow ID.
        """
        if len(traces) != len(self.trace_lengths):
            raise ValueError("Trace count must match the discovery result.")
        for frame, length in zip(traces, self.trace_lengths, strict=True):
            if not isinstance(frame, pl.DataFrame):
                raise TypeError("HyDRA traces must be Polars DataFrames.")
            if frame.height != length:
                raise ValueError(
                    "Trace height must match the discovery result."
                )
            if "flow_id" in frame.columns:
                raise ValueError("flow_id is reserved for HyDRA assignments.")
        return [
            frame.with_columns(
                pl.Series(
                    "flow_id",
                    ids,
                    dtype=pl.Int64,
                )
            )
            for frame, ids in zip(traces, self.to_flow_ids(), strict=True)
        ]


class HyDRALearner:
    """Discover shared flow models from ordered traces with one scalar target.

    HyDRA fits candidates on growing windows of consecutive observations,
    groups matching observations across traces, and refits each discovered
    flow on its accepted group. Supply derivatives as targets to learn rates
    of change, or another regression target to learn its shared behaviors.

    Args:
        regressor_factory: Creates a fresh batch ``SupervisedLearner`` for
            every candidate window and final fit. Each instance receives one
            ``learn`` call with the observations for that fit.
        threshold: Finite, non-negative maximum absolute prediction error.
            Candidate acceptance uses the strict comparison error < threshold.
        start_width: Positive initial window size in consecutive observations.
        step_width: Positive number of observations added at each window step.
        callback: Observer for discovery progress; defaults to a no-op ``HyDRACallback``.
    """

    regressor_factory: Callable[[], SupervisedLearner]
    threshold: float
    start_width: int
    step_width: int
    callback: HyDRACallback

    def __init__(
        self,
        regressor_factory: Callable[[], SupervisedLearner],
        threshold: float,
        start_width: int = 10,
        step_width: int = 5,
        callback: HyDRACallback | None = None,
    ) -> None:
        if not np.isfinite(threshold) or threshold < 0:
            message = "threshold must be finite and non-negative."
            raise ValueError(message)
        if (
            isinstance(start_width, bool)
            or not isinstance(start_width, int)
            or start_width <= 0
        ):
            message = "start_width must be positive and an integer."
            raise ValueError(message)
        if (
            isinstance(step_width, bool)
            or not isinstance(step_width, int)
            or step_width <= 0
        ):
            message = "step_width must be positive and an integer."
            raise ValueError(message)
        self.regressor_factory = regressor_factory
        self.threshold = threshold
        self.start_width = start_width
        self.step_width = step_width
        self.callback = callback if callback is not None else HyDRACallback()

    def learn(
        self,
        traces: Sequence[pl.DataFrame],
        *,
        input_features: Sequence[str],
        output_features: Sequence[str],
    ) -> LearnedFlows:
        """Discover flows and return their models and observation membership.

        Args:
            traces: Nonempty collection of nonempty Polars frames in observation
                order. Each frame is an independent trace. Additional columns
                can be retained when projecting the result onto these frames.
            input_features: Nonempty, unique input column names.
            output_features: Exactly one target column, distinct from inputs.

        Selected columns must contain finite numeric values with no nulls.
        ``flow_id`` is reserved for assignments and must be absent from frames
        and feature lists. Fitting uses Float64 values for selected columns.

        Returns:
            Models and segments assigning every supplied observation to a flow.

        Raises:
            ValueError: Data or feature roles violate the stated constraints.
            TypeError: A supplied trace is not a Polars DataFrame.
            HyDRAIdentificationError: A pending segment cannot be identified.
                Its ``segment`` identifies the failing trace and row bounds.

        Backend and callback exceptions propagate to the caller.
        """
        input_columns = list(input_features)
        output_columns = list(output_features)
        prepared_traces = self._prepare_traces(
            traces,
            input_columns,
            output_columns,
        )
        logger.info(
            "HyDRA started: traces=%d, threshold=%g, start_width=%d, step_width=%d",
            len(prepared_traces),
            self.threshold,
            self.start_width,
            self.step_width,
        )
        self.callback.start(
            trace_count=len(prepared_traces),
            threshold=self.threshold,
            start_width=self.start_width,
            step_width=self.step_width,
        )

        learned_flows = self._discover_flows(
            traces=prepared_traces,
            input_columns=input_columns,
            output_columns=output_columns,
        )
        self.callback.finish(learned_flows)
        logger.info(
            "HyDRA finished: flows=%d",
            len(learned_flows.flows),
        )
        return learned_flows

    def _prepare_traces(
        self,
        frames: Sequence[pl.DataFrame],
        input_columns: list[str],
        output_columns: list[str],
    ) -> list[pl.DataFrame]:
        for role, columns in (
            ("input", input_columns),
            ("output", output_columns),
        ):
            if any(not isinstance(column, str) for column in columns):
                raise ValueError(f"{role} features must be column names.")
            if not columns or len(set(columns)) != len(columns):
                raise ValueError(
                    f"{role} features must be nonempty and unique."
                )
            if "flow_id" in columns:
                raise ValueError("flow_id is reserved for HyDRA assignments.")
        if set(input_columns) & set(output_columns):
            raise ValueError("input and output features must be disjoint.")
        if len(output_columns) != 1:
            raise ValueError(
                "HyDRALearner supports single-output training only."
            )
        if not frames:
            raise ValueError(
                "HyDRALearner requires a nonempty trace collection."
            )
        traces = []
        for trace_index, frame in enumerate(frames):
            if not isinstance(frame, pl.DataFrame):
                raise TypeError("HyDRA traces must be Polars DataFrames.")
            if not frame.height:
                raise ValueError("Each HyDRA trace requires at least one row.")
            if "flow_id" in frame.columns:
                raise ValueError("flow_id is reserved for HyDRA assignments.")
            for column in input_columns + output_columns:
                if column not in frame.columns:
                    raise ValueError(
                        f"Trace {trace_index} is missing column {column!r}."
                    )
                _validate_numeric_column(
                    frame[column], context=f"Trace {trace_index}"
                )
            traces.append(frame.select(input_columns + output_columns))
        return traces

    def _discover_flows(
        self,
        traces: Sequence[pl.DataFrame],
        input_columns: list[str],
        output_columns: list[str],
    ) -> LearnedFlows:
        trace_lengths = tuple(trace.height for trace in traces)
        pending = [
            [TraceSegment(trace_index, 0, length)]
            for trace_index, length in enumerate(trace_lengths)
        ]
        learned_flows: list[LearnedFlow] = []
        pending_segment = _first_pending_segment(pending)
        while pending_segment is not None:
            logger.info("Current pending segment: %s", pending_segment)
            self.callback.pending_segment_found(pending_segment)

            flow_id = len(learned_flows)
            triggering_trace = traces[pending_segment.trace_index]
            candidate_model = self._fit_candidate_flow(
                trace_frame=triggering_trace.slice(
                    pending_segment.start,
                    pending_segment.stop - pending_segment.start,
                ),
                input_columns=input_columns,
                output_columns=output_columns,
                trace_index=pending_segment.trace_index,
                segment_start_index=pending_segment.start,
            )
            if candidate_model is None:
                raise HyDRAIdentificationError(
                    pending_segment,
                    "Candidate accuracy did not meet the strict error "
                    f"threshold {self.threshold}.",
                )

            considered_count = sum(
                segment.stop - segment.start
                for intervals in pending
                for segment in intervals
            )
            accepted_segments = _group_matching_segments(
                traces=traces,
                pending=pending,
                model=candidate_model,
                input_columns=input_columns,
                output_columns=output_columns,
                threshold=self.threshold,
            )
            logger.info(
                "HyDRA grouping: flow_id=%d, accepted_rows=%d/%d",
                flow_id,
                sum(
                    segment.stop - segment.start
                    for segment in accepted_segments
                ),
                considered_count,
            )
            self.callback.grouping_evaluated(
                flow_id=flow_id,
                accepted_segments=accepted_segments,
                considered_count=considered_count,
            )

            if not accepted_segments:
                raise HyDRAIdentificationError(
                    pending_segment,
                    "Candidate grouping accepted no observations at the "
                    f"strict error threshold {self.threshold}.",
                )

            accepted_rows = pl.concat(
                [
                    _numeric_frame(
                        traces[segment.trace_index].slice(
                            segment.start, segment.stop - segment.start
                        ),
                        input_columns + output_columns,
                    )
                    for segment in accepted_segments
                ],
                how="vertical",
            )
            finalized_model = self._fit_rows(
                accepted_rows,
                input_columns,
                output_columns,
            )
            flow = LearnedFlow(finalized_model, accepted_segments)
            learned_flows.append(flow)
            pending = _subtract_segments(pending, accepted_segments)
            self.callback.flow_finalized(flow_id=flow_id, flow=flow)
            logger.info(
                "Learned flow %d with model %s",
                flow_id,
                finalized_model,
            )
            pending_segment = _first_pending_segment(pending)

        return LearnedFlows(
            flows=tuple(learned_flows),
            trace_lengths=trace_lengths,
            input_features=tuple(input_columns),
            output_features=tuple(output_columns),
        )

    def _fit_rows(
        self,
        frame: pl.DataFrame,
        input_columns: list[str],
        output_columns: list[str],
    ) -> Model:
        learner = self.regressor_factory()
        return learner.learn(
            _numeric_frame(frame, input_columns).lazy(),
            _numeric_frame(frame, output_columns).lazy(),
        )

    def _fit_candidate_flow(
        self,
        trace_frame: pl.DataFrame,
        input_columns: list[str],
        output_columns: list[str],
        *,
        trace_index: int,
        segment_start_index: int,
    ) -> Model | None:
        if trace_frame.height < self.start_width:
            return deepcopy(
                self._fit_rows(trace_frame, input_columns, output_columns)
            )

        best_fit: float | None = None
        best_segment: TraceSegment | None = None
        best_model: Model | None = None
        window_size = self.start_width - self.step_width
        while window_size < trace_frame.height:
            window_size = min(
                window_size + self.step_width,
                trace_frame.height,
            )
            window_frame = trace_frame.slice(0, window_size)
            candidate_model = self._fit_rows(
                window_frame, input_columns, output_columns
            )
            prediction = candidate_model.predict(
                _numeric_frame(window_frame, input_columns),
            ).collect()
            _validate_prediction(
                prediction, window_frame.height, output_columns
            )
            # Python scalar conversion differs from Polars Float64 conversion
            # for Decimal targets near strict threshold boundaries.
            fit = max(
                abs(float(actual) - float(predicted))
                for actual, predicted in zip(
                    window_frame[output_columns[0]],
                    prediction[output_columns[0]],
                    strict=True,
                )
            )
            segment = TraceSegment(
                trace_index,
                segment_start_index,
                segment_start_index + window_size,
            )
            self.callback.candidate_window_evaluated(segment=segment, fit=fit)
            logger.info(
                "Window size %d produced fit %s",
                window_size,
                fit,
            )
            if fit < self.threshold:
                best_fit = fit
                best_segment = segment
                best_model = deepcopy(candidate_model)
                if window_size == trace_frame.height:
                    break
                continue
            break

        if best_model is None or best_fit is None or best_segment is None:
            logger.error(
                "Flow Identification failed: required accuracy not met "
                "with threshold %.2f and start_width %d.",
                self.threshold,
                self.start_width,
            )
            return None

        self.callback.candidate_selected(segment=best_segment, fit=best_fit)
        logger.info(
            "Selected window fit %s below threshold %.2f.",
            best_fit,
            self.threshold,
        )
        return best_model


def _numeric_frame(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    # Give regressors a consistent numeric schema.
    return frame.select(columns).cast(pl.Float64)


def _validate_numeric_column(series: pl.Series, *, context: str) -> None:
    if not series.dtype.is_numeric():
        raise ValueError(f"{context} column {series.name!r} must be numeric.")
    if series.null_count() or not np.all(
        np.isfinite(series.cast(pl.Float64).to_numpy())
    ):
        raise ValueError(
            f"{context} column {series.name!r} must be finite with no nulls."
        )


def _validate_prediction(
    prediction: pl.DataFrame,
    row_count: int,
    output_columns: list[str],
) -> None:
    if prediction.height != row_count:
        raise ValueError("Backend prediction row count must match input rows.")
    for column in output_columns:
        if column not in prediction.columns:
            raise ValueError(
                f"Backend prediction is missing output column {column!r}."
            )
        _validate_numeric_column(
            prediction[column], context="Backend prediction"
        )


def _first_pending_segment(
    pending: Sequence[Sequence[TraceSegment]],
) -> TraceSegment | None:
    for segments in pending:
        if segments:
            return segments[0]
    return None


def _segments_from_mask(
    mask: np.ndarray, *, trace_index: int, start: int = 0
) -> list[TraceSegment]:
    segments: list[TraceSegment] = []
    run_start: int | None = None
    for row_index, selected in enumerate(mask):
        if selected and run_start is None:
            run_start = row_index
        elif not selected and run_start is not None:
            segments.append(
                TraceSegment(trace_index, start + run_start, start + row_index)
            )
            run_start = None
    if run_start is not None:
        segments.append(
            TraceSegment(trace_index, start + run_start, start + len(mask))
        )
    return segments


def _subtract_segments(
    pending: Sequence[Sequence[TraceSegment]],
    accepted: Sequence[TraceSegment],
) -> list[list[TraceSegment]]:
    """Subtract ordered accepted pieces, rejecting anything outside pending."""
    accepted_by_trace: list[list[TraceSegment]] = [[] for _ in pending]
    previous: TraceSegment | None = None
    for segment in accepted:
        if (
            not 0 <= segment.trace_index < len(pending)
            or segment.start >= segment.stop
            or (
                previous is not None
                and (
                    segment.trace_index < previous.trace_index
                    or (
                        segment.trace_index == previous.trace_index
                        and segment.start < previous.stop
                    )
                )
            )
        ):
            raise ValueError("Accepted segments must be nonempty and ordered.")
        accepted_by_trace[segment.trace_index].append(segment)
        previous = segment

    remaining: list[list[TraceSegment]] = []
    for trace_index, (intervals, pieces) in enumerate(
        zip(pending, accepted_by_trace, strict=True)
    ):
        result: list[TraceSegment] = []
        piece_index = 0
        for interval in intervals:
            cursor = interval.start
            while piece_index < len(pieces):
                piece = pieces[piece_index]
                if piece.start >= interval.stop:
                    break
                if piece.start < cursor or piece.stop > interval.stop:
                    raise ValueError(
                        "Accepted segment is outside pending coverage."
                    )
                if cursor < piece.start:
                    result.append(
                        TraceSegment(trace_index, cursor, piece.start)
                    )
                cursor = piece.stop
                piece_index += 1
            if cursor < interval.stop:
                result.append(TraceSegment(trace_index, cursor, interval.stop))
        if piece_index != len(pieces):
            raise ValueError("Accepted segment is outside pending coverage.")
        remaining.append(result)
    return remaining


def _group_matching_segments(
    *,
    traces: Sequence[pl.DataFrame],
    pending: Sequence[Sequence[TraceSegment]],
    model: Model,
    input_columns: list[str],
    output_columns: list[str],
    threshold: float,
) -> tuple[TraceSegment, ...]:
    accepted_segments: list[TraceSegment] = []
    target_column = output_columns[0]

    for trace_index, (trace, intervals) in enumerate(
        zip(traces, pending, strict=True)
    ):
        # Predict the whole trace, including assigned rows: some backends or
        # transforms depend on the context of the full input batch.
        predictions = model.predict(
            _numeric_frame(trace, input_columns)
        ).collect()
        _validate_prediction(predictions, trace.height, output_columns)
        errors = np.abs(
            predictions[target_column].cast(pl.Float64).to_numpy()
            - trace[target_column].cast(pl.Float64).to_numpy()
        )
        for interval in intervals:
            accepted_segments.extend(
                _segments_from_mask(
                    errors[interval.start : interval.stop] < threshold,
                    trace_index=trace_index,
                    start=interval.start,
                )
            )

    return tuple(accepted_segments)
