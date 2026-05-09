from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from flowcean.core.learner import (
    SupervisedIncrementalLearner,
    SupervisedLearner,
)
from flowcean.hydra.callbacks import (
    HyDRACallback,
    HyDRACandidateFit,
    HyDRAGroupingEvaluation,
    HyDRAGroupingTrace,
    NoOpCallback,
)
from flowcean.hydra.model import HyDRAModel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from flowcean.core.model import Model
    from flowcean.hydra.schema import HyDRATraceSchema
    from flowcean.hydra.selector import (
        HybridDecisionTreeLearner,
        HybridDecisionTreeModel,
    )

logger = logging.getLogger(__name__)
UNLABELED_MODE = -1


class SelectorTrainingIncompleteError(ValueError):
    """Raised when selector learning sees partially labeled traces."""


@dataclass(frozen=True)
class TraceSegment:
    trace_index: int
    start_index: int
    end_index: int


@dataclass(frozen=True)
class HyDRATrace:
    frame: pl.DataFrame
    mode_labels: np.ndarray

    def __post_init__(self) -> None:
        if self.mode_labels.ndim != 1:
            message = "HyDRA trace mode labels must be a 1D array."
            raise ValueError(message)
        if self.mode_labels.size != self.frame.height:
            message = "HyDRA trace mode labels must match frame height."
            raise ValueError(message)
        if "mode" in self.frame.columns:
            message = "HyDRA trace frames must not store mode labels."
            raise ValueError(message)

    @classmethod
    def unlabeled(cls, frame: pl.DataFrame) -> HyDRATrace:
        return cls(
            frame=frame,
            mode_labels=np.full(frame.height, UNLABELED_MODE, dtype=np.int64),
        )

    @property
    def height(self) -> int:
        return self.frame.height

    @property
    def unlabeled_mask(self) -> np.ndarray:
        return self.mode_labels == UNLABELED_MODE

    def unlabeled_indices(self) -> list[int]:
        return np.flatnonzero(self.unlabeled_mask).tolist()

    def with_mode_labels(self, mode_labels: np.ndarray) -> HyDRATrace:
        return HyDRATrace(self.frame, mode_labels.copy())

    def with_labeled_segment(
        self,
        *,
        start_index: int,
        end_index: int,
        mode_id: int,
    ) -> HyDRATrace:
        mode_labels = self.mode_labels.copy()
        mode_labels[start_index : end_index + 1] = mode_id
        return self.with_mode_labels(mode_labels)

    def to_labeled_frame(self) -> pl.DataFrame:
        mode_values = [
            None if label == UNLABELED_MODE else int(label)
            for label in self.mode_labels
        ]
        return self.frame.with_columns(pl.Series("mode", mode_values))

    def segment_frame(self, segment: TraceSegment) -> pl.DataFrame:
        return self.frame.slice(
            segment.start_index,
            segment.end_index - segment.start_index + 1,
        )


@dataclass(frozen=True)
class ModeLabelingResult:
    traces: list[HyDRATrace]
    accepted_rows: pl.DataFrame
    grouping: HyDRAGroupingEvaluation


@dataclass(frozen=True)
class LearnedModes:
    traces: list[HyDRATrace]
    models: list[Model]


class HyDRALearner(SupervisedLearner):
    """Identify hybrid-system modes from trace inputs and derivatives.

    ``regressor_factory`` must create fresh incremental supervised learners.
    The current learner supports single-output derivative training. When a
    selector learner is provided, HyDRA labels accurate trace segments and
    trains a selector to route future rows to learned modes.
    """

    regressor_factory: Callable[[], SupervisedIncrementalLearner]
    threshold: float
    start_width: int
    step_width: int
    selector_learner: HybridDecisionTreeLearner | None
    callback: HyDRACallback
    trace_schema: HyDRATraceSchema | None

    def __init__(
        self,
        regressor_factory: Callable[[], SupervisedIncrementalLearner],
        threshold: float,
        start_width: int = 10,
        step_width: int = 5,
        selector_learner: HybridDecisionTreeLearner | None = None,
        callback: HyDRACallback | None = None,
        trace_schema: HyDRATraceSchema | None = None,
    ) -> None:
        super().__init__()
        if threshold < 0:
            message = "threshold must be non-negative."
            raise ValueError(message)
        if start_width <= 0:
            message = "start_width must be positive."
            raise ValueError(message)
        if step_width <= 0:
            message = "step_width must be positive."
            raise ValueError(message)
        self.regressor_factory = regressor_factory
        self.threshold = threshold
        self.start_width = start_width
        self.step_width = step_width
        self.selector_learner = selector_learner
        self.callback = callback or NoOpCallback()
        self.trace_schema = trace_schema

    def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> HyDRAModel:
        input_columns = inputs.collect_schema().names()
        output_columns = outputs.collect_schema().names()
        self._validate_trace_schema_for_learning(input_columns, output_columns)
        traces = self._initialize_traces(inputs, outputs)
        self.callback.start(
            trace_count=len(traces),
            threshold=self.threshold,
            start_width=self.start_width,
            step_width=self.step_width,
        )

        learned_modes = self._discover_modes(
            traces=traces,
            input_columns=input_columns,
            output_columns=output_columns,
        )
        if not learned_modes.models:
            message = "No modes were identified during HyDRA learning."
            raise ValueError(message)

        selector = self._learn_selector(
            learned_modes.traces,
            learned_modes.models,
        )
        model = HyDRAModel(
            learned_modes.models,
            input_features=input_columns,
            output_features=output_columns,
            selector=selector,
            trace_schema=self.trace_schema,
        )
        self.callback.finish(final_mode_count=len(learned_modes.models))
        return model

    def _validate_trace_schema_for_learning(
        self,
        input_columns: list[str],
        output_columns: list[str],
    ) -> None:
        if self.trace_schema is None:
            return
        self.trace_schema.validate_input_features(input_columns)
        self.trace_schema.validate_output_features(output_columns)
        self.trace_schema.validate_state_derivative_width()
        if len(self.trace_schema.derivative) != 1:
            message = (
                "HyDRALearner currently supports single-output training only."
            )
            raise ValueError(message)

    def _initialize_traces(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> list[HyDRATrace]:
        output_columns = outputs.collect_schema().names()
        if len(output_columns) != 1:
            message = (
                "HyDRALearner currently supports single-output training only."
            )
            raise ValueError(message)

        frame = pl.concat([inputs, outputs], how="horizontal").collect()
        if frame.height == 0:
            message = (
                "HyDRALearner requires at least one row of training data."
            )
            raise ValueError(message)
        return [HyDRATrace.unlabeled(frame)]

    def _discover_modes(
        self,
        traces: Sequence[HyDRATrace],
        input_columns: list[str],
        output_columns: list[str],
    ) -> LearnedModes:
        labeled_traces = list(traces)
        learned_models: list[Model] = []

        pending_segment = find_next_pending_segment(labeled_traces)
        while pending_segment is not None:
            logger.info("Current pending segment: %s", pending_segment)
            self.callback.pending_segment_found(pending_segment)

            mode_learner = self.regressor_factory()
            mode_id = len(learned_models)
            triggering_trace = labeled_traces[pending_segment.trace_index]
            candidate_model = self._fit_candidate_mode(
                trace_frame=triggering_trace.segment_frame(pending_segment),
                learner=mode_learner,
                input_columns=input_columns,
                output_columns=output_columns,
                trace_index=pending_segment.trace_index,
                segment_start_index=pending_segment.start_index,
            )
            if candidate_model is None:
                self._stop_learning(
                    pending_segment,
                    "flow_identification_failed",
                )
                break

            before_labeling = labeled_traces
            labeling = label_matching_rows(
                traces=labeled_traces,
                model=candidate_model,
                input_columns=input_columns,
                output_columns=output_columns,
                threshold=self.threshold,
                mode_id=mode_id,
                triggering_segment=pending_segment,
            )
            labeled_traces = labeling.traces
            self.callback.grouping_evaluated(labeling.grouping)

            if labeling.accepted_rows.is_empty():
                self._stop_learning(
                    pending_segment,
                    "no_accurate_segments_found",
                )
                break

            finalized_model = mode_learner.learn_incremental(
                pl.LazyFrame(labeling.accepted_rows[input_columns]),
                pl.LazyFrame(labeling.accepted_rows[output_columns]),
            )
            learned_models.append(finalized_model)
            self.callback.mode_finalized(
                mode_id=mode_id,
                triggering_segment=pending_segment,
                accepted_segments=_new_mode_segments(
                    before=before_labeling,
                    after=labeled_traces,
                    mode_id=mode_id,
                ),
            )
            logger.info(
                "Learned mode %d with model %s",
                mode_id,
                finalized_model,
            )
            pending_segment = find_next_pending_segment(labeled_traces)

        return LearnedModes(traces=labeled_traces, models=learned_models)

    def _fit_candidate_mode(
        self,
        trace_frame: pl.DataFrame,
        learner: SupervisedIncrementalLearner,
        input_columns: list[str],
        output_columns: list[str],
        *,
        trace_index: int,
        segment_start_index: int,
    ) -> Model | None:
        if trace_frame.height < self.start_width:
            return learner.learn_incremental(
                pl.LazyFrame(trace_frame[input_columns]),
                pl.LazyFrame(trace_frame[output_columns]),
            )

        best_candidate_fit: HyDRACandidateFit | None = None
        best_model: Model | None = None
        window_size = self.start_width - self.step_width
        while window_size < trace_frame.height:
            window_size = min(
                window_size + self.step_width,
                trace_frame.height,
            )
            window_frame = trace_frame.slice(0, window_size)
            candidate_model = learner.learn_incremental(
                pl.LazyFrame(window_frame[input_columns]),
                pl.LazyFrame(window_frame[output_columns]),
            )
            candidate_fit = _candidate_fit_for_window(
                trace_index=trace_index,
                start_index=segment_start_index,
                frame=window_frame,
                target_column=output_columns[0],
                prediction=candidate_model.predict(
                    window_frame[input_columns],
                ).collect(),
                threshold=self.threshold,
            )
            self.callback.candidate_window_evaluated(candidate_fit)
            logger.info(
                "Window size %d produced fit %s",
                candidate_fit.window_size,
                candidate_fit.fit,
            )
            if candidate_fit.fit < self.threshold:
                best_candidate_fit = candidate_fit
                best_model = deepcopy(candidate_model)
                if window_size == trace_frame.height:
                    break
                continue
            break

        if best_model is None or best_candidate_fit is None:
            logger.error(
                "Flow Identification failed: required accuracy not met "
                "with threshold %.2f and start_width %d.",
                self.threshold,
                self.start_width,
            )
            return None

        self.callback.candidate_selected(best_candidate_fit)
        logger.info(
            "Selected window fit %s below threshold %.2f.",
            best_candidate_fit.fit,
            self.threshold,
        )
        return best_model

    def _learn_selector(
        self,
        traces: Sequence[HyDRATrace],
        modes: list[Model],
    ) -> HybridDecisionTreeModel | None:
        if self.selector_learner is None:
            return None

        selector_traces = [trace.to_labeled_frame() for trace in traces]
        if any(trace["mode"].null_count() > 0 for trace in selector_traces):
            message = "selector training requires fully labeled HyDRA traces"
            raise SelectorTrainingIncompleteError(message)
        return self.selector_learner.learn_from_traces(
            selector_traces,
            mode_to_flow=dict(enumerate(modes)),
        )

    def _stop_learning(self, segment: TraceSegment, reason: str) -> None:
        logger.error(
            "Flow Identification failed. Stopping HyDRA learning: {reason}",
        )
        self.callback.learning_stopped(
            segment=segment,
            reason=reason,
        )


def _series_to_float_list(series: pl.Series) -> list[float]:
    return [float(value) for value in series.to_list()]


def _single_output_errors(
    actual: pl.Series,
    predicted: pl.Series,
) -> list[float]:
    return [
        abs(float(actual_value) - float(predicted_value))
        for actual_value, predicted_value in zip(
            actual.to_list(),
            predicted.to_list(),
            strict=True,
        )
    ]


def _candidate_fit_for_window(
    *,
    trace_index: int,
    start_index: int,
    frame: pl.DataFrame,
    target_column: str,
    prediction: pl.DataFrame,
    threshold: float,
) -> HyDRACandidateFit:
    actual = frame[target_column]
    fitted = prediction[target_column]
    errors = _single_output_errors(actual, fitted)
    return HyDRACandidateFit(
        segment=TraceSegment(
            trace_index=trace_index,
            start_index=start_index,
            end_index=start_index + frame.height - 1,
        ),
        window_size=frame.height,
        threshold=threshold,
        fit=max(errors) if errors else 0.0,
        actual_derivative=_series_to_float_list(actual),
        fitted_derivative=_series_to_float_list(fitted),
        errors=errors,
    )


def find_next_pending_segment(
    traces: Sequence[HyDRATrace],
) -> TraceSegment | None:
    """Find the first contiguous unlabeled segment across traces."""
    for trace_index, trace in enumerate(traces):
        start_index: int | None = None
        for row_index, is_unlabeled in enumerate(trace.unlabeled_mask):
            if is_unlabeled and start_index is None:
                start_index = row_index
            elif not is_unlabeled and start_index is not None:
                return TraceSegment(
                    trace_index=trace_index,
                    start_index=start_index,
                    end_index=row_index - 1,
                )
        if start_index is not None:
            return TraceSegment(
                trace_index=trace_index,
                start_index=start_index,
                end_index=trace.height - 1,
            )
    return None


def label_matching_rows(
    *,
    traces: Sequence[HyDRATrace],
    model: Model,
    input_columns: list[str],
    output_columns: list[str],
    threshold: float,
    mode_id: int,
    triggering_segment: TraceSegment,
) -> ModeLabelingResult:
    updated_traces: list[HyDRATrace] = []
    accepted_frames: list[pl.DataFrame] = []
    grouping_traces: list[HyDRAGroupingTrace] = []

    for trace_index, trace in enumerate(traces):
        predictions = model.predict(trace.frame[input_columns]).collect()
        row_errors = np.abs(
            predictions.select(output_columns).to_numpy()
            - trace.frame.select(output_columns).to_numpy(),
        )
        max_errors = (
            row_errors.max(axis=1) if row_errors.size else np.array([])
        )
        unlabeled_indices = trace.unlabeled_indices()
        accepted_unlabeled_mask = trace.unlabeled_mask & (
            max_errors < threshold
        )

        grouping_traces.append(
            _grouping_trace(
                trace_index=trace_index,
                trace=trace,
                predictions=predictions,
                target_column=output_columns[0],
                unlabeled_indices=unlabeled_indices,
                max_errors=max_errors,
                threshold=threshold,
            ),
        )

        updated_trace = trace.with_mode_labels(trace.mode_labels)
        for segment in _segments_from_mask(accepted_unlabeled_mask):
            updated_trace = updated_trace.with_labeled_segment(
                start_index=segment.start_index,
                end_index=segment.end_index,
                mode_id=mode_id,
            )
            accepted_frames.append(
                updated_trace.to_labeled_frame()[
                    segment.start_index : segment.end_index + 1
                ],
            )
            logger.info(
                "Found accurate segment in trace %d: %d to %d",
                trace_index,
                segment.start_index,
                segment.end_index,
            )
        updated_traces.append(updated_trace)

    grouping = HyDRAGroupingEvaluation(
        mode_id=mode_id,
        threshold=threshold,
        triggering_segment=triggering_segment,
        traces=tuple(grouping_traces),
    )
    return ModeLabelingResult(
        traces=updated_traces,
        accepted_rows=(
            pl.concat(accepted_frames, how="vertical")
            if accepted_frames
            else _empty_labeled_frame(traces)
        ),
        grouping=grouping,
    )


def _grouping_trace(
    *,
    trace_index: int,
    trace: HyDRATrace,
    predictions: pl.DataFrame,
    target_column: str,
    unlabeled_indices: Sequence[int],
    max_errors: np.ndarray,
    threshold: float,
) -> HyDRAGroupingTrace:
    return HyDRAGroupingTrace(
        trace_index=trace_index,
        row_indices=list(unlabeled_indices),
        accepted_mask=[
            bool(max_errors[row_index] < threshold)
            for row_index in unlabeled_indices
        ],
        actual_derivative=[
            float(trace.frame[target_column][row_index])
            for row_index in unlabeled_indices
        ],
        fitted_derivative=[
            float(predictions[target_column][row_index])
            for row_index in unlabeled_indices
        ],
        errors=[
            float(max_errors[row_index]) for row_index in unlabeled_indices
        ],
    )


def _empty_labeled_frame(traces: Sequence[HyDRATrace]) -> pl.DataFrame:
    if not traces:
        return pl.DataFrame()
    return traces[0].to_labeled_frame().head(0)


def _new_mode_segments(
    before: Sequence[HyDRATrace],
    after: Sequence[HyDRATrace],
    mode_id: int,
) -> list[TraceSegment]:
    segments: list[TraceSegment] = []
    for trace_index, (before_trace, after_trace) in enumerate(
        zip(before, after, strict=True),
    ):
        new_mode_mask = (before_trace.mode_labels == UNLABELED_MODE) & (
            after_trace.mode_labels == mode_id
        )
        segments.extend(
            TraceSegment(
                trace_index=trace_index,
                start_index=segment.start_index,
                end_index=segment.end_index,
            )
            for segment in _segments_from_mask(new_mode_mask)
        )
    return segments


def _segments_from_mask(mask: np.ndarray) -> list[TraceSegment]:
    segments: list[TraceSegment] = []
    start_index: int | None = None
    for row_index, selected in enumerate(mask):
        if selected and start_index is None:
            start_index = row_index
        elif not selected and start_index is not None:
            segments.append(TraceSegment(0, start_index, row_index - 1))
            start_index = None
    if start_index is not None:
        segments.append(TraceSegment(0, start_index, len(mask) - 1))
    return segments
