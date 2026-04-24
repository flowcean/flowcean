import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from flowcean.core.learner import (
    SupervisedIncrementalLearner,
    SupervisedLearner,
)
from flowcean.core.model import Model
from flowcean.hydra.callbacks import HyDRACallback
from flowcean.hydra.model import HyDRAModel
from flowcean.hydra.selector import (
    HybridDecisionTreeLearner,
    HybridDecisionTreeModel,
)

logger = logging.getLogger(__name__)


class SelectorTrainingIncompleteError(ValueError):
    """Raised when selector learning is attempted on partially labeled traces."""


@dataclass(frozen=True)
class PendingSegment:
    trace_index: int
    start_index: int
    end_index: int


@dataclass(frozen=True)
class AccurateSegmentResult:
    updated_traces: list[pl.DataFrame]
    accurate_rows: pl.DataFrame


@dataclass(frozen=True)
class LearnedModes:
    traces: list[pl.DataFrame]
    models: list[Model]


class HyDRALearner(SupervisedLearner):
    regressor_factory: Callable[[], SupervisedIncrementalLearner]
    threshold: float
    start_width: int
    step_width: int
    selector_learner: HybridDecisionTreeLearner | None
    callback: HyDRACallback | None

    def __init__(
        self,
        regressor_factory: Callable[[], SupervisedIncrementalLearner],
        threshold: float,
        start_width: int = 10,
        step_width: int = 5,
        selector_learner: HybridDecisionTreeLearner | None = None,
        callback: HyDRACallback | None = None,
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
        self.callback = callback

    def learn_new_flow(
        self,
        data_frame: pl.DataFrame,
        learner: SupervisedIncrementalLearner,
        inputs: list[str],
        target: list[str],
        *,
        trace_index: int,
        segment_start_index: int,
    ) -> Model | None:
        """Perform segmentation on the given data frame.

        Args:
            data_frame: The data frame to be segmented.
            learner: The learner to be used.
            inputs: The input feature names.
            target: The target feature names.
            trace_index: Trace index used for replay annotations.
            segment_start_index: Starting row index within the trace.

        Returns:
            The learned model for a new mode or None if
            segmentation fails.

        """
        window_size = self.start_width - self.step_width
        fit = None
        function = None
        best_fit = None
        best_function = None
        if len(data_frame) < self.start_width:
            return learner.learn_incremental(
                pl.LazyFrame(data_frame[inputs]),
                pl.LazyFrame(data_frame[target]),
            )
        while fit is None or (
            fit < self.threshold and window_size < len(data_frame)
        ):
            window_size += self.step_width
            segment = data_frame.slice(0, window_size)
            function = learner.learn_incremental(
                pl.LazyFrame(segment[inputs]),
                pl.LazyFrame(segment[target]),
            )
            predictions = function.predict(segment[inputs]).collect()
            fit = (segment[target[0]] - predictions[target[0]]).abs().max()
            if self.callback is not None:
                self.callback.candidate_window_evaluated(
                    trace_index,
                    segment_start_index,
                    segment_start_index + window_size - 1,
                    window_size=window_size,
                    fit=float(fit),
                )
            logger.info("Window size %d produced fit %s", window_size, fit)
            if fit < self.threshold:
                best_fit = fit
                best_function = function
            else:
                # Keep the current HyDRA heuristic: once a larger window misses
                # the threshold, stop instead of scanning even larger windows.
                break

        if best_function is None:
            logger.error(
                "Flow Identification failed: required accuracy not met "
                "with threshold %.2f and start_width %d.",
                self.threshold,
                self.start_width,
            )
            return None
        logger.info(
            "Selected window fit %s below threshold %.2f.",
            best_fit,
            self.threshold,
        )
        return best_function

    def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> HyDRAModel:
        """Learn a HyDRA model from the given inputs and outputs.

        Args:
            inputs: The input data as a LazyFrame.
            outputs: The output data as a LazyFrame.

        Returns:
            A HyDRAModel containing the learned modes.

        """
        traces = self._initialize_traces(inputs, outputs)
        input_columns = inputs.collect_schema().names()
        output_columns = outputs.collect_schema().names()
        if self.callback is not None:
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
        )

        if self.callback is not None:
            self.callback.finish(final_mode_count=len(learned_modes.models))

        return model

    def _discover_modes(
        self,
        traces: list[pl.DataFrame],
        input_columns: list[str],
        output_columns: list[str],
    ) -> LearnedModes:
        learned_models: list[Model] = []
        current_traces = traces
        pending_segment = find_next_pending_segment(current_traces)

        while pending_segment is not None:
            logger.info("Current pending segment: %s", pending_segment)
            if self.callback is not None:
                self.callback.pending_segment_found(
                    pending_segment.trace_index,
                    pending_segment.start_index,
                    pending_segment.end_index,
                )

            mode_learner = self.regressor_factory()
            mode_id = len(learned_models)

            candidate_model = self.learn_new_flow(
                current_traces[pending_segment.trace_index].slice(
                    pending_segment.start_index,
                    pending_segment.end_index
                    - pending_segment.start_index
                    + 1,
                ),
                mode_learner,
                input_columns,
                output_columns,
                trace_index=pending_segment.trace_index,
                segment_start_index=pending_segment.start_index,
            )

            if candidate_model is None:
                logger.error(
                    "Flow Identification failed. Stopping HyDRA learning.",
                )
                if self.callback is not None:
                    self.callback.learning_stopped(
                        trace_index=pending_segment.trace_index,
                        start_index=pending_segment.start_index,
                        end_index=pending_segment.end_index,
                        reason="flow_identification_failed",
                    )
                break

            previous_traces = current_traces
            segment_result = apply_model_to_traces(
                traces=current_traces,
                model=candidate_model,
                inputs=input_columns,
                target=output_columns,
                threshold=self.threshold,
                mode_id=mode_id,
            )
            if segment_result.accurate_rows.is_empty():
                logger.info(
                    "No accurate segments found for mode %d; stopping "
                    "HyDRA learning.",
                    mode_id,
                )
                if self.callback is not None:
                    self.callback.learning_stopped(
                        trace_index=pending_segment.trace_index,
                        start_index=pending_segment.start_index,
                        end_index=pending_segment.end_index,
                        reason="no_accurate_segments_found",
                    )
                break

            replay_segments: list[PendingSegment] = []
            if self.callback is not None:
                replay_segments = _find_new_mode_segments(
                    before=previous_traces,
                    after=segment_result.updated_traces,
                    mode_id=mode_id,
                )
                for segment in replay_segments:
                    self.callback.accurate_segment_found(
                        trace_index=segment.trace_index,
                        start_index=segment.start_index,
                        end_index=segment.end_index,
                        mode_id=mode_id,
                        threshold=self.threshold,
                    )

            finalized_model = mode_learner.learn_incremental(
                pl.LazyFrame(segment_result.accurate_rows[input_columns]),
                pl.LazyFrame(segment_result.accurate_rows[output_columns]),
            )
            learned_models.append(finalized_model)
            if self.callback is not None:
                finalized_segment = _select_replay_finalized_segment(
                    pending_segment=pending_segment,
                    accepted_segments=replay_segments,
                )
                self.callback.mode_finalized(
                    trace_index=finalized_segment.trace_index,
                    start_index=finalized_segment.start_index,
                    end_index=finalized_segment.end_index,
                    mode_id=mode_id,
                )
            logger.info(
                "Learned mode %d with model %s",
                len(learned_models) - 1,
                finalized_model,
            )
            current_traces = segment_result.updated_traces
            pending_segment = find_next_pending_segment(current_traces)

        return LearnedModes(traces=current_traces, models=learned_models)

    def _initialize_traces(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> list[pl.DataFrame]:
        output_columns = outputs.collect_schema().names()
        if len(output_columns) != 1:
            message = (
                "HyDRALearner currently supports single-output training only."
            )
            raise ValueError(message)

        trace = pl.concat([inputs, outputs], how="horizontal").collect()
        if trace.height == 0:
            message = (
                "HyDRALearner requires at least one row of training data."
            )
            raise ValueError(message)

        return [trace.with_columns(pl.lit(None).alias("mode"))]

    def _learn_selector(
        self,
        traces: list[pl.DataFrame],
        modes: list[Model],
    ) -> HybridDecisionTreeModel | None:
        if self.selector_learner is None:
            return None

        if any(trace["mode"].null_count() > 0 for trace in traces):
            message = "selector training requires fully labeled HyDRA traces"
            raise SelectorTrainingIncompleteError(message)

        return self.selector_learner.learn_from_traces(
            traces,
            mode_to_flow=dict(enumerate(modes)),
        )


def find_next_start(traces: list[pl.DataFrame]) -> tuple[int, int, int] | None:
    """Find the next start index among the traces.

    Args:
        traces: A list of DataFrames representing the traces.

    Returns:
        A tuple of (trace_index, start_index, end_index) or None if not found.
    """
    pending = find_next_pending_segment(traces)
    if pending is None:
        return None
    return (pending.trace_index, pending.start_index, pending.end_index)


def _find_new_mode_segments(
    before: list[pl.DataFrame],
    after: list[pl.DataFrame],
    mode_id: int,
) -> list[PendingSegment]:
    segments: list[PendingSegment] = []
    for trace_index, (before_trace, after_trace) in enumerate(
        zip(before, after, strict=True),
    ):
        start_index: int | None = None
        for row_index, (before_mode, after_mode) in enumerate(
            zip(
                before_trace["mode"].to_list(),
                after_trace["mode"].to_list(),
                strict=True,
            ),
        ):
            became_mode = before_mode is None and after_mode == mode_id
            if became_mode and start_index is None:
                start_index = row_index
            elif not became_mode and start_index is not None:
                segments.append(
                    PendingSegment(
                        trace_index=trace_index,
                        start_index=start_index,
                        end_index=row_index - 1,
                    ),
                )
                start_index = None
        if start_index is not None:
            segments.append(
                PendingSegment(
                    trace_index=trace_index,
                    start_index=start_index,
                    end_index=after_trace.height - 1,
                ),
            )
    return segments


def _select_replay_finalized_segment(
    *,
    pending_segment: PendingSegment,
    accepted_segments: list[PendingSegment],
) -> PendingSegment:
    for segment in accepted_segments:
        if segment.trace_index == pending_segment.trace_index:
            return segment
    if accepted_segments:
        return accepted_segments[0]
    return pending_segment


def find_next_pending_segment(
    traces: list[pl.DataFrame],
) -> PendingSegment | None:
    """Find the first pending contiguous unlabeled segment across traces."""
    for trace_i, trace in enumerate(traces):
        mode_column = trace["mode"]
        in_none_mode = False
        start_idx = 0
        for i, mode in enumerate(mode_column):
            if mode is None and not in_none_mode:
                in_none_mode = True
                start_idx = i
            elif mode is not None and in_none_mode:
                end_idx = i - 1
                return PendingSegment(
                    trace_index=trace_i,
                    start_index=start_idx,
                    end_index=end_idx,
                )
        if in_none_mode:
            end_idx = len(mode_column) - 1
            return PendingSegment(
                trace_index=trace_i,
                start_index=start_idx,
                end_index=end_idx,
            )
    return None


def get_accurate_segments(
    traces: list[pl.DataFrame],
    model: Model | None,
    inputs: list[str],
    target: list[str],
    threshold: float,
    mode_id: int,
) -> pl.DataFrame:
    """Get accurate segments from the traces using the given model.

    Args:
        traces: A list of DataFrames representing the traces.
        model: The model of the flow function to match against.
        inputs: The input feature names.
        target: The target feature names.
        threshold: The accuracy threshold.
        mode_id: The mode identifier to assign to accurate segments.

    Returns:
        A DataFrame containing all accurate segments across traces.

    """
    result = apply_model_to_traces(
        traces=traces,
        model=model,
        inputs=inputs,
        target=target,
        threshold=threshold,
        mode_id=mode_id,
    )
    traces[:] = result.updated_traces
    return result.accurate_rows


def apply_model_to_traces(
    traces: list[pl.DataFrame],
    model: Model | None,
    inputs: list[str],
    target: list[str],
    threshold: float,
    mode_id: int,
) -> AccurateSegmentResult:
    updated_traces = list(traces)
    all_accurate_data = []

    for trace_i in range(len(updated_traces)):
        if model is None:
            logger.error("No model provided for segmentation.")
            break
        trace = updated_traces[trace_i]
        predictions = model.predict(trace[inputs]).collect()
        errors = np.abs(predictions - trace[target])

        start = None
        for i, error in enumerate(errors):
            if trace["mode"][i] is None and np.all(error < threshold):
                if start is None:
                    start = i
            elif start is not None:
                end_idx = i - 1
                trace, accurate_rows = _label_segment(
                    trace=trace,
                    start_index=start,
                    end_index=end_idx,
                    mode_id=mode_id,
                )
                updated_traces[trace_i] = trace
                all_accurate_data.append(accurate_rows)
                logger.info(
                    "Found accurate segment in trace %d: %d to %d",
                    trace_i,
                    start,
                    end_idx,
                )
                start = None
        if start is not None:
            end_idx = len(errors) - 1
            trace, accurate_rows = _label_segment(
                trace=trace,
                start_index=start,
                end_index=end_idx,
                mode_id=mode_id,
            )
            updated_traces[trace_i] = trace
            all_accurate_data.append(accurate_rows)
            logger.info(
                "Found accurate segment in trace %d: %d to %d",
                trace_i,
                start,
                end_idx,
            )

    if not all_accurate_data:
        return AccurateSegmentResult(
            updated_traces=updated_traces,
            accurate_rows=(
                updated_traces[0].head(0) if updated_traces else pl.DataFrame()
            ),
        )
    return AccurateSegmentResult(
        updated_traces=updated_traces,
        accurate_rows=pl.concat(all_accurate_data, how="vertical"),
    )


def _label_segment(
    trace: pl.DataFrame,
    start_index: int,
    end_index: int,
    mode_id: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    trace_with_mode = trace.with_row_index("__idx")
    mask = (pl.col("__idx") >= start_index) & (pl.col("__idx") <= end_index)
    trace_with_mode = trace_with_mode.with_columns(
        pl.when(mask)
        .then(pl.lit(mode_id))
        .otherwise(pl.col("mode"))
        .alias("mode"),
    ).drop("__idx")
    return trace_with_mode, trace_with_mode[start_index : end_index + 1]
