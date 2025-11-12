import logging
from collections.abc import Callable

import numpy as np
import polars as pl

from flowcean.core.learner import (
    SupervisedIncrementalLearner,
    SupervisedLearner,
)
from flowcean.core.model import Model
from flowcean.hydra.model import HyDRAModel

logger = logging.getLogger(__name__)


class HyDRALearner(SupervisedLearner):
    regressor_factory: Callable[[], SupervisedIncrementalLearner]
    threshold: float
    start_width: int
    step_width: int

    def __init__(
        self,
        regressor_factory: Callable[[], SupervisedIncrementalLearner],
        threshold: float,
        start_width: int = 10,
        step_width: int = 5,
    ) -> None:
        super().__init__()
        self.regressor_factory = regressor_factory
        self.threshold = threshold
        self.start_width = start_width
        self.step_width = step_width

    def learn_new_flow(
        self,
        data_frame: pl.DataFrame,
        learner: SupervisedIncrementalLearner,
        inputs: list[str],
        target: list[str],
    ) -> Model | None:
        """Perform segmentation on the given data frame.

        Args:
            data_frame: The data frame to be segmented.
            learner: The learner to be used.
            inputs: The input feature names.
            target: The target feature names.

        Returns:
            The learned model for a new mode or None if
            segmentation fails.

        """
        window_size = self.start_width - self.step_width
        fit = None
        function = None
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
            print(f"Window size: {window_size}, Fit: {fit}")

        print(f"Selected window size: {window_size} with fit: {fit}")
        if window_size == self.start_width and fit >= self.threshold:
            logger.error(
                "Flow Identification failed: required accuracy not met "
                "with threshold %.2f and start_width %d.",
                self.threshold,
                self.start_width,
            )
            return None
        return function

    def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> HyDRAModel:
        """Learn a HyDRA model from the given inputs and outputs.

        Args:
            inputs: The input data as a LazyFrame.
            outputs: The output data as a LazyFrame.

        Returns:
            A HyDRAModel containing the learned modes.

        """
        input_columns = inputs.collect_schema().names()
        output_columns = outputs.collect_schema().names()

        traces = [pl.concat([inputs, outputs], how="horizontal").collect()]
        traces = [
            trace.with_columns(pl.lit(None).alias("mode")) for trace in traces
        ]
        modes = []
        num_modes = 0
        next_start = (0, 0, len(traces[0]) - 1)

        while next_start is not None:
            logger.info("Current start index: %s", next_start)

            regressor_instance = self.regressor_factory()

            model = self.learn_new_flow(
                traces[next_start[0]].slice(
                    next_start[1],
                    next_start[2] - next_start[1] + 1,
                ),
                regressor_instance,
                input_columns,
                output_columns,
            )

            if model is None:
                logger.error(
                    "Flow Identification failed. Stopping HyDRA learning.",
                )
                break

            mode_data = get_accurate_segments(
                traces,
                model,
                input_columns,
                output_columns,
                self.threshold,
                num_modes,
            )

            model = regressor_instance.learn_incremental(
                pl.LazyFrame(mode_data[input_columns]),
                pl.LazyFrame(mode_data[output_columns]),
            )
            modes.append(model)
            logger.info("Learned mode %d with model %s", num_modes, model)
            num_modes += 1

            if len(traces) == 0:
                break

            next_start = find_next_start(traces)

        return HyDRAModel(
            modes,
            input_features=input_columns,
            output_features=output_columns,
        )


def find_next_start(traces: list[pl.DataFrame]) -> tuple[int, int, int] | None:
    """Find the next start index among the traces.

    Args:
        traces: A list of DataFrames representing the traces.

    Returns:
        A tuple of (trace_index, start_index, end_index) or None if not found.
    """
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
                return (trace_i, start_idx, end_idx)
        if in_none_mode:
            end_idx = len(mode_column) - 1
            return (trace_i, start_idx, end_idx)
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
    all_accurate_data = []

    for trace_i in range(len(traces)):
        if model is None:
            logger.error("No model provided for segmentation.")
            break
        predictions = model.predict(traces[trace_i][inputs]).collect()
        errors = np.abs(predictions - traces[trace_i][target])

        start = None
        for i, error in enumerate(errors):
            if traces[trace_i]["mode"][i] is None and np.all(
                error < threshold,
            ):
                if start is None:
                    # Start of a new interval
                    start = i
            elif start is not None:
                end_idx = i - 1
                trace_with_mode = traces[trace_i].with_row_index("__idx")
                mask = (pl.col("__idx") >= start) & (
                    pl.col("__idx") <= end_idx
                )
                trace_with_mode = trace_with_mode.with_columns(
                    pl.when(mask)
                    .then(pl.lit(mode_id))
                    .otherwise(pl.col("mode"))
                    .alias("mode"),
                ).drop("__idx")
                traces[trace_i] = trace_with_mode
                all_accurate_data.append(
                    traces[trace_i][start:i],
                )
                logger.info(
                    "Found accurate segment in trace %d: %d to %d",
                    trace_i,
                    start,
                    end_idx,
                )
                start = None
        if start is not None:
            # Collect data for the last segment
            end_idx = len(errors) - 1
            trace_with_mode = traces[trace_i].with_row_index("__idx")
            mask = (pl.col("__idx") >= start) & (pl.col("__idx") <= end_idx)
            trace_with_mode = trace_with_mode.with_columns(
                pl.when(mask)
                .then(pl.lit(mode_id))
                .otherwise(pl.col("mode"))
                .alias("mode"),
            ).drop("__idx")
            traces[trace_i] = trace_with_mode
            all_accurate_data.append(
                traces[trace_i][start : len(errors)],
            )
            logger.info(
                "Found accurate segment in trace %d: %d to %d",
                trace_i,
                start,
                end_idx,
            )

    return pl.concat(all_accurate_data, how="vertical")
