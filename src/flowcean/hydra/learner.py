import logging
from copy import deepcopy

import numpy as np
import polars as pl

from flowcean.core.learner import (
    SupervisedIncrementalLearner,
    SupervisedLearner,
)
from flowcean.core.model import Model
from flowcean.hydra.model import HyDRAModel
from flowcean.hydra.segmentor import Segmentor

logger = logging.getLogger(__name__)


class HyDRALearner(SupervisedLearner):
    regressor: SupervisedIncrementalLearner
    threshold: float

    def __init__(
        self,
        regressor: SupervisedIncrementalLearner,
        threshold: float,
    ) -> None:
        # Additional initialization for HyDRA if needed
        super().__init__()
        self.regressor = regressor
        self.threshold = threshold

    def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> HyDRAModel:
        # Implement the learning process specific to HyDRA

        input_columns = inputs.columns
        output_columns = outputs.columns

        # Combine inputs and outputs into a single DataFrame
        traces = [pl.concat([inputs, outputs], how="horizontal").collect()]
        trajectory = traces[0]
        modes = []

        while len(trajectory) > 0:
            logger.info("Remaining traces: %f", len(traces))
            logger.info("Current trajectory length: %f", len(trajectory))

            # Segmentation
            segmentor = Segmentor(
                output_columns,
                input_columns,
                threshold=self.threshold,
                start_width=10,
                step_width=5,
            )

            regressor_instance = deepcopy(self.regressor)

            model = segmentor.segment(trajectory, regressor_instance)

            accurate_segments, mode_data = get_accurate_segments(
                traces,
                model,
                input_columns,
                output_columns,
                self.threshold,
            )

            model = regressor_instance.learn_incremental(
                pl.LazyFrame(mode_data[input_columns]),
                pl.LazyFrame(mode_data[output_columns]),
            )
            modes.append(model)

            # Build remaining traces
            traces = build_remaining_traces(
                traces,
                accurate_segments,
            )
            if len(traces) == 0:
                break

            trajectory = traces[0]

        for i, mode in enumerate(modes):
            print(f"Mode {i}:")
            print(mode)
        return HyDRAModel(
            modes,
            input_features=input_columns,
            output_features=output_columns,
        )


def build_remaining_traces(
    traces: list[pl.DataFrame],
    accurate_segments: list[list[tuple[int, int]]],
) -> list[pl.DataFrame]:
    """Build the remaining traces from the accurate segments.

    Args:
        traces (list[DataFrame]): The list of traces.
        accurate_segments (list[DataFrame]): The list of accurate segments.

    Returns:
        list[DataFrame]: The list of remaining traces.
    """
    remaining_traces = []
    for trace, segments in zip(traces, accurate_segments, strict=True):
        remaining_trace = trace
        removed_size = 0
        for start, end in segments:
            updated_start = start - removed_size
            removed_size += end - removed_size + 1
            # Add the part before start as a new trace
            new_segment = remaining_trace.slice(0, updated_start)
            remaining_traces.append(new_segment)

            # Update the remaining trace to the part after end
            remaining_trace = remaining_trace.slice(end + 1)
        remaining_traces.append(remaining_trace)
    # Remove empty traces
    return [trace for trace in remaining_traces if len(trace) > 0]


def get_accurate_segments(
    traces: list[pl.DataFrame],
    model: Model | None,
    inputs: list[str],
    target: list[str],
    threshold: float,
) -> tuple[list[list[tuple[int, int]]], pl.DataFrame]:
    """Get accurate segments from the traces using the given model."""
    accurate_segments = []
    all_accurate_data = []

    for trace in traces:
        if model is None:
            continue  # or handle the None case as needed
        predictions = model.predict(trace[inputs]).collect()
        errors = np.abs(predictions - trace[target])
        local_segments = []

        start = None
        for i, error in enumerate(errors):
            if np.all(error < threshold):
                if start is None:
                    start = i  # Start of a new interval
            elif start is not None:
                local_segments.append(
                    (start, i - 1),
                )  # End of the interval
                all_accurate_data.append(
                    trace[start:i],
                )  # Collect data for the segment
                start = None
        if start is not None:
            local_segments.append(
                (start, len(errors) - 1),
            )  # Handle last interval
            all_accurate_data.append(
                trace[start : len(errors)],
            )  # Collect data for the last segment
        accurate_segments.append(local_segments)
        print("Accurate segments:", len(accurate_segments))

    combined_accurate_data = pl.concat(all_accurate_data, how="vertical")

    return accurate_segments, combined_accurate_data
