import logging

import polars as pl
from river.base import Regressor
from typing_extensions import override

from flowcean.core import (
    LearnerCallback,
    Model,
    SupervisedIncrementalLearner,
    create_callback_manager,
)

logger = logging.getLogger(__name__)


class RiverModel(Model):
    def __init__(self, model: Regressor, output_column: str) -> None:
        self.model = model
        self.output_column = output_column

    @override
    def _predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        df = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        predictions = [self.model.predict_one(row) for row in df.to_dicts()]
        return pl.LazyFrame({self.output_column: predictions})


class RiverLearner(SupervisedIncrementalLearner):
    """Wrapper for River regressors.

    Args:
        model: The River regressor to use.
        callbacks: Optional callbacks for progress feedback. Defaults to
            RichCallback if not specified.
        progress_interval: Report progress every N samples. Default is 100.
    """

    def __init__(
        self,
        model: Regressor,
        callbacks: list[LearnerCallback] | LearnerCallback | None = None,
        progress_interval: int = 100,
    ) -> None:
        self.model = model
        self.callback_manager = create_callback_manager(callbacks)
        self.progress_interval = progress_interval

    def learn_incremental(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> Model:
        # Collect the inputs and outputs into DataFrames
        inputs_df = inputs.collect()
        outputs_df = outputs.collect()

        total_samples = len(inputs_df)

        # Notify callbacks that learning is starting
        context = {
            "n_samples": total_samples,
            "n_features": len(inputs_df.columns),
            "algorithm": type(self.model).__name__,
        }
        self.callback_manager.on_learning_start(self, context)

        try:
            # Iterate over the rows of the inputs and outputs incrementally
            for idx, (input_row, output_row) in enumerate(
                zip(
                    inputs_df.iter_rows(named=True),
                    outputs_df.iter_rows(named=True),
                    strict=False,
                ),
                start=1,
            ):
                xi = dict(input_row)  # Convert input row to a dictionary
                yi = next(
                    iter(output_row.values()),
                )  # Extract the first (and only) output value
                self.model.learn_one(xi, yi)  # Incrementally train the model

                # Report progress periodically
                if idx % self.progress_interval == 0 or idx == total_samples:
                    progress = (
                        idx / total_samples if total_samples > 0 else None
                    )
                    self.callback_manager.on_learning_progress(
                        self,
                        progress=progress,
                        metrics={
                            "samples_processed": idx,
                            "total_samples": total_samples,
                        },
                    )

            # Return the trained RiverModel
            y_col = pl.LazyFrame.collect_schema(outputs).names()[0]
            model = RiverModel(self.model, y_col)

            # Notify callbacks that learning is complete
            self.callback_manager.on_learning_end(
                self,
                model,
                metrics={"total_samples": total_samples},
            )
        except Exception as e:
            # Notify callbacks of the error
            self.callback_manager.on_learning_error(self, e)
            raise
        else:
            return model
