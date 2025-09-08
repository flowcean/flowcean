import logging

import polars as pl
from river.base import Regressor
from typing_extensions import override

from flowcean.core import Model, SupervisedIncrementalLearner

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
    """Wrapper for River regressors."""

    def __init__(self, model: Regressor) -> None:
        self.model = model

    def learn_incremental(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> Model:
        # Collect the inputs and outputs into DataFrames
        inputs_df = inputs.collect()
        outputs_df = outputs.collect()

        # Iterate over the rows of the inputs and outputs incrementally
        for input_row, output_row in zip(
            inputs_df.iter_rows(named=True),
            outputs_df.iter_rows(named=True),
            strict=False,
        ):
            xi = dict(input_row)  # Convert input row to a dictionary
            yi = next(
                iter(output_row.values()),
            )  # Extract the first (and only) output value
            self.model.learn_one(xi, yi)  # Incrementally train the model

        # Return the trained RiverModel
        y_col = pl.LazyFrame.collect_schema(outputs).names()[0]
        return RiverModel(self.model, y_col)


logger = logging.getLogger(__name__)
