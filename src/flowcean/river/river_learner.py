import logging
from typing import Any

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
    def predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        df = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        predictions = [self.model.predict_one(row) for row in df.to_dicts()]
        return pl.LazyFrame({self.output_column: predictions})

    @override
    def save_state(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "output_column": self.output_column,
        }

    @classmethod
    @override
    def load_from_state(cls, state: dict) -> "RiverModel":
        data = state
        return cls(model=data["model"], output_column=data["output_column"])


class RiverLearner(SupervisedIncrementalLearner):
    """Wrapper for River regressors."""

    def __init__(self, model: Regressor) -> None:
        self.model = model

    def learn_incremental(
        self, inputs: pl.LazyFrame, outputs: pl.LazyFrame
    ) -> Model:
        logger.info("Training river model incrementally")

        # Iterate over the rows of the inputs and outputs incrementally
        for input_row, output_row in zip(
            pl.LazyFrame.collect(inputs),
            pl.LazyFrame.collect(outputs),
            strict=False,
        ):
            xi = dict(input_row)  # Convert input row to a dictionary
            yi = next(
                iter(output_row.to_list()),
            )  # Extract the first (and only) output value
            self.model.learn_one(xi, yi)  # Incrementally train the model

        # Return the trained RiverModel
        y_col = outputs.columns[0]
        return RiverModel(self.model, y_col)


logger = logging.getLogger(__name__)
