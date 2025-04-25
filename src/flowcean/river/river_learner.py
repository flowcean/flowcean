import logging
import pickle

import polars as pl
from river.base import Regressor
from typing_extensions import override

from flowcean.core import Model, SupervisedLearner

logger = logging.getLogger(__name__)


class RiverModel(Model):
    def __init__(self, model: Regressor, output_column: str) -> None:
        self.model = model
        self.output_column = output_column

    @override
    def predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        df = input_features.collect() if isinstance(input_features, pl.LazyFrame) else inputs
        predictions = [self.model.predict_one(row) for row in df.to_dicts()]
        return pl.LazyFrame({self.output_column: predictions})

    @override
    def save_state(self) -> bytes:
        return pickle.dumps(
            {
                "model": self.model,
                "output_column": self.output_column,
            },
        )

    @classmethod
    @override
    def load_from_state(cls, state: bytes) -> "RiverModel":
        data = pickle.loads(state)
        return cls(model=data["model"], output_column=data["output_column"])


class RiverLearner(SupervisedLearner):
    """Wrapper for River regressors."""

    def __init__(self, model: Regressor) -> None:
        self.model = model

    @override
    def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> Model:
        x = inputs.collect().to_dicts()
        y_col = outputs.columns[0]
        y = outputs.collect()[y_col].to_list()

        logger.info("Training river model")

        for xi, yi in zip(x, y, strict=False):
            self.model.learn_one(xi, yi)

        return RiverModel(self.model, y_col)
    
    def learn_incremental(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> Model:
        logger.info("Training river model incrementally")

        # Iterate over the rows of the inputs and outputs incrementally
        for input_row, output_row in zip(inputs.iter_rows(named=True), outputs.iter_rows(named=True), strict=False):
            xi = dict(input_row)  # Convert input row to a dictionary
            yi = list(output_row.values())[0]  # Extract the first (and only) output value
            self.model.learn_one(xi, yi)  # Incrementally train the model

        # Return the trained RiverModel
        y_col = outputs.columns[0]
        return RiverModel(self.model, y_col)


logger = logging.getLogger(__name__)
