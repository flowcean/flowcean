import logging
import pickle

import polars as pl
from river.base import Regressor
from typing_extensions import override

from flowcean.core import Model, SupervisedLearner

# from flowcean.polars import to_numpy

logger = logging.getLogger(__name__)


class RiverModel(Model):
    def __init__(self, model: Regressor, output_column: str) -> None:
        self.model = model
        self.output_column = output_column

    @override
    def predict(self, inputs: pl.DataFrame) -> pl.DataFrame:
        df = inputs.collect() if isinstance(inputs, pl.LazyFrame) else inputs
        predictions = [self.model.predict_one(row) for row in df.to_dicts()]
        return pl.DataFrame({self.output_column: predictions})

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


logger = logging.getLogger(__name__)
