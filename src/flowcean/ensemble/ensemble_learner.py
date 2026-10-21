from functools import reduce

import polars as pl

from flowcean.core.learner import SupervisedLearner
from flowcean.core.model import Model, Predictor


class EnsembleModel(Model[pl.LazyFrame, pl.LazyFrame]):
    """Combine tabular model predictions by summing matching output columns."""

    def __init__(self, *models: Predictor[pl.LazyFrame, pl.LazyFrame]) -> None:
        if not models:
            message = "An ensemble requires at least one model"
            raise ValueError(message)
        self.models = models

    def _predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        predictions = [model.predict(input_features) for model in self.models]
        column_names = predictions[0].collect_schema().names()
        return reduce(
            lambda x, y: pl.concat(
                [x, y.select(pl.all().name.suffix("_"))],
                how="horizontal",
            ).select(
                [
                    (pl.col(col) + pl.col(f"{col}_")).alias(col)
                    for col in column_names
                ],
            ),
            predictions,
        )


class EnsembleLearner(
    SupervisedLearner[pl.LazyFrame, pl.LazyFrame, EnsembleModel],
):
    """Fit tabular learners sequentially to preceding learners' residuals."""

    def __init__(
        self,
        *learners: SupervisedLearner[
            pl.LazyFrame, pl.LazyFrame, Predictor[pl.LazyFrame, pl.LazyFrame]
        ],
    ) -> None:
        if not learners:
            message = "An ensemble requires at least one learner"
            raise ValueError(message)
        self.learners = learners

    def learn(
        self, inputs: pl.LazyFrame, outputs: pl.LazyFrame
    ) -> EnsembleModel:
        models: list[Predictor[pl.LazyFrame, pl.LazyFrame]] = []
        output_names = outputs.collect_schema().names()
        for learner in self.learners:
            model = learner.learn(inputs, outputs)
            models.append(model)
            predicted_outputs = model.predict(inputs).select(
                pl.all().name.suffix("_predicted"),
            )
            outputs = pl.concat(
                [outputs, predicted_outputs],
                how="horizontal",
            ).select(
                [
                    (pl.col(col) - pl.col(f"{col}_predicted")).alias(col)
                    for col in output_names
                ],
            )
        return EnsembleModel(*models)
