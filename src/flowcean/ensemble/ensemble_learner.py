from functools import reduce

import polars as pl

from flowcean.core.data import Data
from flowcean.core.learner import SupervisedLearner
from flowcean.core.model import Model


class EnsembleModel(Model):
    """Ensemble model that combines multiple models."""

    def __init__(self, *models: Model) -> None:
        """Initialize the ensemble model.

        Args:
            models: The tuple of models to combine.
        """
        self.models = models

    def _predict(self, input_features: Data) -> Data:
        predictions: list[Data] = []

        for model in self.models:
            prediction = model.predict(input_features)
            predictions.append(prediction)

        if isinstance(predictions[0], pl.LazyFrame):
            column_names = predictions[0].collect_schema().names()
            total_prediction = reduce(
                lambda x, y: pl.concat(
                    [x, y.select(pl.all().name.suffix("_"))],
                    how="horizontal",
                ).select(
                    [
                        (pl.col(col) + pl.col(f"{col}_")).alias(col)
                        for col in column_names
                    ],
                ),
                [pred.lazy() for pred in predictions],
            )
        else:
            total_prediction = reduce(lambda x, y: x + y, predictions)
        return total_prediction


class EnsembleLearner(SupervisedLearner):
    """Ensemble learner that combines multiple supervised learners.

    The ensemble learner trains each of its constituent learners sequentially.
    After each learner is trained, its residual errors are calculated and used
    as the target outputs for the next learner in the sequence.

    The final model produced by the ensemble learner combines the predictions
    of all constituent models by summing their outputs.

    This learner works with any supervised learner and arbitrary data types
    supported by those learners, as long as the `Data` type is consistent
    across all learners and implements `__add__` and `__sub__` operations.
    Special handling is included for Polars DataFrames and LazyFrames to be
    compatible with the default flowcean learners.
    """

    def __init__(self, *learners: SupervisedLearner) -> None:
        """Initialize the ensemble learner.

        Args:
            learners: A tuple of supervised learner instances to combine.
        """
        self.learners = learners

    def learn(self, inputs: Data, outputs: Data) -> EnsembleModel:
        models: list[Model] = []

        output_names: list[str] = []
        if isinstance(outputs, pl.LazyFrame):
            output_names = outputs.collect_schema().names()

        for learner in self.learners:
            model = learner.learn(inputs, outputs)
            models.append(model)

            predicted_outputs = model.predict(inputs)

            if isinstance(outputs, pl.LazyFrame) and isinstance(
                predicted_outputs,
                pl.LazyFrame | pl.DataFrame,
            ):
                predicted_outputs = predicted_outputs.lazy().select(
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
            else:
                outputs = outputs - predicted_outputs

        return EnsembleModel(*tuple(models))
