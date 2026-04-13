import polars as pl
from sklearn.linear_model import LinearRegression as SKLearnLinearRegression

from flowcean.core.learner import SupervisedLearner
from flowcean.sklearn.model import SciKitModel


class LinearRegression(SupervisedLearner):
    def learn(
        self,
        inputs: pl.LazyFrame | pl.DataFrame,
        outputs: pl.LazyFrame | pl.DataFrame,
    ) -> SciKitModel:
        if isinstance(inputs, pl.LazyFrame):
            inputs = inputs.collect()
        if isinstance(outputs, pl.LazyFrame):
            outputs = outputs.collect()

        model = SKLearnLinearRegression()
        model.fit(
            inputs,
            outputs,
        )

        return SciKitModel(model, output_names=outputs.columns)
