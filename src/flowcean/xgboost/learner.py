import polars as pl
from xgboost import XGBClassifier

from flowcean.core.learner import SupervisedLearner

from .model import XGBoostModel


class XGBoostLearner(SupervisedLearner):
    """Wrapper for XGBoost classifiers."""

    def __init__(self) -> None:
        self.classifier = XGBClassifier(
            n_estimators=2,
            max_depth=2,
            learning_rate=1,
            objective="binary:logistic",
        )
        super().__init__()

    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> XGBoostModel:
        # Collect the inputs and outputs. Using collect_all ensures that polars
        # can apply optimizations to both dataframes simultaneously.
        dfs = pl.collect_all([inputs, outputs])
        inputs_collected = dfs[0]
        outputs_collected = dfs[1]

        # Fit the classifier
        self.classifier.fit(
            inputs_collected.to_numpy(),
            outputs_collected.to_numpy(),
        )
        return XGBoostModel(
            self.classifier,
            input_features=inputs_collected.columns,
            output_features=outputs_collected.columns,
        )
