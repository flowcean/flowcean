from typing import Any

import polars as pl
from xgboost import XGBClassifier, XGBRegressor

from flowcean.core.learner import SupervisedLearner

from .model import XGBoostClassifierModel, XGBoostRegressorModel


class XGBoostClassifierLearner(SupervisedLearner):
    """Wrapper for XGBoost classifiers."""

    def __init__(self, **kwargs: Any) -> None:
        self.classifier = XGBClassifier(**kwargs)
        super().__init__()

    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> XGBoostClassifierModel:
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
        return XGBoostClassifierModel(
            self.classifier,
            input_features=inputs_collected.columns,
            output_features=outputs_collected.columns,
        )


class XGBoostRegressorLearner(SupervisedLearner):
    """Wrapper for XGBoost regressor."""

    def __init__(self, **kwargs: Any) -> None:
        self.regressor = XGBRegressor(**kwargs)
        super().__init__()

    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> XGBoostRegressorModel:
        dfs = pl.collect_all([inputs, outputs])
        inputs_collected = dfs[0]
        outputs_collected = dfs[1]

        # Fit the classifier
        self.regressor.fit(
            inputs_collected.to_numpy(),
            outputs_collected.to_numpy(),
        )

        return XGBoostRegressorModel(
            self.regressor,
            input_features=inputs_collected.columns,
            output_features=outputs_collected.columns,
        )
