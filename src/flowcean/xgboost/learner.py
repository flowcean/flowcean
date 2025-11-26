from typing import Any

import polars as pl
from xgboost import XGBClassifier, XGBRegressor

from flowcean.core import LearnerCallback, create_callback_manager
from flowcean.core.learner import SupervisedLearner

from .model import XGBoostClassifierModel, XGBoostRegressorModel


class XGBoostClassifierLearner(SupervisedLearner):
    """Wrapper for XGBoost classifiers.

    Args:
        threshold: Decision threshold for binary classification (default: 0.5).
        **kwargs: Additional arguments passed to XGBClassifier.
    """

    def __init__(self, threshold: float = 0.5, **kwargs: Any) -> None:
        # Store threshold separately - don't pass to XGBoost
        self.threshold = threshold
        self.classifier = XGBClassifier(**kwargs)
        self.callback_manager = create_callback_manager(callbacks)
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
            threshold=self.threshold,
        )


class XGBoostRegressorLearner(SupervisedLearner):
    """Wrapper for XGBoost regressor.

    Args:
        callbacks: Optional callbacks for progress feedback. Use `None` for
            silent learning.
        **kwargs: Arguments passed to XGBRegressor
            (n_estimators, max_depth, etc.)
    """

    def __init__(
        self,
        callbacks: list[LearnerCallback] | LearnerCallback | None = None,
        **kwargs: Any,
    ) -> None:
        self.regressor = XGBRegressor(**kwargs)
        self.callback_manager = create_callback_manager(callbacks)
        super().__init__()

    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> XGBoostRegressorModel:
        dfs = pl.collect_all([inputs, outputs])
        inputs_collected = dfs[0]
        outputs_collected = dfs[1]

        # Notify callbacks that learning is starting
        context = {
            "n_estimators": self.regressor.n_estimators,
            "n_samples": len(inputs_collected),
            "n_features": len(inputs_collected.columns),
        }
        self.callback_manager.on_learning_start(self, context)

        try:
            # Create bridge callback for XGBoost
            xgb_callback = XGBoostCallbackBridge(
                learner=self,
                callback_manager=self.callback_manager,
                max_iterations=self.regressor.n_estimators,
            )

            # In XGBoost 3.0+, callbacks must be set on the model, not in fit()
            # Get current params and recreate regressor with callbacks
            params = self.regressor.get_params()
            params["callbacks"] = [xgb_callback]
            self.regressor = XGBRegressor(**params)

            # Fit the regressor
            self.regressor.fit(
                inputs_collected.to_numpy(),
                outputs_collected.to_numpy(),
            )

            # Create the model
            model = XGBoostRegressorModel(
                self.regressor,
                input_features=inputs_collected.columns,
                output_features=outputs_collected.columns,
            )

            # Notify callbacks that learning is complete
            self.callback_manager.on_learning_end(self, model)
        except Exception as e:
            # Notify callbacks of the error
            self.callback_manager.on_learning_error(self, e)
            raise
        else:
            return model
