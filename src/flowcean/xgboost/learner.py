from typing import Any

import polars as pl
from xgboost import XGBClassifier, XGBRegressor
from xgboost.callback import TrainingCallback

from flowcean.core import LearnerCallback, create_callback_manager
from flowcean.core.learner import SupervisedLearner
from flowcean.core.named import Named

from .model import XGBoostClassifierModel, XGBoostRegressorModel


class XGBoostCallbackBridge(TrainingCallback):
    """Bridge between XGBoost callbacks and flowcean callbacks.

    This adapter forwards XGBoost training events to flowcean callbacks.
    """

    def __init__(
        self,
        learner: Named,
        callback_manager: Any,
        max_iterations: int | None = None,
    ) -> None:
        self.learner = learner
        self.callback_manager = callback_manager
        self.max_iterations = max_iterations

    def before_training(self, model: Any) -> Any:
        """Called before training starts."""
        # XGBoost callbacks don't get called at the very beginning,
        # so we'll call on_learning_start from the learn method instead
        return model

    def after_iteration(
        self,
        _model: Any,
        epoch: int,
        evals_log: dict[str, Any],
    ) -> bool:
        """Called after each training iteration."""
        # Calculate progress if we know the max iterations
        progress = None
        if self.max_iterations and self.max_iterations > 0:
            progress = (epoch + 1) / self.max_iterations

        # Extract metrics from evals_log if available
        metrics = {"iteration": epoch + 1}
        if evals_log:
            # Flatten the nested evals_log structure
            for dataset_name, dataset_metrics in evals_log.items():
                for metric_name, values in dataset_metrics.items():
                    if values:
                        metrics[f"{dataset_name}_{metric_name}"] = values[-1]

        self.callback_manager.on_learning_progress(
            self.learner,
            progress=progress,
            metrics=metrics,
        )

        # Return False to continue training
        return False

    def after_training(self, model: Any) -> Any:
        """Called after training completes."""
        # We'll call on_learning_end from the learn method instead
        # to have access to the wrapped model
        return model


class XGBoostClassifierLearner(SupervisedLearner):
    """Wrapper for XGBoost classifiers.

    Args:
        callbacks: Optional callbacks for progress feedback. Defaults to
            RichCallback if not specified.
        **kwargs: Arguments passed to XGBClassifier
            (n_estimators, max_depth, etc.)
    """

    def __init__(
        self,
        callbacks: list[LearnerCallback] | LearnerCallback | None = None,
        **kwargs: Any,
    ) -> None:
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

        # Notify callbacks that learning is starting
        context = {
            "n_estimators": self.classifier.n_estimators,
            "n_samples": len(inputs_collected),
            "n_features": len(inputs_collected.columns),
        }
        self.callback_manager.on_learning_start(self, context)

        try:
            # Create bridge callback for XGBoost
            xgb_callback = XGBoostCallbackBridge(
                learner=self,
                callback_manager=self.callback_manager,
                max_iterations=self.classifier.n_estimators,
            )

            # Fit the classifier with callbacks
            self.classifier.fit(
                inputs_collected.to_numpy(),
                outputs_collected.to_numpy(),
                callbacks=[xgb_callback],
            )

            # Create the model
            model = XGBoostClassifierModel(
                self.classifier,
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


class XGBoostRegressorLearner(SupervisedLearner):
    """Wrapper for XGBoost regressor.

    Args:
        callbacks: Optional callbacks for progress feedback. Defaults to
            RichCallback if not specified.
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

            # Fit the regressor with callbacks
            self.regressor.fit(
                inputs_collected.to_numpy(),
                outputs_collected.to_numpy(),
                callbacks=[xgb_callback],
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
