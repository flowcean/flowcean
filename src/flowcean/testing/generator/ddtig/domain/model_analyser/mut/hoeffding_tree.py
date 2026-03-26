import logging
from typing import Any

import polars as pl
from river import metrics
from river.tree import HoeffdingTreeClassifier, HoeffdingTreeRegressor

from flowcean.testing.generator.ddtig.application import ModelHandler
from flowcean.testing.generator.ddtig.domain import DataModel
from flowcean.testing.generator.ddtig.user_interface import SystemSpecsHandler

logger = logging.getLogger(__name__)


class HoeffdingTree:
    """Train a Hoeffding Tree on synthetic samples.

    Samples are generated from another model.

    Attributes:
    ----------
    datamodel : DataModel
        Object used to generate synthetic training inputs based
        on the original dataset.

    samples : list
        Original training inputs transformed to River-compatible
        format with predictions.

    nominal_attributes : list
        List of indices for nominal features.

    Methods:
    -------
    train_tree()
        Trains a Hoeffding Tree and returns the trained model.
    """

    N_SAMPLES = (
        5000  # Default number of samples to generate when more data is needed
    )

    def __init__(
        self,
        inputs: pl.DataFrame,
        seed: int,
        model_handler: ModelHandler,
        specs_handler: SystemSpecsHandler,
    ) -> None:
        """Initializes the HoeffdingTree trainer.

        Args:
            inputs : Original training dataset including target column.
            seed : Random seed for reproducible synthetic sample generation.
            model_handler : Object used to generate predictions from
                the Flowcean model.
            specs_handler : Object containing feature specifications
                and metadata.
        """
        # Remove target column to isolate input features
        inputs = inputs.drop(inputs.columns[-1])
        self.datamodel = DataModel(inputs, seed, model_handler, specs_handler)

        # Generate River-compatible samples using original data
        self.samples = self.datamodel.generate_dataset(original_data=True)
        self.nominal_attributes = specs_handler.get_nominal_features()

    def _create_model_and_metric(
        self,
        *,
        classification: bool,
        **kwargs: Any,
    ) -> tuple[
        metrics.F1 | metrics.MAE,
        HoeffdingTreeClassifier | HoeffdingTreeRegressor,
    ]:
        if classification:
            metric = metrics.F1()
            model = HoeffdingTreeClassifier(
                nominal_attributes=self.nominal_attributes,
                binary_split=True,
                grace_period=10,
                leaf_prediction="mc",
                **kwargs,
            )
            return metric, model

        metric = metrics.MAE()
        model = HoeffdingTreeRegressor(
            nominal_attributes=self.nominal_attributes,
            binary_split=True,
            grace_period=10,
            leaf_prediction="adaptive",
            **kwargs,
        )
        return metric, model

    @staticmethod
    def _normalize_target(target: Any, *, classification: bool) -> Any:
        if classification and isinstance(target, float):
            return bool(target)
        return target

    @staticmethod
    def _normalize_prediction(
        prediction: Any,
        *,
        classification: bool,
    ) -> bool | float | dict[bool, float]:
        if isinstance(prediction, dict):
            return {
                bool(key): float(value)
                for key, value in prediction.items()
            }
        if isinstance(prediction, (bool, int, float)):
            return bool(prediction) if classification else float(prediction)
        return False if classification else 0.0

    def _run_training_loop(
        self,
        *,
        model: HoeffdingTreeClassifier | HoeffdingTreeRegressor,
        metric: metrics.F1 | metrics.MAE,
        performance_threshold: float,
        n_predictions: int,
        sample_limit: int,
        classification: bool,
    ) -> None:
        correct_predictions = 0
        for i, (x, y) in enumerate(self.samples, start=1):
            y_true = self._normalize_target(y, classification=classification)
            y_pred = self._normalize_prediction(
                model.predict_one(x),
                classification=classification,
            )
            metric.update(y_true, y_pred)

            if metric.get() <= performance_threshold:
                correct_predictions += 1
                if correct_predictions == n_predictions:
                    break
            else:
                model.learn_one(x, y_true)
                correct_predictions = 0

            if i == len(self.samples):
                if len(self.samples) >= sample_limit:
                    break
                samples_to_generate = (
                    sample_limit - len(self.samples)
                    if len(self.samples) + self.N_SAMPLES >= sample_limit
                    else self.N_SAMPLES
                )
                self.samples.extend(
                    self.datamodel.generate_dataset(
                        n_samples=samples_to_generate,
                    ),
                )

    def train_tree(
        self,
        performance_threshold: float,
        sample_limit: int,
        n_predictions: int,
        *,
        classification: bool,
        **kwargs: Any,
    ) -> HoeffdingTreeRegressor | HoeffdingTreeClassifier:
        """Train a Hoeffding Tree using synthetic samples.

        Continue until performance criteria are met.

        Args:
            performance_threshold : Minimum performance required to
                finalize the model.
            sample_limit : Maximum number of samples to use during training.
            n_predictions : Number of consecutive correct predictions
                required to stop training.
            classification : Indicates whether the task is
                classification or regression.
            **kwargs : Additional hyperparameters for the Hoeffding Tree model.

        Returns:
            Trained Hoeffding Tree model.
        """
        metric, model = self._create_model_and_metric(
            classification=classification,
            **kwargs,
        )

        # Pre-train
        for x, y in self.samples:
            y_true = self._normalize_target(y, classification=classification)
            model.learn_one(x, y_true)

        self._run_training_loop(
            model=model,
            metric=metric,
            performance_threshold=performance_threshold,
            n_predictions=n_predictions,
            sample_limit=sample_limit,
            classification=classification,
        )

        logger.info("Hoeffding Tree training completed successfully.")
        return model
