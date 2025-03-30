import logging
from typing import Any

import polars as pl
from sklearn.ensemble import AdaBoostClassifier
from typing_extensions import override

from flowcean.core import Model, SupervisedLearner
from flowcean.utils import get_seed

from .model import SciKitModel

logger = logging.getLogger(__name__)


class AdaBoost(SupervisedLearner):
    """AdaBoost classifier that boosts any SupervisedLearner.

    Args:
        base_learner: The base learner to boost.
        base_input_features: Features for the base learner.
        boost_features: Additional features for AdaBoost to use in boosting.
        n_estimators: Number of boosting iterations.
        learning_rate: Learning rate for AdaBoost.
    """

    def __init__(
        self,
        base_learner: SupervisedLearner,
        base_input_features: list[str],
        boost_features: list[str],
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.base_learner = base_learner
        self.base_input_features = base_input_features
        self.boost_features = boost_features
        self.classifier = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=get_seed(),
            **kwargs,
        )
        self.base_model = NotImplemented

    @override
    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> Model:
        """Train the AdaBoost classifier with the base learner's predictions.

        Args:
            inputs: Input features as a Polars LazyFrame (all features).
            outputs: Target values as a Polars LazyFrame.

        Returns:
            A trained Model instance wrapping the AdaBoostClassifier.
        """
        collected_inputs = inputs.collect()
        collected_outputs = outputs.collect()

        # Train the base learner on its specific input features
        base_inputs = inputs.select(self.base_input_features)
        self.base_model = self.base_learner.learn(base_inputs, outputs)
        # Get base learner predictions
        base_predictions = (
            self.base_model.predict(base_inputs).collect().to_numpy().ravel()
        )

        # Combine base predictions with boost features for AdaBoost
        boost_data = (
            collected_inputs.select(self.boost_features)
            .with_columns(
                pl.Series("base_pred", base_predictions),
            )
            .to_numpy()
        )

        # Fit AdaBoost
        self.classifier.fit(boost_data, collected_outputs.to_numpy().ravel())

        return SciKitModel(self.classifier, collected_outputs.columns[0])
