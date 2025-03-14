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
    """Wrapper class for sklearn's AdaBoostClassifier.

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    """

    classifier: AdaBoostClassifier

    def __init__(
        self,
        estimator: Any = None,
        base_estimator: Any = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        *,
        random_state: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the AdaBoost classifier learner.

        Args:
            estimator: The base estimator from which the boosted ensemble is
                built (newer sklearn versions).
            base_estimator: The base estimator from which the boosted ensemble
                is built (deprecated).
            n_estimators: The maximum number of estimators at which boosting
                is terminated.
            learning_rate: Weight applied to each classifier at each boosting
                iteration.
            random_state: Controls the randomness of the estimator.
            **kwargs: Additional keyword arguments to pass to the
                AdaBoostClassifier.
        """
        # Use get_seed() as default random_state if not provided
        random_state = random_state if random_state is not None else get_seed()

        # Handle both base_estimator (older versions)
        # and estimator (newer versions) for compatibility
        if estimator is not None:
            self.classifier = AdaBoostClassifier(
                estimator=estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                **kwargs,
            )
        else:
            self.classifier = AdaBoostClassifier(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state,
                **kwargs,
            )

    @override
    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> Model:
        """Train the AdaBoost classifier.

        Args:
            inputs: Input features as a Polars LazyFrame.
            outputs: Target values as a Polars LazyFrame.

        Returns:
            A trained Model instance wrapping the AdaBoostClassifier.
        """
        collected_inputs = inputs.collect()
        collected_outputs = outputs.collect()

        # Fit the classifier - assuming outputs is a single column
        self.classifier.fit(
            collected_inputs,
            collected_outputs.to_numpy().ravel(),
        )

        return SciKitModel(self.classifier, collected_outputs.columns[0])
