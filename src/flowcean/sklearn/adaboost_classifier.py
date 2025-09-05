import logging
from typing import Any

import numpy as np
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
        *args: Any,
        base_estimator: object = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the AdaBoost classifier learner.

        Args:
            *args: Positional arguments to pass to the AdaBoostClassifier.
            base_estimator: The base estimator from which the boosted ensemble
                is built. If None, then the base estimator is
                DecisionTreeClassifier(max_depth=1).
            n_estimators: The maximum number of estimators at which boosting is
                terminated. Defaults to 50.
            learning_rate: Learning rate shrinks the contribution of each
                classifier. Defaults to 1.0.
            **kwargs: Keyword arguments to pass to the AdaBoostClassifier.
        """
        self.classifier = AdaBoostClassifier(
            *args,
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=get_seed(),
            **kwargs,
        )

    @override
    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> Model:
        collected_inputs = inputs.collect()
        collected_outputs = outputs.collect()

        # Explicitly convert inputs to a numeric NumPy array
        x = collected_inputs.to_numpy()
        if not np.issubdtype(x.dtype, np.number):
            msg = f"Input data contains non-numeric values: {x.dtype}"
            raise ValueError(
                msg,
            )
        if np.any(np.isnan(x)):
            logger.warning("Input data contains NaN values; replacing with 0")
            x = np.nan_to_num(x, nan=0.0)

        # Convert outputs to 1D numeric array
        y = collected_outputs.to_numpy().ravel()
        if not np.issubdtype(y.dtype, np.number):
            msg = f"Output data contains non-numeric values: {y.dtype}"
            raise ValueError(
                msg,
            )

        self.classifier.fit(x, y)
        return SciKitModel(
            self.classifier,
            output_names=[collected_outputs.columns[0]],
        )
