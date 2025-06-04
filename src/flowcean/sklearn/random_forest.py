import logging
from typing import Any

import polars as pl
from sklearn.ensemble import RandomForestRegressor
from typing_extensions import override

from flowcean.core.learner import SupervisedLearner
from flowcean.core.model import Model
from flowcean.sklearn import SciKitModel
from flowcean.utils.random import get_seed

logger = logging.getLogger(__name__)


class RandomForestRegressorLearner(SupervisedLearner):
    """Wrapper class for sklearn's RandomForestRegressor.

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    """

    regressor: RandomForestRegressor

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the random forest learner.

        Args:
            *args: Positional arguments to pass to the RandomForestRegressor.
            **kwargs: Keyword arguments to pass to the RandomForestRegressor.
        """
        self.regressor = RandomForestRegressor(
            *args,
            **kwargs,
            random_state=get_seed(),
        )

    @override
    def learn(
        self,
        inputs: pl.DataFrame,
        outputs: pl.DataFrame,
    ) -> Model:
        """Fit the random forest regressor on the given inputs and outputs."""
        self.regressor.fit(inputs, outputs)
        logger.info("Using Random Forest Regressor")
        return SciKitModel(self.regressor, outputs.columns[0])
