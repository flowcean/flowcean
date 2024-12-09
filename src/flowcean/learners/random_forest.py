import logging
from typing import Any, override

import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz

from flowcean.core.learner import SupervisedLearner
from flowcean.core.model import Model
from flowcean.models.sklearn import SciKitModel
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
        dot_graph_export_path: None | str = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the random forest learner.

        Args:
            *args: Positional arguments to pass to the RandomForestRegressor.
            dot_graph_export_path: Path to export a tree from the random forest to.
            **kwargs: Keyword arguments to pass to the RandomForestRegressor.
        """
        self.regressor = RandomForestRegressor(
            *args, **kwargs, random_state=get_seed()
        )
        self.dot_graph_export_path = dot_graph_export_path

    @override
    def learn(
        self,
        inputs: pl.DataFrame,
        outputs: pl.DataFrame,
    ) -> Model:
        """Fit the random forest regressor on the given inputs and outputs."""
        self.regressor.fit(inputs, outputs)
        print("Using Random Forest Regressor")
        if self.dot_graph_export_path is not None:
            # Exporting one tree from the forest (usually the first tree) as a dot graph
            logger.info(
                "Exporting one tree from the random forest to %s",
                self.dot_graph_export_path,
            )
            export_graphviz(
                self.regressor.estimators_[0],  # Exporting the first tree
                out_file=self.dot_graph_export_path,
            )
        return SciKitModel(self.regressor, outputs.columns[0])
