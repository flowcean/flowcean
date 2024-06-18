import logging
from typing import Any, override

import polars as pl
from sklearn.tree import DecisionTreeRegressor, export_graphviz

from flowcean.core import Model, SupervisedLearner
from flowcean.models.sklearn import SciKitModel

logger = logging.getLogger(__name__)


class RegressionTree(SupervisedLearner):
    """Wrapper class for sklearn's DecisionTreeRegressor.

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    """

    regressor: DecisionTreeRegressor

    def __init__(
        self,
        *,
        dot_graph_export_path: None | str = None,
        **kwargs: Any,
    ) -> None:
        self.regressor = DecisionTreeRegressor(**kwargs)
        self.dot_graph_export_path = dot_graph_export_path

    @override
    def learn(
        self,
        inputs: pl.DataFrame,
        outputs: pl.DataFrame,
    ) -> Model:
        self.regressor.fit(inputs, outputs)
        if self.dot_graph_export_path is not None:
            logger.info(
                "Exporting decision tree graph to %s",
                self.dot_graph_export_path,
            )
            export_graphviz(
                self.regressor,
                out_file=self.dot_graph_export_path,
                feature_names=inputs.columns,
            )
        return SciKitModel(self.regressor, outputs.columns[0])
