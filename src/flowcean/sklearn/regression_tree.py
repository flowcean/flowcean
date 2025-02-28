import logging
from typing import Any

import polars as pl
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from typing_extensions import override

from flowcean.core import Model, SupervisedLearner
from flowcean.utils import get_seed

from .model import SciKitModel

logger = logging.getLogger(__name__)


class RegressionTree(SupervisedLearner):
    """Wrapper class for sklearn's DecisionTreeRegressor.

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    """

    regressor: DecisionTreeRegressor

    def __init__(
        self,
        *args: Any,
        dot_graph_export_path: None | str = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the regression tree learner.

        Args:
            *args: Positional arguments to pass to the DecisionTreeRegressor.
            dot_graph_export_path: Path to export the decision tree graph to.
            **kwargs: Keyword arguments to pass to the DecisionTreeRegressor.
        """
        self.regressor = DecisionTreeRegressor(
            *args,
            **kwargs,
            random_state=get_seed(),
        )
        self.dot_graph_export_path = dot_graph_export_path

    @override
    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> Model:
        collected_inputs = inputs.collect()
        collected_outputs = outputs.collect()
        self.regressor.fit(collected_inputs, collected_outputs)
        if self.dot_graph_export_path is not None:
            logger.info(
                "Exporting decision tree graph to %s",
                self.dot_graph_export_path,
            )
            export_graphviz(
                self.regressor,
                out_file=self.dot_graph_export_path,
            )
        return SciKitModel(self.regressor, collected_outputs.columns[0])
