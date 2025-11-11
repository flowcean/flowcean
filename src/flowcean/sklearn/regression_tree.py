import logging

import polars as pl
from numpy.typing import NDArray
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
        *,
        dot_graph_export_path: None | str = None,
        criterion: str = "squared_error",
        splitter: str = "best",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: float | None = None,
        random_state: int | None = None,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        monotonic_cst: NDArray | None = None,
    ) -> None:
        """Initialize the regression tree learner.

        Reference: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
        """
        self.regressor = DecisionTreeRegressor(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            random_state=random_state or get_seed(),
            ccp_alpha=ccp_alpha,
            monotonic_cst=monotonic_cst,
        )
        self.dot_graph_export_path = dot_graph_export_path

    @override
    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> Model:
        dfs = pl.collect_all([inputs, outputs])
        collected_inputs = dfs[0]
        collected_outputs = dfs[1]
        self.regressor.fit(collected_inputs, collected_outputs)
        if self.dot_graph_export_path is not None:
            logger.info(
                "Exporting decision tree graph to %s",
                self.dot_graph_export_path,
            )
            export_graphviz(
                self.regressor,
                out_file=self.dot_graph_export_path,
                feature_names=collected_inputs.columns,
            )
        return SciKitModel(
            self.regressor,
            input_names=collected_inputs.columns,
            output_names=collected_outputs.columns,
        )
