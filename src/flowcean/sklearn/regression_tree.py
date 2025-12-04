import logging

import polars as pl
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from typing_extensions import override

from flowcean.core import (
    LearnerCallback,
    Model,
    SupervisedLearner,
    create_callback_manager,
)
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
        callbacks: list[LearnerCallback] | LearnerCallback | None = None,
    ) -> None:
        """Initialize the regression tree learner.

        Reference: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

        Args:
            dot_graph_export_path: Path to export the decision tree graph
                in Graphviz DOT format.
            criterion: Function to measure the quality of a split.
            splitter: Strategy used to choose the split at each node.
            max_depth: Maximum depth of the tree.
            min_samples_split: Minimum number of samples required to split
                an internal node.
            min_samples_leaf: Minimum number of samples required to be at
                a leaf node.
            min_weight_fraction_leaf: Minimum weighted fraction of the sum
                total of weights required to be at a leaf node.
            max_features: Number of features to consider when looking for
                the best split.
            random_state: Controls the randomness of the estimator.
            max_leaf_nodes: Grow a tree with max_leaf_nodes in best-first
                fashion.
            min_impurity_decrease: A node will be split if this split
                induces a decrease of the impurity greater than or equal
                to this value.
            ccp_alpha: Complexity parameter used for Minimal Cost-Complexity
                Pruning.
            monotonic_cst: Monotonicity constraints.
            callbacks: Optional callbacks for progress feedback. Defaults to
                RichCallback if not specified.
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
        self.callback_manager = create_callback_manager(callbacks)

    @override
    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> Model:
        dfs = pl.collect_all([inputs, outputs])
        collected_inputs = dfs[0]
        collected_outputs = dfs[1]

        # Notify callbacks that learning is starting
        context = {
            "max_depth": self.regressor.max_depth or "unlimited",
            "n_samples": len(collected_inputs),
            "n_features": len(collected_inputs.columns),
        }
        self.callback_manager.on_learning_start(self, context)

        try:
            # Fit the model (flatten outputs to 1D if single column)
            outputs_array = collected_outputs.to_numpy()
            if outputs_array.shape[1] == 1:
                outputs_array = outputs_array.ravel()
            self.regressor.fit(collected_inputs, outputs_array)

            # Export graph if requested
            if self.dot_graph_export_path is not None:
                logger.info(
                    "Exporting decision tree graph to %s",
                    self.dot_graph_export_path,
                )
                export_graphviz(
                    self.regressor,
                    out_file=self.dot_graph_export_path,
                )

            # Create the model
            model = SciKitModel(
                self.regressor,
                output_names=outputs.collect_schema().names(),
            )

            # Notify callbacks that learning is complete
            self.callback_manager.on_learning_end(self, model)
        except Exception as e:
            # Notify callbacks of the error
            self.callback_manager.on_learning_error(self, e)
            raise
        else:
            return model
