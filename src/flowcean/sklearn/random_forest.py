import logging

import polars as pl
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from typing_extensions import override

from flowcean.core import LearnerCallback, create_callback_manager
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
        n_estimators: int = 100,
        *,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: float = 1.0,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        max_samples: int | float | None = None,  # noqa: PYI041
        monotonic_cst: NDArray | None = None,
        callbacks: list[LearnerCallback] | LearnerCallback | None = None,
    ) -> None:
        """Initialize the random forest learner.

        Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

        Args:
            n_estimators: Number of trees in the forest.
            criterion: Function to measure the quality of a split.
            max_depth: Maximum depth of the tree.
            min_samples_split: Minimum number of samples required to split
                an internal node.
            min_samples_leaf: Minimum number of samples required to be at
                a leaf node.
            min_weight_fraction_leaf: Minimum weighted fraction of the sum
                total of weights required to be at a leaf node.
            max_features: Number of features to consider when looking for
                the best split.
            max_leaf_nodes: Grow trees with max_leaf_nodes in best-first
                fashion.
            min_impurity_decrease: A node will be split if this split
                induces a decrease of the impurity greater than or equal
                to this value.
            bootstrap: Whether bootstrap samples are used when building trees.
            oob_score: Whether to use out-of-bag samples to estimate the R^2
                on unseen data.
            n_jobs: Number of jobs to run in parallel.
            random_state: Controls the randomness of the estimator.
            verbose: Controls the verbosity when fitting and predicting.
            warm_start: When set to True, reuse the solution of the previous
                call to fit.
            ccp_alpha: Complexity parameter used for Minimal Cost-Complexity
                Pruning.
            max_samples: If bootstrap is True, the number of samples to draw
                from X to train each base estimator.
            monotonic_cst: Monotonicity constraints.
            callbacks: Optional callbacks for progress feedback. Defaults to
                RichCallback if not specified.
        """
        self.regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state or get_seed(),
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )
        self.callback_manager = create_callback_manager(callbacks)

    @override
    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> Model:
        """Fit the random forest regressor on the given inputs and outputs."""
        dfs = pl.collect_all([inputs, outputs])
        collected_inputs = dfs[0]
        collected_outputs = dfs[1]

        # Notify callbacks that learning is starting
        context = {
            "n_estimators": self.regressor.n_estimators,
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
            logger.info("Using Random Forest Regressor")

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
