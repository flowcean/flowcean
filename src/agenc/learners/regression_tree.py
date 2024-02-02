import logging
from pathlib import Path
from typing import Any, cast

import joblib
import polars as pl
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeRegressor, export_graphviz

from agenc.core import Learner

logger = logging.getLogger(__name__)


class RegressionTree(Learner):
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
        self.regressor = DecisionTreeRegressor(*args, **kwargs)
        self.dot_graph_export_path = dot_graph_export_path

    def train(
        self,
        input_features: pl.DataFrame,
        output_features: pl.DataFrame,
    ) -> None:
        input_data = input_features.to_numpy()
        output_data = output_features.to_numpy()
        self.regressor.fit(input_data, output_data)
        if self.dot_graph_export_path is not None:
            logger.info(
                "Exporting decision tree graph to"
                f" {self.dot_graph_export_path}"
            )
            export_graphviz(
                self.regressor,
                out_file=self.dot_graph_export_path,
            )

    def predict(self, input_features: pl.DataFrame) -> NDArray[Any]:
        return cast(
            NDArray[Any], self.regressor.predict(input_features.to_numpy())
        )

    def save(self, path: Path) -> None:
        joblib.dump(self.regressor, path)

    def load(self, path: Path) -> None:
        self.regressor = joblib.load(path)
