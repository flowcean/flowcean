import logging
from pathlib import Path
from typing import Any

import joblib
import polars as pl
from sklearn.tree import DecisionTreeRegressor, export_graphviz

from agenc.core import Learner, Model

logger = logging.getLogger(__name__)


class SciKitModel(Model):
    def __init__(
        self,
        model: Any,
        output_name: str,
    ) -> None:
        self.model = model
        self.output_name = output_name

    def predict(
        self,
        input_features: pl.DataFrame,
    ) -> pl.DataFrame:
        outputs = self.model.predict(input_features)
        return pl.DataFrame({self.output_name: outputs})

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)


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
    ) -> Model:
        self.regressor.fit(input_features, output_features)
        if self.dot_graph_export_path is not None:
            logger.info(
                "Exporting decision tree graph to"
                f" {self.dot_graph_export_path}"
            )
            export_graphviz(
                self.regressor,
                out_file=self.dot_graph_export_path,
            )
        return SciKitModel(self.regressor, output_features.columns[0])
