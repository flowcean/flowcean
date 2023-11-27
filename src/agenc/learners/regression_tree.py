from pathlib import Path
from typing import Any, cast

import joblib
import polars as pl
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeRegressor

from agenc.core import Learner


class RegressionTree(Learner):
    """Wrapper class for sklearn's DecisionTreeRegressor.

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    """

    regressor: DecisionTreeRegressor

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.regressor = DecisionTreeRegressor(*args, **kwargs)

    def train(
        self,
        data: pl.DataFrame,
        inputs: list[str],
        outputs: list[str],
    ) -> None:
        input_data = data.select(inputs).to_numpy()
        output_data = data.select(outputs).to_numpy()
        self.regressor.fit(input_data, output_data)

    def predict(self, inputs: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], self.regressor.predict(inputs))

    def save(self, path: Path) -> None:
        joblib.dump(self.regressor, path)

    def load(self, path: Path) -> None:
        self.regressor = joblib.load(path)
