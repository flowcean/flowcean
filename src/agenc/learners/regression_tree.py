from pathlib import Path
from typing import Any, cast

import joblib
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

    def train(self, inputs: NDArray[Any], outputs: NDArray[Any]) -> None:
        self.regressor.fit(inputs, outputs)

    def predict(self, inputs: NDArray[Any]) -> NDArray[Any]:
        return cast(NDArray[Any], self.regressor.predict(inputs))

    def save(self, path: Path) -> None:
        joblib.dump(self.regressor, path)

    def load(self, path: Path) -> None:
        self.regressor = joblib.load(path)
