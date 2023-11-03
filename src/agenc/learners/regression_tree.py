from typing import Any, cast

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
