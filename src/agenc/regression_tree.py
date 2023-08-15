import numpy as np
from sklearn.tree import DecisionTreeRegressor


class RegressionTree:
    """Wrapper class for sklearn's DecisionTreeRegressor

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    """

    regressor: DecisionTreeRegressor

    def __init__(self, *args, **kwargs) -> None:
        self.regressor = DecisionTreeRegressor(*args, **kwargs)

    def train(self, inputs: np.ndarray, outputs: np.ndarray):
        self.regressor.fit(inputs, outputs)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        y = self.regressor.predict(inputs)
        return y
