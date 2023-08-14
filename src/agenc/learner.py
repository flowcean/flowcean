import numpy as np


class Learner:
    def train(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        pass

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
