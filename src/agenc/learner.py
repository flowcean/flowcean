import numpy as np
from abc import ABC, abstractmethod


class Learner(ABC):
    @abstractmethod
    def train(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        ...
