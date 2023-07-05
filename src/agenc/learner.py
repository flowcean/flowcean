import numpy as np

from agenc.data import Dataset


class Learner:
    def train(self, dataset: Dataset):
        pass

    def predict(self, dataset: Dataset) -> np.ndarray:
        raise NotImplementedError
