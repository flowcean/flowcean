from typing import Any

import numpy as np


class Metric:
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __call__(self, y_true: np.ndarray, y_predicted: np.ndarray) -> Any:
        raise NotImplementedError
