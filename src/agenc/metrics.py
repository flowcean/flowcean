import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Metric:
    @property
    def name(self):
        return self.__class__.__name__

    def __call__(self, y, y_):
        raise NotImplementedError


class RMSE(Metric):
    def __call__(self, y, y_):
        return np.sqrt(mean_squared_error(y, y_))


class MAE(Metric):
    def __call__(self, y, y_):
        return mean_absolute_error(y, y_)
