import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y, y_):
    return np.sqrt(mean_squared_error(y, y_))


def mae(y, y_):
    return mean_absolute_error(y, y_)
