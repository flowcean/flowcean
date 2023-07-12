from sklearn import metrics
from .metric import Metric


class MaxError(Metric):
    def __call__(self, y_true, y_pred, *args):
        return metrics.max_error(y_true, y_pred)


class MeanAbsoluteError(Metric):
    def __call__(
        self, y_true, y_pred, *, sample_weight=None, multioutput="uniform_average"
    ):
        return metrics.mean_absolute_error(y_true, y_pred, sample_weight, multioutput)


class MeanSquaredError(Metric):
    def __call__(
        self,
        y_true,
        y_pred,
        *,
        sample_weight=None,
        multioutput="uniform_average",
        squared=True
    ):
        return metrics.mean_squared_error(
            y_true, y_pred, sample_weight, multioutput, squared
        )


class R2Score(Metric):
    def __call__(
        self,
        y_true,
        y_pred,
        *,
        sample_weight=None,
        multioutput="uniform_average",
        force_finite=True
    ):
        return metrics.r2_score(
            y_true, y_pred, sample_weight, multioutput, force_finite
        )
