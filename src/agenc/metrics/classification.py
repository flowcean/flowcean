from sklearn import metrics
from .metric import Metric


class Accuracy(Metric):
    def __call__(self, y_true, y_pred, *, normalize=True, sample_weight=None):
        return metrics.accuracy_score(y_true, y_pred, normalize, sample_weight)


class ClassificationReport(Metric):
    def __call__(
        self,
        y_true,
        y_pred,
        *,
        labels=None,
        target_names=None,
        sample_weight=None,
        digits=2,
        output_dict=False,
        zero_division="warn",
    ):
        return metrics.classification_report(
            y_true,
            y_pred,
            labels,
            target_names,
            sample_weight,
            digits,
            output_dict,
            zero_division,
        )


class F1Score(Metric):
    def __call__(
        self,
        y_true,
        y_pred,
        *,
        labels=None,
        pos_label=1,
        average="binary",
        sample_weight=None,
        zero_division="warn",
    ):
        return metrics.f1_score(
            y_true, y_pred, labels, pos_label, average, sample_weight, zero_division
        )


class FBetaScore(Metric):
    def __call__(
        self,
        y_true,
        y_pred,
        *,
        beta,
        labels=None,
        pos_label=1,
        average="binary",
        sample_weight=None,
        zero_division="warn",
    ):
        return metrics.fbeta_score(
            y_true,
            y_pred,
            beta,
            labels,
            pos_label,
            average,
            sample_weight,
            zero_division,
        )


class PrecisionScore(Metric):
    def __call__(
        self,
        y_true,
        y_pred,
        *,
        labels=None,
        pos_label=1,
        average="binary",
        sample_weight=None,
        zero_division="warn",
    ):
        return metrics.precision_score(
            y_true, y_pred, labels, pos_label, average, sample_weight, zero_division
        )


class Recall(Metric):
    def __call__(
        self,
        y_true,
        y_pred,
        *,
        labels=None,
        pos_label=1,
        average="binary",
        sample_weight=None,
        zero_division="warn",
    ):
        return metrics.recall_score(
            y_true, y_pred, labels, pos_label, average, sample_weight, zero_division
        )
