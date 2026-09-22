"""Tests for the scikit-learn metrics facade."""

from flowcean import sklearn
from flowcean.sklearn import metrics
from flowcean.sklearn.metrics._multioutput import MultiOutputMixin

EXPECTED_METRICS = (
    "Accuracy",
    "ClassificationReport",
    "FBetaScore",
    "MaxError",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanSquaredError",
    "MultiOutputMixin",
    "PrecisionScore",
    "R2Score",
    "Recall",
)


def test_metrics_facade_exports_the_complete_metric_api() -> None:
    assert metrics.__all__ == EXPECTED_METRICS
    assert metrics.MultiOutputMixin is MultiOutputMixin

    for name in metrics.__all__:
        if name != "MultiOutputMixin":
            assert getattr(sklearn, name) is getattr(metrics, name)

    assert "MultiOutputMixin" not in sklearn.__all__
    assert not hasattr(sklearn, "MultiOutputMixin")
