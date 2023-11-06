import unittest

import numpy as np
from agenc.metrics import (
    Accuracy,
    ClassificationReport,
    FBetaScore,
    MaxError,
    MeanAbsoluteError,
    MeanSquaredError,
    PrecisionScore,
    R2Score,
    Recall,
)
from agenc.metrics.dummy_metric import DummyMetric

class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.true = np.array([0, 1, 0, 1])
        self.predicted = np.array([1, 1, 0, 0])

    def test_name(self) -> None:
        assert Accuracy().name == "Accuracy"

    def test_accuracy(self) -> None:
        Accuracy()(self.true, self.predicted)

    def test_classification_report(self) -> None:
        ClassificationReport()(self.true, self.predicted)

    def test_fbeta_score(self) -> None:
        FBetaScore(beta=1.0)(self.true, self.predicted)

    def test_precision_score(self) -> None:
        PrecisionScore()(self.true, self.predicted)

    def test_recall(self) -> None:
        Recall()(self.true, self.predicted)

    def test_max_error(self) -> None:
        MaxError()(self.true, self.predicted)

    def test_mean_absolute_error(self) -> None:
        MeanAbsoluteError()(self.true, self.predicted)

    def test_mean_squared_error(self) -> None:
        MeanSquaredError()(self.true, self.predicted)

    def test_r2_score(self) -> None:
        R2Score()(self.true, self.predicted)

    def test_dummy_metric(self):
        dummy_metric = DummyMetric()

        # Test with some example data
        y_true = np.ndarray([1, 2, 3])
        y_predicted = np.ndarray([1, 2, 3])
        metric_value = dummy_metric(y_true, y_predicted)

        # The DummyMetric always returns 0, so the metric value should be 0
        self.assertEqual(metric_value, 0)


if __name__ == "__main__":
    unittest.main()
