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
        accuracy = Accuracy()(self.true, self.predicted)
        assert accuracy == 0.5

    def test_classification_report(self) -> None:
        report_value = ClassificationReport()(self.true, self.predicted)
        assert report_value.startswith("              precision    recall ")

    def test_fbeta_score(self) -> None:
        fbeta = FBetaScore(beta=1.0)(self.true, self.predicted)
        assert fbeta == 0.5

    def test_precision_score(self) -> None:
        precision = PrecisionScore()(self.true, self.predicted)
        assert precision == 0.5

    def test_recall(self) -> None:
        recall = Recall()(self.true, self.predicted)
        assert recall == 0.5

    def test_max_error(self) -> None:
        max_error = MaxError()(self.true, self.predicted)
        assert max_error == 1

    def test_mean_absolute_error(self) -> None:
        mae = MeanAbsoluteError()(self.true, self.predicted)
        assert mae == 0.5

    def test_mean_squared_error(self) -> None:
        mse = MeanSquaredError()(self.true, self.predicted)
        assert mse == 0.5

    def test_r2_score(self) -> None:
        r2 = R2Score()(self.true, self.predicted)
        assert r2 == -1.0

    def test_dummy_metric(self) -> None:
        dummy_value = DummyMetric()(self.true, self.predicted)
        assert dummy_value == 0.0


if __name__ == "__main__":
    unittest.main()
