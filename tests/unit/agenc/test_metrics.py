import unittest

import numpy as np
import pytest

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


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.true = np.array([0, 1, 0, 1])
        self.pred = np.array([1, 1, 0, 0])
        self.wrong_length = np.array([0, 0])
        self.empty = np.array([])

    def test_accuracy(self):
        accuracy = Accuracy()

        with pytest.raises(ValueError):
            accuracy(self.true, self.wrong_length)

        with pytest.warns(RuntimeWarning):
            accuracy(self.empty, self.empty)

    def test_classification_report(self):
        classification_report = ClassificationReport()

        with pytest.raises(ValueError):
            classification_report(self.true, self.wrong_length)

        with pytest.raises(ValueError):
            classification_report(self.empty, self.empty)

    def test_fbeta_score(self):
        fbeta_score = FBetaScore(beta=1.0)

        with pytest.raises(ValueError):
            fbeta_score(self.true, self.wrong_length)

        with pytest.warns(Warning):
            fbeta_score(self.empty, self.empty)

    def test_precision_score(self):
        precision_score = PrecisionScore()

        with pytest.raises(ValueError):
            precision_score(self.true, self.wrong_length)

        with pytest.warns(Warning):
            precision_score(self.empty, self.empty)

    def test_recall(self):
        recall = Recall()

        with pytest.raises(ValueError):
            recall(self.true, self.wrong_length)

        with pytest.warns(Warning):
            recall(self.empty, self.empty)

    def test_max_error(self):
        max_error = MaxError()

        with pytest.raises(ValueError):
            max_error(self.true, self.wrong_length)

        with pytest.raises(ValueError):
            max_error(self.empty, self.empty)

    def test_mean_absolute_error(self):
        mean_absolute_error = MeanAbsoluteError()

        with pytest.raises(ValueError):
            mean_absolute_error(self.true, self.wrong_length)

        with pytest.raises(ValueError):
            mean_absolute_error(self.empty, self.empty)

    def test_mean_squared_error(self):
        mean_squared_error = MeanSquaredError()

        with pytest.raises(ValueError):
            mean_squared_error(self.true, self.wrong_length)

        with pytest.raises(ValueError):
            mean_squared_error(self.empty, self.empty)

    def test_r2_score(self):
        r2_score = R2Score()

        with pytest.raises(ValueError):
            r2_score(self.true, self.wrong_length)

        with pytest.raises(ValueError):
            r2_score(self.empty, self.empty)


if __name__ == "__main__":
    unittest.main()
