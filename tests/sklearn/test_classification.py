import unittest

import polars as pl

from flowcean.sklearn import (
    Accuracy,
    ClassificationReport,
    FBetaScore,
    PrecisionScore,
    Recall,
)


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.true = pl.DataFrame({"a": [0, 1, 0, 1]}).lazy()
        self.predicted = pl.DataFrame({"a": [1, 1, 0, 0]}).lazy()

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


if __name__ == "__main__":
    unittest.main()
