import unittest

import polars as pl

from flowcean.polars import DummyLearner


class TestDummyLearner(unittest.TestCase):
    def test_dummy_learner(self) -> None:
        learner = DummyLearner()
        dataset = pl.DataFrame(
            {
                "input": [1, 2, 3],
                "output": [4, 5, 6],
            },
        ).lazy()
        model = learner.learn(
            dataset.select("input"),
            dataset.select("output"),
        )
        predictions = model.predict(dataset.select(["input"]))
        assert predictions.collect().shape == (3, 1)


if __name__ == "__main__":
    unittest.main()
