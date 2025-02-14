import unittest
from io import BytesIO

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.core.model import Model, ModelWithTransform
from flowcean.environments.dataset import Dataset
from flowcean.environments.train_test_split import TrainTestSplit
from flowcean.learners.linear_regression import LinearRegression
from flowcean.learners.regression_tree import RegressionTree
from flowcean.models.pytorch import PyTorchModel
from flowcean.models.sklearn import SciKitModel
from flowcean.strategies.incremental import learn_incremental
from flowcean.strategies.offline import learn_offline
from flowcean.transforms import Select


class TestSaveLoad(unittest.TestCase):
    def test_save_load_sklearn(self) -> None:
        n = 100
        data = Dataset(
            pl.DataFrame(
                {
                    "x": pl.arange(0, n, eager=True).cast(pl.Float32) / n,
                    "y": pl.arange(n, 0, -1, eager=True).cast(pl.Float32) / n,
                },
            ),
        )

        train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(
            data,
        )

        learner = RegressionTree()
        model = learn_offline(
            train_env,
            learner,
            ["x"],
            ["y"],
        )

        assert isinstance(model, SciKitModel)
        model_bytes = BytesIO()
        model.save(model_bytes)
        model_bytes.seek(0)

        loaded_model = Model.load(model_bytes)
        assert isinstance(loaded_model, SciKitModel)

        # Test a random prediction
        test_frame = pl.DataFrame(
            {"x": [0.5]},
        ).lazy()
        assert_frame_equal(
            model.predict(test_frame).collect(),
            loaded_model.predict(test_frame).collect(),
        )

    def test_save_load_pytorch(self) -> None:
        n = 100
        data = Dataset(
            pl.DataFrame(
                {
                    "x": pl.arange(0, n, eager=True).cast(pl.Float32) / n,
                    "y": pl.arange(n, 0, -1, eager=True).cast(pl.Float32) / n,
                },
            ),
        )

        train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(
            data,
        )

        learner = LinearRegression(
            input_size=1,
            output_size=1,
            learning_rate=0.01,
        )
        model = learn_incremental(
            train_env.as_stream(batch_size=1),
            learner,
            ["x"],
            ["y"],
        )

        assert isinstance(model, PyTorchModel)
        model_bytes = BytesIO()
        model.save(model_bytes)
        model_bytes.seek(0)

        loaded_model = Model.load(model_bytes)
        assert isinstance(loaded_model, PyTorchModel)

        # Test a random prediction
        test_frame = pl.DataFrame(
            {"x": [0.5]},
        ).lazy()
        assert_frame_equal(
            model.predict(test_frame).collect(),
            loaded_model.predict(test_frame).collect(),
        )

    def test_save_load_model_with_transforms(self) -> None:
        n = 100
        data = Dataset(
            pl.DataFrame(
                {
                    "x": pl.arange(0, n, eager=True).cast(pl.Float32) / n,
                    "y": pl.arange(n, 0, -1, eager=True).cast(pl.Float32) / n,
                },
            ),
        )

        train_env, test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(
            data,
        )

        learner = RegressionTree()
        model = learn_offline(
            train_env,
            learner,
            ["x"],
            ["y"],
            input_transform=Select(pl.col("x") * 2),
        )

        assert isinstance(model, ModelWithTransform)
        model_bytes = BytesIO()
        model.save(model_bytes)
        model_bytes.seek(0)

        loaded_model = Model.load(model_bytes)
        assert isinstance(loaded_model, ModelWithTransform)

        # Test a random prediction
        test_frame = pl.DataFrame(
            {"x": [0.5]},
        ).lazy()
        assert_frame_equal(
            model.predict(test_frame).collect(),
            loaded_model.predict(test_frame).collect(),
        )


if __name__ == "__main__":
    unittest.main()
