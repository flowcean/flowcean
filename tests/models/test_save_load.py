import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.core import Model, learn_incremental, learn_offline
from flowcean.hydra import (
    HybridDecisionTreeLearner,
    HyDRAModel,
    SelectorFeatureConfig,
)
from flowcean.polars import (
    DataFrame,
    Select,
    StreamingOfflineEnvironment,
    TrainTestSplit,
)
from flowcean.sklearn import RegressionTree, SciKitModel
from flowcean.torch import LinearRegression, PyTorchModel


class ConstantModel(Model):
    def __init__(self, value: float, output_name: str = "y") -> None:
        self.value = value
        self.output_name = output_name

    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        return pl.DataFrame(
            {self.output_name: [self.value] * frame.height},
        ).lazy()


class TestSaveLoad(unittest.TestCase):
    def test_save_load_sklearn(self) -> None:
        n = 100
        data = DataFrame(
            pl.DataFrame(
                {
                    "x": pl.arange(0, n, eager=True).cast(pl.Float32) / n,
                    "y": pl.arange(n, 0, -1, eager=True).cast(pl.Float32) / n,
                },
            ),
        )

        train_env, _test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(
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

        test_frame = pl.DataFrame(
            {"x": [0.5]},
        ).lazy()
        assert_frame_equal(
            model.predict(test_frame).collect(),
            loaded_model.predict(test_frame).collect(),
        )

    def test_save_load_model_via_path(self) -> None:
        model = ConstantModel(3.14)

        with TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "models" / "constant.fml"
            model.save(model_path)

            loaded_model = Model.load(model_path)

        assert isinstance(loaded_model, ConstantModel)
        assert_frame_equal(
            loaded_model.predict(
                pl.DataFrame({"x": [0.0, 1.0]}).lazy(),
            ).collect(),
            pl.DataFrame({"y": [3.14, 3.14]}),
        )

    def test_save_load_pytorch(self) -> None:
        n = 100
        data = DataFrame(
            pl.DataFrame(
                {
                    "x": pl.arange(0, n, eager=True).cast(pl.Float32) / n,
                    "y": pl.arange(n, 0, -1, eager=True).cast(pl.Float32) / n,
                },
            ),
        )

        train_env, _test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(
            data,
        )

        learner = LinearRegression(
            output_size=1,
            learning_rate=0.01,
        )
        model = learn_incremental(
            StreamingOfflineEnvironment(train_env, batch_size=1),
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

        test_frame = pl.DataFrame(
            {"x": [0.5]},
        ).lazy()
        assert_frame_equal(
            model.predict(test_frame).collect(),
            loaded_model.predict(test_frame).collect(),
        )

    def test_save_load_model_with_transforms(self) -> None:
        n = 100
        data = DataFrame(
            pl.DataFrame(
                {
                    "x": pl.arange(0, n, eager=True).cast(pl.Float32) / n,
                    "y": pl.arange(n, 0, -1, eager=True).cast(pl.Float32) / n,
                },
            ),
        )

        train_env, _test_env = TrainTestSplit(ratio=0.8, shuffle=False).split(
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

        model_bytes = BytesIO()
        model.save(model_bytes)
        model_bytes.seek(0)

        loaded_model = Model.load(model_bytes)

        test_frame = pl.DataFrame(
            {"x": [0.5]},
        ).lazy()
        assert_frame_equal(
            model.predict(test_frame).collect(),
            loaded_model.predict(test_frame).collect(),
        )

    def test_save_load_selector_backed_hydra_model(self) -> None:
        selector = HybridDecisionTreeLearner(
            SelectorFeatureConfig(state_columns=("x",)),
            random_state=7,
            max_depth=2,
        ).learn_from_traces(
            [pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]})],
        )
        model = HyDRAModel(
            [ConstantModel(10.0), ConstantModel(20.0)],
            input_features=["x"],
            output_features=["y"],
            selector=selector,
        )

        model_bytes = BytesIO()
        model.save(model_bytes)
        model_bytes.seek(0)

        loaded_model = Model.load(model_bytes)
        assert isinstance(loaded_model, HyDRAModel)

        test_frame = pl.DataFrame({"x": [0.05, 1.05]}).lazy()
        assert_frame_equal(
            loaded_model.predict(test_frame).collect(),
            pl.DataFrame({"y": [10.0, 20.0]}),
        )


if __name__ == "__main__":
    unittest.main()
