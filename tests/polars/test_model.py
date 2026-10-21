from typing import assert_type, override

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.core import Lambda, Predictor, evaluate_offline
from flowcean.polars import DataFrame, PolarsModel


class EchoModel(PolarsModel):
    @override
    def _predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        return input_features


def test_eager_and_lazy_public_prediction_normalize_before_transforms() -> (
    None
):
    model = EchoModel()
    calls = 0

    def preprocess(frame: pl.LazyFrame) -> pl.LazyFrame:
        nonlocal calls
        assert isinstance(frame, pl.LazyFrame)
        calls += 1
        return frame.select(pl.all() * 2)

    model.pre_transform = Lambda(preprocess)
    model.post_transform = Lambda(lambda frame: frame.select(pl.all() + 1))
    data = pl.DataFrame({"x": [2, 3]})
    expected = pl.LazyFrame({"x": [5, 7]})
    assert_frame_equal(
        assert_type(model.predict(data), pl.LazyFrame), expected
    )
    assert_frame_equal(assert_type(model(data), pl.LazyFrame), expected)
    assert_frame_equal(model.predict(data.lazy()), expected)
    assert_frame_equal(model(data.lazy()), expected)
    assert calls == 4


def test_read_only_consumers_accept_structural_contravariant_predictors() -> (
    None
):
    class ConstantPredictor:
        name = "constant"

        def predict(self, input_features: object) -> pl.LazyFrame:
            return pl.LazyFrame({"x": [1]})

    predictor: Predictor[pl.LazyFrame, pl.LazyFrame] = ConstantPredictor()
    report = evaluate_offline(
        predictor,
        DataFrame(pl.LazyFrame({"x": [1]})),
        ["x"],
        ["x"],
        [],
    )
    assert list(report) == ["constant"]
