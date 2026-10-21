from io import BytesIO
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.core import Lambda
from flowcean.xgboost.learner import (
    XGBoostClassifierLearner,
    XGBoostRegressorLearner,
)
from flowcean.xgboost.model import (
    XGBoostClassifierModel,
)


@pytest.mark.parametrize("classification", [True, False])
def test_serialization_preserves_predictions_and_live_callbacks(
    classification: bool,
) -> None:
    inputs = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    targets = pl.LazyFrame(
        {"y": [0, 1, 0, 1] if classification else [1, 4, 9, 16]}
    )
    learner = (
        XGBoostClassifierLearner(objective="binary:logistic")
        if classification
        else XGBoostRegressorLearner(objective="reg:squarederror")
    )
    model = learner.learn(inputs, targets)
    estimator = (
        model.classifier
        if isinstance(model, XGBoostClassifierModel)
        else model.regressor
    )
    callbacks = estimator.callbacks
    blob = BytesIO()
    model.save(blob)
    blob.seek(0)
    loaded = type(model).load(blob)
    assert estimator.callbacks is callbacks
    assert_frame_equal(
        model.predict(inputs).collect(), loaded.predict(inputs).collect()
    )


def test_multiclass_predictions_preserve_native_xgboost_labels() -> None:
    inputs = pl.LazyFrame({"x": list(range(12))})
    targets = pl.LazyFrame({"y": [0] * 4 + [1] * 4 + [2] * 4})
    model = XGBoostClassifierLearner(
        objective="multi:softprob",
        n_estimators=5,
        max_depth=2,
    ).learn(inputs, targets)
    assert model.threshold is None
    np.testing.assert_array_equal(
        model.predict(inputs).collect().to_series().to_numpy(),
        model.classifier.predict(inputs.collect().to_numpy()),
    )
    with pytest.raises(ValueError, match="one target and two classes"):
        model.predict_proba(inputs)
    model.threshold = 0.5
    with pytest.raises(ValueError, match="one target and two classes"):
        model.predict(inputs)


def test_binary_threshold_preprocessing_and_probability_boundary() -> None:
    inputs = pl.LazyFrame({"x": [0.0, 0.0]})
    model = XGBoostClassifierLearner(
        objective="binary:logistic",
        n_estimators=1,
        threshold=0.5,
    ).learn(inputs, pl.LazyFrame({"y": [0, 1]}))
    calls = 0

    def preprocess(data: pl.LazyFrame) -> pl.LazyFrame:
        nonlocal calls
        calls += 1
        return data

    model.pre_transform = Lambda(preprocess)
    model.post_transform = Lambda(
        lambda data: data.with_columns(pl.col("y") + 10)
    )
    assert model.predict(inputs).collect()["y"].to_list() == [11, 11]
    assert calls == 1
    assert model.predict_proba(inputs).collect()["y"].to_list() == [0.5, 0.5]
    assert calls == 2
    model.threshold = 0.51
    assert model.predict(inputs).collect()["y"].to_list() == [10, 10]
    with (
        patch.object(
            model.classifier, "predict_proba", return_value=np.zeros((2, 1))
        ),
        pytest.raises(ValueError, match="probabilities must have shape"),
    ):
        model.predict_proba(inputs)
    model.threshold = None
    with (
        patch.object(model.classifier, "predict", return_value=np.zeros(1)),
        pytest.raises(ValueError, match="Prediction shape"),
    ):
        model.predict(inputs)
