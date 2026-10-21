from io import BytesIO
from typing import Any, assert_type

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray
from polars.testing import assert_frame_equal
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from flowcean.core import Lambda, learn_offline
from flowcean.polars import DataFrame, Select
from flowcean.sklearn import AdaBoost, SciKitClassifierModel, SciKitModel


@pytest.mark.parametrize(
    "labels", [["a", "b", "c"], [-4, 7, 12], ["no", "yes"], [-4, 7]]
)
def test_native_predictions_preserve_estimator_labels(
    labels: list[str] | list[int],
) -> None:
    inputs = pl.DataFrame({"x": list(range(len(labels)))})
    estimator = DecisionTreeClassifier(random_state=0).fit(inputs, labels)
    model = SciKitClassifierModel(estimator, output_names=["target"])
    assert model.threshold is None
    for data in (inputs, inputs.lazy()):
        np.testing.assert_array_equal(
            model.predict(data).collect().to_series().to_numpy(),
            estimator.predict(inputs),
        )
    if len(labels) > 2:
        with pytest.raises(ValueError, match="one target and two classes"):
            model.predict_proba(inputs)


@pytest.mark.parametrize("labels", [["absent", "present"], [-4, 7]])
def test_threshold_equality_maps_to_second_original_class(
    labels: list[str] | list[int],
) -> None:
    inputs = pl.DataFrame({"x": [0, 1]})
    estimator = DummyClassifier(strategy="prior").fit(inputs, labels)
    model = SciKitClassifierModel(
        estimator, output_names=["target"], threshold=0.5
    )
    assert (
        model.predict(inputs).collect().to_series().to_list()
        == [labels[1]] * 2
    )
    model.threshold = 0.51
    assert (
        model.predict(inputs).collect().to_series().to_list()
        == [labels[0]] * 2
    )
    assert model.predict_proba(inputs).collect().to_dict(as_series=False) == {
        "target": [0.5, 0.5]
    }
    model.threshold = None
    blob = BytesIO()
    model.save(blob)
    blob.seek(0)
    loaded = SciKitClassifierModel.load(blob)
    assert loaded.threshold is None
    assert_frame_equal(
        model.predict(inputs).collect(), loaded.predict(inputs).collect()
    )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_threshold_rejected_at_construction_and_prediction(
    threshold: float,
) -> None:
    inputs = pl.DataFrame({"x": [0, 1]})
    estimator = DummyClassifier().fit(inputs, [0, 1])
    with pytest.raises(ValueError, match="threshold"):
        SciKitClassifierModel(
            estimator, output_names=["y"], threshold=threshold
        )
    model = SciKitClassifierModel(estimator, output_names=["y"])
    model.threshold = threshold
    with pytest.raises(ValueError, match="threshold"):
        model.predict(inputs)


class ArrayEstimator:
    def __init__(
        self,
        predictions: NDArray[Any],
        probabilities: NDArray[Any],
        classes: object = None,
    ) -> None:
        self.predictions = predictions
        self.probabilities = probabilities
        self.classes_ = np.array([0, 1]) if classes is None else classes

    def predict(self, X: Any) -> NDArray[Any]:
        return self.predictions

    def predict_proba(self, X: Any) -> NDArray[Any]:
        return self.probabilities


@pytest.mark.parametrize("shape", [(1,), (2, 2), (2, 1, 1)])
def test_malformed_prediction_shape_is_rejected(
    shape: tuple[int, ...],
) -> None:
    estimator = ArrayEstimator(np.zeros(shape), np.zeros((2, 2)))
    model = SciKitModel(estimator, output_names=["y"])
    with pytest.raises(ValueError, match="Prediction shape"):
        model.predict(pl.LazyFrame({"x": [0, 1]}))


@pytest.mark.parametrize(
    "probabilities",
    [
        np.zeros((1, 2)),
        np.zeros((2, 1)),
        np.zeros((2, 3)),
        np.zeros((2, 2, 1)),
        np.array([[0.5, np.nan], [0.5, 0.5]]),
        np.array([[0.0, 1.1], [0.5, 0.5]]),
    ],
)
def test_malformed_binary_probabilities_are_rejected(
    probabilities: NDArray[Any],
) -> None:
    estimator = ArrayEstimator(np.zeros(2), probabilities)
    model = SciKitClassifierModel(estimator, output_names=["y"])
    with pytest.raises(ValueError, match=r"[Pp]robabilities"):
        model.predict_proba(pl.LazyFrame({"x": [0, 1]}))


@pytest.mark.parametrize(
    ("classes", "names"),
    [
        (np.array([0]), ["y"]),
        (np.array([0, 1, 2]), ["y"]),
        ([np.array([0, 1]), np.array([0, 1])], ["y", "z"]),
        (np.array([0, 1]), ["y", "z"]),
    ],
)
def test_binary_contract_rejects_nonbinary_and_multioutput(
    classes: object, names: list[str]
) -> None:
    estimator = ArrayEstimator(np.zeros(2), np.zeros((2, 2)), classes)
    with pytest.raises(ValueError, match="one target and two classes"):
        SciKitClassifierModel(estimator, output_names=names, threshold=0.5)
    model = SciKitClassifierModel(estimator, output_names=names)
    with pytest.raises(ValueError, match="one target and two classes"):
        model.predict_proba(pl.LazyFrame({"x": [0, 1]}))


def test_preprocessing_runs_once_and_probabilities_skip_label_postprocessing() -> (
    None
):
    inputs = pl.DataFrame({"x": [0, 1]})
    estimator = DummyClassifier(strategy="prior").fit(inputs, [0, 1])
    model = SciKitClassifierModel(estimator, output_names=["y"], threshold=0.5)
    calls = 0

    def preprocess(data: pl.LazyFrame) -> pl.LazyFrame:
        nonlocal calls
        assert isinstance(data, pl.LazyFrame)
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


def test_adaboost_preserves_string_targets() -> None:
    inputs = pl.LazyFrame({"x": [0, 1, 2, 3]})
    targets = pl.LazyFrame({"label": ["low", "low", "high", "high"]})
    model = learn_offline(
        DataFrame(pl.concat([inputs, targets], how="horizontal")),
        AdaBoost(random_state=1),
        ["x"],
        ["label"],
        input_transform=Select(pl.col("x") * 2),
    )
    assert_type(model, SciKitClassifierModel)
    assert model.threshold is None
    assert_frame_equal(model.predict(inputs).collect(), targets.collect())
    assert_frame_equal(model(inputs.collect()).collect(), targets.collect())
