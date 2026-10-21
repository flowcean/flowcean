import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sklearn.linear_model import LinearRegression as SKLearnLinearRegression

from flowcean.hybrid.hydra import HyDRALearner, TraceSegment
from flowcean.sklearn import LinearRegression, SciKitModel


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize(
    "target_shape", ["1d", "single-column", "multi-output"]
)
def test_predictions_preserve_scalar_columns_and_rows(
    target_shape: str, batch_size: int
) -> None:
    inputs = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    outputs = pl.DataFrame({"y": [1.0, 3.0, 5.0, 7.0]})
    if target_shape == "multi-output":
        outputs = outputs.with_columns(z=pl.Series([4.0, 3.0, 2.0, 1.0]))
    targets = (
        outputs.to_series().to_numpy()
        if target_shape == "1d"
        else outputs.to_numpy()
    )
    estimator = SKLearnLinearRegression().fit(inputs, targets)
    model = SciKitModel(estimator, output_names=outputs.columns)

    predicted = model.predict(inputs.head(batch_size).lazy()).collect()

    assert_frame_equal(predicted, outputs.head(batch_size))


def test_linear_regression_interoperates_with_hydra() -> None:
    trace = pl.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "dx": [1.0, 3.0, 5.0, 7.0]}
    )
    result = HyDRALearner(
        LinearRegression, threshold=1e-10, start_width=2
    ).learn([trace], input_features=["x"], output_features=["dx"])

    assert len(result.flows) == 1
    assert result.flows[0].segments == (TraceSegment(0, 0, trace.height),)
    assert result.to_flow_ids()[0].tolist() == [0] * trace.height
    assert_frame_equal(
        result.flows[0].model.predict(trace.select("x").lazy()).collect(),
        trace.select("dx"),
    )
