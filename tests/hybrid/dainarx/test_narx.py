import numpy as np
import polars as pl
import pytest

from flowcean.hybrid.dainarx import fit_narx
from flowcean.hybrid.dainarx._narx import solve


def test_independent_multioutput_nonlinear_lags_and_current_input():
    rng = np.random.default_rng(73)
    coefficients = np.array(
        [
            [0.04, 0.3, -0.2, 0.1, 0.05, 0.7, 0.2],
            [-0.03, 0.1, 0.4, -0.15, 0.03, -0.4, -0.1],
        ]
    )
    frames = []
    for initial in [-2, 3, 1]:
        x = np.empty((90, 2))
        x[:2] = rng.normal(size=(2, 2)) + initial
        u = rng.normal(size=(90, 1))
        for t in range(2, len(x)):
            linear = np.r_[x[t - 1], x[t - 2], u[t], 1]
            x[t] = (
                coefficients[:, 0] * x[t - 2] ** 2
                + coefficients[:, 1:] @ linear
            )
        frames.append(
            pl.DataFrame(
                {
                    "t": np.arange(len(x)) * 0.01,
                    "a": x[:, 0],
                    "b": x[:, 1],
                    "u": u[:, 0],
                    "metadata": ["ignored"] * len(x),
                }
            )
        )
    model = fit_narx(
        frames,
        input_features=["u"],
        output_features=["a", "b"],
        time_column="t",
        order=2,
        nonlinear_terms=[lambda history, current: history[0] ** 2],
    )
    np.testing.assert_allclose(model.coefficients, coefficients, atol=1e-12)
    assert model.rmse < 1e-12
    np.testing.assert_allclose(
        model.predict(x[-3:-1], u[-1]), x[-1], atol=1e-12
    )
    assert frames[0]["metadata"][0] == "ignored"


def test_rank_deficient_fit_measures_residuals_and_max_output_rmse():
    rows = np.ones((10, 2, 3))
    targets = np.column_stack((np.arange(10.0), np.arange(10.0) * 4))
    _, error = solve(rows, targets)
    assert error == pytest.approx(np.std(targets[:, 1]))
    assert error > 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None])
def test_nonfinite_selected_data_is_rejected(bad):
    frame = pl.DataFrame(
        {"t": [0.0, 1.0, 2.0, 3.0], "x": [0.0, 1.0, bad, 3.0]}
    )
    with pytest.raises(ValueError, match="finite"):
        fit_narx(
            [frame],
            input_features=[],
            output_features=["x"],
            time_column="t",
            order=1,
        )


@pytest.mark.parametrize(
    "term", [lambda h, u: [np.nan], lambda h, u: [1.0, 2.0]]
)
def test_invalid_nonlinear_result_is_rejected(term):
    frame = pl.DataFrame({"t": np.arange(10), "x": np.arange(10)})
    with pytest.raises(ValueError, match=r"finite|shape"):
        fit_narx(
            [frame],
            input_features=[],
            output_features=["x"],
            time_column="t",
            order=1,
            nonlinear_terms=[term],
        )


@pytest.mark.parametrize(
    "time", [[0.0, 1.0, 1.0, 3.0], [0.0, 1.0, 2.1, 3.0], [0.0, 2.0, 4.0, 6.0]]
)
def test_invalid_or_inconsistent_time_grid(time):
    first = pl.DataFrame(
        {"t": [0.0, 1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0, 3.0]}
    )
    second = first.with_columns(pl.Series("t", time))
    with pytest.raises(ValueError, match=r"timestamp|Timestamp"):
        fit_narx(
            [first, second],
            input_features=[],
            output_features=["x"],
            time_column="t",
            order=1,
        )


def test_too_few_targets_and_non_numeric_columns_are_rejected():
    frame = pl.DataFrame({"t": [0.0, 1.0, 2.0], "x": [0.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="Insufficient"):
        fit_narx(
            [frame],
            input_features=[],
            output_features=["x"],
            time_column="t",
            order=2,
        )
    with pytest.raises(ValueError, match="numeric"):
        fit_narx(
            [frame.with_columns(pl.lit("bad").alias("x"))],
            input_features=[],
            output_features=["x"],
            time_column="t",
            order=1,
        )
