import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.polars import ScaleToRange, Standardize


def test_standardize_inverse_matches_statistics_by_feature_name() -> None:
    transform = Standardize(
        mean={"x": 10.0, "y": 20.0}, std={"y": 2.0, "x": 5.0}
    )
    data = pl.LazyFrame({"x": [15.0], "y": [22.0]})
    assert_frame_equal(transform.inverse()(transform(data)), data)
    with pytest.raises(ValueError, match="matching feature names"):
        Standardize(mean={"x": 10.0}, std={"y": 2.0}).inverse()


def test_explicit_scale_limits_map_endpoints_and_roundtrip() -> None:
    transform = ScaleToRange.from_limits(
        {"x": (10.0, 20.0)}, lower_range=2.0, upper_range=4.0
    )
    data = pl.LazyFrame({"x": [10.0, 15.0, 20.0]})
    assert_frame_equal(transform(data), pl.LazyFrame({"x": [2.0, 3.0, 4.0]}))
    assert_frame_equal(transform.inverse()(transform(data)), data)


@pytest.mark.parametrize(
    "bounds", [(1.0, 1.0), (float("nan"), 1.0), (0.0, float("inf"))]
)
def test_scale_rejects_impossible_output_ranges(
    bounds: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="finite and distinct"):
        ScaleToRange(lower_range=bounds[0], upper_range=bounds[1])


@pytest.mark.parametrize("scale", [0.0, float("nan"), float("inf")])
def test_scale_rejects_noninvertible_public_coefficients(scale: float) -> None:
    transform = ScaleToRange()
    transform.m = {"x": scale}
    transform.b = {"x": 1.0}
    with pytest.raises(ValueError, match="finite nonzero"):
        transform.inverse()
    with pytest.raises(ValueError, match="finite nonzero"):
        transform(pl.LazyFrame({"x": [1.0]}))
