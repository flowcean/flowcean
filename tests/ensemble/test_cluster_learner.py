import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.ensemble import ClusterLearner
from flowcean.polars import DummyLearner


@pytest.mark.parametrize("remove_cluster_feature", [False, True])
@pytest.mark.parametrize("target", ["y", "cluster"])
def test_cluster_routing_preserves_requested_target_schema(
    remove_cluster_feature: bool,
    target: str,
) -> None:
    inputs = pl.DataFrame({"x": [10, 20], "cluster": [0, 1]})
    outputs = pl.LazyFrame({target: [2.0, 4.0]})
    model = ClusterLearner(
        "cluster",
        DummyLearner(),
        remove_cluster_feature=remove_cluster_feature,
    ).learn(inputs.lazy(), outputs)
    expected = pl.LazyFrame({target: [0.0, 0.0]})
    assert_frame_equal(model(inputs), expected)
    assert_frame_equal(model(inputs.lazy()), expected)
