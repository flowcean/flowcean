"""Training and runtime use the same learned-flow history features."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.hybrid.hydra.selector import (
    FlowPredictionResult,
    HybridDecisionTreeLearner,
    SelectorFeatureConfig,
    StatefulHybridDecisionTreeSelector,
)
from flowcean.hybrid.hydra.selector.features import build_selector_dataset


def test_runtime_history_matches_batch_training_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "u": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "dx": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            "flow_id": [2, 7, 2, 7, 2, 7],
        }
    )
    config = SelectorFeatureConfig(
        state_features=("x",),
        input_features=("u",),
        derivative_features=("dx",),
        state_history=2,
        input_history=1,
        derivative_history=1,
        flow_history=2,
    )
    dataset = build_selector_dataset([frame], config)
    model = HybridDecisionTreeLearner(
        config, random_state=0
    ).learn_from_traces([frame])
    runtime = StatefulHybridDecisionTreeSelector(model, seed_flows=[2, 7])
    engineered_frames: list[pl.DataFrame] = []
    predict_details = model.predict_details

    def capture(features: pl.DataFrame) -> list[FlowPredictionResult]:
        engineered_frames.append(features)
        return predict_details(features)

    monkeypatch.setattr(model, "predict_details", capture)
    results = [runtime.predict(row) for row in frame.iter_rows(named=True)]

    assert [result.ready for result in results] == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]
    assert [result.flow_id for result in results[2:]] == [2, 7, 2, 7]
    assert dataset.features.columns[-2:] == [
        "flow_t_minus_1",
        "flow_t_minus_2",
    ]
    assert dataset.features["flow_t_minus_1"].to_list() == [7, 2, 7, 2]
    assert dataset.features["flow_t_minus_2"].to_list() == [2, 7, 2, 7]
    assert_frame_equal(pl.concat(engineered_frames), dataset.features)
