import polars as pl
import pytest

from flowcean.hydra.selector.config import SelectorFeatureConfig
from flowcean.hydra.selector.features import (
    build_selector_dataset,
    build_selector_inference_frame,
    validate_global_mode_labels,
)


def test_build_selector_dataset_adds_current_and_history_features() -> None:
    traces = [
        pl.DataFrame(
            {
                "x": [0.0, 1.0, 2.0],
                "u": [10.0, 11.0, 12.0],
                "dx": [0.5, 0.5, 0.5],
                "mode": [0, 0, 1],
            },
        ),
    ]
    config = SelectorFeatureConfig(
        state_columns=("x",),
        input_columns=("u",),
        derivative_columns=("dx",),
        state_history=1,
        input_history=1,
        derivative_history=1,
    )

    dataset = build_selector_dataset(traces, config)

    assert dataset.feature_columns == (
        "x",
        "u",
        "dx",
        "x_t_minus_1",
        "u_t_minus_1",
        "dx_t_minus_1",
    )
    assert dataset.features.to_dict(as_series=False) == {
        "x": [1.0, 2.0],
        "u": [11.0, 12.0],
        "dx": [0.5, 0.5],
        "x_t_minus_1": [0.0, 1.0],
        "u_t_minus_1": [10.0, 11.0],
        "dx_t_minus_1": [0.5, 0.5],
    }
    assert dataset.labels.to_list() == [0, 1]
    assert dataset.dropped_rows_by_trace == {0: 1}


def test_build_selector_dataset_uses_previous_mode_history_per_trace() -> None:
    traces = [
        pl.DataFrame({"x": [0.0, 1.0], "mode": [0, 1]}),
        pl.DataFrame({"x": [5.0, 6.0], "mode": [2, 2]}),
    ]
    config = SelectorFeatureConfig(state_columns=("x",), mode_history=1)

    dataset = build_selector_dataset(traces, config)

    assert dataset.features.to_dict(as_series=False) == {
        "x": [1.0, 6.0],
        "mode_t_minus_1": [0, 2],
    }
    assert dataset.labels.to_list() == [1, 2]
    assert dataset.row_metadata.to_dict(as_series=False) == {
        "trace_id": [0, 1],
        "row_index": [1, 1],
    }


def test_build_selector_dataset_rejects_missing_columns() -> None:
    traces = [pl.DataFrame({"x": [0.0], "mode": [0]})]
    config = SelectorFeatureConfig(
        state_columns=("x",),
        derivative_columns=("dx",),
    )

    with pytest.raises(ValueError, match="missing required selector columns"):
        build_selector_dataset(traces, config)


def test_build_selector_dataset_orders_history_columns() -> None:
    traces = [pl.DataFrame({"x": [0.0, 1.0, 2.0], "mode": [0, 0, 1]})]
    config = SelectorFeatureConfig(state_columns=("x",), state_history=2)

    dataset = build_selector_dataset(traces, config)

    assert dataset.feature_columns == ("x", "x_t_minus_1", "x_t_minus_2")
    assert dataset.features.to_dict(as_series=False) == {
        "x": [2.0],
        "x_t_minus_1": [1.0],
        "x_t_minus_2": [0.0],
    }


def test_build_selector_inference_frame_does_not_require_mode_column() -> None:
    frame = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "u": [10.0, 11.0, 12.0],
            "dx": [0.5, 0.6, 0.7],
        },
    )
    config = SelectorFeatureConfig(
        state_columns=("x",),
        input_columns=("u",),
        derivative_columns=("dx",),
        state_history=1,
        input_history=1,
        derivative_history=1,
    )

    inference = build_selector_inference_frame(frame, config)

    assert inference.features.to_dict(as_series=False) == {
        "x": [1.0, 2.0],
        "u": [11.0, 12.0],
        "dx": [0.6, 0.7],
        "x_t_minus_1": [0.0, 1.0],
        "u_t_minus_1": [10.0, 11.0],
        "dx_t_minus_1": [0.5, 0.6],
    }
    assert inference.row_metadata.to_dict(as_series=False) == {
        "row_index": [1, 2],
    }


def test_selector_feature_config_rejects_missing_history_columns() -> None:
    config = SelectorFeatureConfig(state_history=1)

    with pytest.raises(
        ValueError,
        match="state_history requires at least one configured state column",
    ):
        config.validate()


def test_selector_feature_config_rejects_duplicate_raw_feature_names() -> None:
    config = SelectorFeatureConfig(
        state_columns=("x",),
        input_columns=("x",),
    )

    with pytest.raises(ValueError, match="duplicate selector columns"):
        config.validate()


def test_validate_global_mode_labels_accepts_multiple_modes() -> None:
    traces = [
        pl.DataFrame({"mode": [0, 0]}),
        pl.DataFrame({"mode": [1, 2]}),
    ]

    validate_global_mode_labels(traces)


def test_validate_global_mode_labels_accepts_single_mode() -> None:
    traces = [pl.DataFrame({"mode": [1, 1]})]

    validate_global_mode_labels(traces)


def test_build_selector_inference_frame_rejects_mode_history() -> None:
    frame = pl.DataFrame({"x": [0.0, 1.0]})
    config = SelectorFeatureConfig(state_columns=("x",), mode_history=1)

    with pytest.raises(
        NotImplementedError,
        match=(
            "batch selector inference does not support previous-mode features"
        ),
    ):
        build_selector_inference_frame(frame, config)
