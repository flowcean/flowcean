from dataclasses import dataclass

import polars as pl

from flowcean.hydra.selector.config import SelectorFeatureConfig


@dataclass(frozen=True)
class SelectorDataset:
    features: pl.DataFrame
    labels: pl.Series
    feature_columns: tuple[str, ...]
    row_metadata: pl.DataFrame
    dropped_rows_by_trace: dict[int, int]


@dataclass(frozen=True)
class SelectorInferenceFrame:
    features: pl.DataFrame
    row_metadata: pl.DataFrame


def build_selector_dataset(
    traces: list[pl.DataFrame],
    config: SelectorFeatureConfig,
) -> SelectorDataset:
    config.validate()
    if not traces or not any(trace.height > 0 for trace in traces):
        message = "selector dataset requires at least one labeled trace"
        raise ValueError(message)

    feature_columns = _feature_columns(config)
    feature_frames: list[pl.DataFrame] = []
    label_series: list[pl.Series] = []
    metadata_frames: list[pl.DataFrame] = []
    dropped_rows_by_trace: dict[int, int] = {}

    for trace_id, trace in enumerate(traces):
        _validate_trace(trace, config)
        projected = _project_trace(trace, config)
        dropped_rows = min(config.max_history, projected.height)
        dropped_rows_by_trace[trace_id] = dropped_rows
        usable = projected.slice(dropped_rows)
        if usable.height == 0:
            continue

        feature_frames.append(usable.select(feature_columns))
        label_series.append(usable["mode"])
        metadata_frames.append(
            pl.DataFrame(
                {
                    "trace_id": [trace_id] * usable.height,
                    "row_index": usable["row_index"].to_list(),
                },
            ),
        )

    if not feature_frames:
        message = "selector dataset has zero usable samples after warmup"
        raise ValueError(message)

    return SelectorDataset(
        features=pl.concat(feature_frames, how="vertical"),
        labels=pl.concat(label_series, how="vertical"),
        feature_columns=feature_columns,
        row_metadata=pl.concat(metadata_frames, how="vertical"),
        dropped_rows_by_trace=dropped_rows_by_trace,
    )


def validate_global_mode_labels(traces: list[pl.DataFrame]) -> None:
    mode_ids: set[int] = set()
    for trace in traces:
        if "mode" not in trace.columns:
            message = "selector traces must include a mode column"
            raise ValueError(message)
        if trace["mode"].null_count() > 0:
            message = "selector mode labels must not contain nulls"
            raise ValueError(message)
        mode_ids.update(trace["mode"].to_list())

    if not mode_ids:
        message = "selector mode labels must contain at least one mode ID"
        raise ValueError(message)


def build_selector_inference_frame(
    frame: pl.DataFrame,
    config: SelectorFeatureConfig,
) -> SelectorInferenceFrame:
    config.validate()
    if config.mode_history:
        msg = (
            "batch selector inference does not support previous-mode "
            "features; use the stateful selector runtime."
        )
        raise NotImplementedError(
            msg,
        )

    missing_columns = set(config.required_columns()) - set(frame.columns)
    if missing_columns:
        message = "missing required selector columns"
        msg = f"{message}: {sorted(missing_columns)}"
        raise ValueError(msg)

    projected = _project_trace(
        frame,
        config,
        include_mode_history=False,
        include_mode=False,
    )
    dropped_rows = min(config.max_history, projected.height)
    usable = projected.slice(dropped_rows)

    return SelectorInferenceFrame(
        features=usable.select(_feature_columns(config)),
        row_metadata=usable.select("row_index"),
    )


def _feature_columns(config: SelectorFeatureConfig) -> tuple[str, ...]:
    columns = [
        *config.state_features,
        *config.input_features,
        *config.derivative_features,
    ]
    columns.extend(
        _history_columns(config.state_features, config.state_history),
    )
    columns.extend(
        _history_columns(config.input_features, config.input_history),
    )
    columns.extend(
        _history_columns(
            config.derivative_features,
            config.derivative_history,
        ),
    )
    columns.extend(
        f"mode_t_minus_{step}" for step in range(1, config.mode_history + 1)
    )
    return tuple(columns)


def _validate_trace(
    trace: pl.DataFrame,
    config: SelectorFeatureConfig,
) -> None:
    missing_columns = {"mode", *config.required_columns()} - set(trace.columns)
    if missing_columns:
        message = "missing required selector columns"
        msg = f"{message}: {sorted(missing_columns)}"
        raise ValueError(msg)
    if trace["mode"].null_count() > 0:
        message = "selector mode labels must not contain nulls"
        raise ValueError(message)


def _history_columns(columns: tuple[str, ...], history: int) -> list[str]:
    return [
        f"{column}_t_minus_{step}"
        for step in range(1, history + 1)
        for column in columns
    ]


def _project_trace(
    trace: pl.DataFrame,
    config: SelectorFeatureConfig,
    *,
    include_mode_history: bool = True,
    include_mode: bool = True,
) -> pl.DataFrame:
    expressions: list[pl.Expr] = [
        *[pl.col(column) for column in config.state_features],
        *[pl.col(column) for column in config.input_features],
        *[pl.col(column) for column in config.derivative_features],
    ]

    expressions.extend(
        _history_expressions(config.state_features, config.state_history),
    )
    expressions.extend(
        _history_expressions(config.input_features, config.input_history),
    )
    expressions.extend(
        _history_expressions(
            config.derivative_features,
            config.derivative_history,
        ),
    )

    if include_mode_history:
        expressions.extend(
            pl.col("mode").shift(step).alias(f"mode_t_minus_{step}")
            for step in range(1, config.mode_history + 1)
        )

    if include_mode:
        expressions.append(pl.col("mode"))

    expressions.append(
        pl.int_range(0, trace.height, eager=False).alias("row_index"),
    )
    return trace.select(expressions)


def _history_expressions(
    columns: tuple[str, ...],
    history: int,
) -> list[pl.Expr]:
    return [
        pl.col(column).shift(step).alias(f"{column}_t_minus_{step}")
        for step in range(1, history + 1)
        for column in columns
    ]
