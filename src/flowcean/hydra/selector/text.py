from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

    from flowcean.hydra.selector.inspection import SelectorInspection
    from flowcean.hydra.selector.model import ModePredictionResult


def _format_value(value: object) -> str:
    return str(value)


def _format_class_support(class_support: dict[int, float]) -> str:
    return ", ".join(
        f"{mode_id}={_format_value(support)}"
        for mode_id, support in class_support.items()
    )


def _format_flow_summary(flow_summary: str) -> str:
    if "\n" not in flow_summary:
        return f"flow={flow_summary}"

    indented_lines = "\n".join(
        f"  {line}" for line in flow_summary.splitlines() or [""]
    )
    return f"flow:\n{indented_lines}"


def render_leaf_summary_text(inspection: SelectorInspection) -> str:
    return "\n".join(
        (
            f"leaf_id={leaf.node_id} mode={leaf.mode_id} "
            f"raw_samples={leaf.sample_count} "
            "weighted_class_support="
            f"[{_format_class_support(leaf.weighted_class_support)}] "
            f"{_format_flow_summary(leaf.flow_summary)}"
        )
        for leaf in inspection.leaves
    )


def render_mode_summary_text(inspection: SelectorInspection) -> str:
    return "\n".join(
        (
            f"mode={mode.mode_id} "
            f"weighted_support={_format_value(mode.weighted_support)} "
            f"{_format_flow_summary(mode.flow_summary)}"
        )
        for mode in inspection.modes
    )


def render_summary_text(inspection: SelectorInspection) -> str:
    feature_columns = ", ".join(inspection.feature_columns)
    return "\n".join(
        [
            f"feature columns: {feature_columns}",
            f"modes: {len(inspection.modes)}",
            f"leaves: {inspection.n_leaves}",
            f"max depth: {inspection.max_depth}",
        ],
    )


def render_prediction_debug_text(
    input_rows: pl.DataFrame,
    predictions: list[ModePredictionResult],
    feature_columns: tuple[str, ...],
) -> str:
    rows = input_rows.select(feature_columns).to_dicts()
    return "\n".join(
        (
            f"row {row_index}: "
            "inputs: "
            f"{_format_input_values(row, feature_columns)}; "
            f"mode={prediction.mode_id}; "
            f"leaf_id={prediction.leaf_id}; "
            "probabilities: "
            f"[{_format_class_support(prediction.probabilities)}]"
        )
        for row_index, (row, prediction) in enumerate(
            zip(rows, predictions, strict=True),
        )
    )


def _format_input_values(
    row: dict[str, object],
    feature_columns: tuple[str, ...],
) -> str:
    return ", ".join(
        f"{column}={_format_value(row[column])}" for column in feature_columns
    )
