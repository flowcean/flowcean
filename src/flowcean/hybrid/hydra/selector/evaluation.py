from dataclasses import dataclass

import polars as pl
from sklearn.metrics import confusion_matrix

from .features import build_selector_dataset
from .model import HybridDecisionTreeModel
from .runtime import StatefulHybridDecisionTreeSelector


@dataclass(frozen=True)
class SelectorEvaluationReport:
    accuracy: float
    samples: int
    confusion_matrix: pl.DataFrame


def evaluate_selector_oracle(
    model: HybridDecisionTreeModel,
    traces: list[pl.DataFrame],
) -> SelectorEvaluationReport:
    dataset = build_selector_dataset(traces, model.feature_config)
    predictions = (
        model.predict(dataset.features).collect()["flow_id"].to_list()
    )
    return _report(dataset.labels.to_list(), predictions)


def evaluate_selector_autoregressive(
    model: HybridDecisionTreeModel,
    traces: list[pl.DataFrame],
    seed_flows: tuple[int, ...] | list[int] = (),
) -> SelectorEvaluationReport:
    _validate_autoregressive_traces(
        traces,
        model.feature_config.required_columns(),
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    required_flow_history = model.feature_config.flow_history

    for trace in traces:
        resolved_seed_flows = tuple(int(flow_id) for flow_id in seed_flows)
        bootstrapped_prefix = 0
        if len(resolved_seed_flows) < required_flow_history:
            if resolved_seed_flows:
                message = (
                    "explicit seed history is shorter than configured "
                    "flow_history"
                )
                raise ValueError(message)
            resolved_seed_flows = _bootstrap_seed_flows(
                trace,
                required_flow_history,
            )
            bootstrapped_prefix = len(resolved_seed_flows)

        runtime = StatefulHybridDecisionTreeSelector(
            model,
            seed_flows=resolved_seed_flows,
        )
        raw_columns = model.feature_config.required_columns()
        for row_index, row in enumerate(trace.iter_rows(named=True)):
            raw_sample = {column: row[column] for column in raw_columns}
            result = runtime.predict(raw_sample)
            if (
                row_index < bootstrapped_prefix
                or not result.ready
                or result.flow_id is None
            ):
                continue
            y_true.append(int(row["flow_id"]))
            y_pred.append(result.flow_id)

    return _report(y_true, y_pred)


def _validate_autoregressive_traces(
    traces: list[pl.DataFrame],
    required_columns: tuple[str, ...],
) -> None:
    if not traces or not any(trace.height > 0 for trace in traces):
        message = "selector evaluation requires at least one labeled trace"
        raise ValueError(message)

    for trace in traces:
        missing_columns = {"flow_id", *required_columns} - set(trace.columns)
        if "flow_id" in missing_columns:
            message = "selector traces must include a flow_id column"
            raise ValueError(message)
        if missing_columns:
            message = "missing required selector columns"
            msg = f"{message}: {sorted(missing_columns)}"
            raise ValueError(msg)
        if trace["flow_id"].null_count() > 0:
            message = "selector flow IDs must not contain nulls"
            raise ValueError(message)


def _bootstrap_seed_flows(
    trace: pl.DataFrame,
    flow_history: int,
) -> tuple[int, ...]:
    if flow_history == 0:
        return ()

    bootstrap_flows = tuple(
        int(flow_id)
        for flow_id in trace["flow_id"].head(flow_history).to_list()
    )
    if len(bootstrap_flows) < flow_history:
        message = (
            "selector traces must contain enough flow IDs to bootstrap history"
        )
        raise ValueError(message)
    return bootstrap_flows


def _report(y_true: list[int], y_pred: list[int]) -> SelectorEvaluationReport:
    if not y_true:
        message = "selector evaluation produced no scored samples"
        raise ValueError(message)

    labels = sorted({*y_true, *y_pred})
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    scored = len(y_true)
    correct = sum(
        int(actual == predicted)
        for actual, predicted in zip(y_true, y_pred, strict=True)
    )

    return SelectorEvaluationReport(
        accuracy=correct / scored,
        samples=scored,
        confusion_matrix=pl.DataFrame(
            {
                "actual_flow_id": labels,
                **{
                    f"predicted_{label}": matrix[:, index].tolist()
                    for index, label in enumerate(labels)
                },
            },
        ),
    )
