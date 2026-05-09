from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from flowcean.core import Model


@runtime_checkable
class SupportsFlowSummary(Protocol):
    def flow_summary(self) -> str: ...


def summarize_flow_model(model: Model | None) -> str:
    if model is None:
        return "<unresolved flow>"

    if isinstance(model, SupportsFlowSummary):
        return model.flow_summary()

    return type(model).__name__


@dataclass(frozen=True)
class SelectorNodeInspection:
    node_id: int
    sample_count: int
    impurity: float
    is_leaf: bool
    predicted_mode_id: int
    weighted_class_support: dict[int, float]
    feature_index: int | None = None
    feature_name: str | None = None
    threshold: float | None = None
    left_child_id: int | None = None
    right_child_id: int | None = None


@dataclass(frozen=True)
class SelectorLeafInspection:
    node_id: int
    mode_id: int
    sample_count: int
    weighted_class_support: dict[int, float]
    flow_summary: str


@dataclass(frozen=True)
class SelectorModeInspection:
    mode_id: int
    weighted_support: float
    flow_summary: str


@dataclass(frozen=True)
class SelectorInspection:
    feature_columns: tuple[str, ...]
    classes: tuple[int, ...]
    max_depth: int
    n_leaves: int
    nodes: tuple[SelectorNodeInspection, ...]
    leaves: tuple[SelectorLeafInspection, ...]
    modes: tuple[SelectorModeInspection, ...]
