"""Curated HyDRA hybrid-system identification API."""

from . import selector
from .callbacks import LogCallback, PlotCallback
from .learner import HyDRALearner
from .model import HyDRAModel
from .schema import HyDRATraceSchema
from .selector import (
    FlowPredictionResult,
    HybridDecisionTreeLearner,
    HybridDecisionTreeModel,
    SelectorEvaluationReport,
    SelectorFeatureConfig,
    SelectorFlowInspection,
    SelectorInspection,
    SelectorLeafInspection,
    SelectorNodeInspection,
    StatefulHybridDecisionTreeSelector,
    evaluate_selector_autoregressive,
    evaluate_selector_oracle,
)
from .simulation import StateTraceComparison, compare_state_traces

__all__ = (
    "FlowPredictionResult",
    "HyDRALearner",
    "HyDRAModel",
    "HyDRATraceSchema",
    "HybridDecisionTreeLearner",
    "HybridDecisionTreeModel",
    "LogCallback",
    "PlotCallback",
    "SelectorEvaluationReport",
    "SelectorFeatureConfig",
    "SelectorFlowInspection",
    "SelectorInspection",
    "SelectorLeafInspection",
    "SelectorNodeInspection",
    "StateTraceComparison",
    "StatefulHybridDecisionTreeSelector",
    "compare_state_traces",
    "evaluate_selector_autoregressive",
    "evaluate_selector_oracle",
    "selector",
)
