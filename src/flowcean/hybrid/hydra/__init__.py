"""Curated HyDRA hybrid-system identification API."""

from . import selector
from .callbacks import HyDRACallback
from .learner import (
    HyDRAIdentificationError,
    HyDRALearner,
    LearnedFlow,
    LearnedFlows,
    TraceSegment,
)
from .model import HyDRAModel
from .plotting import PlotCallback
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
    "HyDRACallback",
    "HyDRAIdentificationError",
    "HyDRALearner",
    "HyDRAModel",
    "HyDRATraceSchema",
    "HybridDecisionTreeLearner",
    "HybridDecisionTreeModel",
    "LearnedFlow",
    "LearnedFlows",
    "PlotCallback",
    "SelectorEvaluationReport",
    "SelectorFeatureConfig",
    "SelectorFlowInspection",
    "SelectorInspection",
    "SelectorLeafInspection",
    "SelectorNodeInspection",
    "StateTraceComparison",
    "StatefulHybridDecisionTreeSelector",
    "TraceSegment",
    "compare_state_traces",
    "evaluate_selector_autoregressive",
    "evaluate_selector_oracle",
    "selector",
)
