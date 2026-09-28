"""HyDRA flow-selector learning, inspection, and evaluation API."""

from .config import SelectorFeatureConfig
from .evaluation import (
    SelectorEvaluationReport,
    evaluate_selector_autoregressive,
    evaluate_selector_oracle,
)
from .inspection import (
    SelectorFlowInspection,
    SelectorInspection,
    SelectorLeafInspection,
    SelectorNodeInspection,
)
from .learner import HybridDecisionTreeLearner
from .model import FlowPredictionResult, HybridDecisionTreeModel
from .runtime import StatefulHybridDecisionTreeSelector

__all__ = [
    "FlowPredictionResult",
    "HybridDecisionTreeLearner",
    "HybridDecisionTreeModel",
    "SelectorEvaluationReport",
    "SelectorFeatureConfig",
    "SelectorFlowInspection",
    "SelectorInspection",
    "SelectorLeafInspection",
    "SelectorNodeInspection",
    "StatefulHybridDecisionTreeSelector",
    "evaluate_selector_autoregressive",
    "evaluate_selector_oracle",
]
