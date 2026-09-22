"""Classification and regression metrics backed by scikit-learn."""

from ._multioutput import MultiOutputMixin
from .classification import (
    Accuracy,
    ClassificationReport,
    FBetaScore,
    PrecisionScore,
    Recall,
)
from .regression import (
    MaxError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    R2Score,
)

__all__ = (
    "Accuracy",
    "ClassificationReport",
    "FBetaScore",
    "MaxError",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanSquaredError",
    "MultiOutputMixin",
    "PrecisionScore",
    "R2Score",
    "Recall",
)
