from flowcean._optional import raise_for_missing_optional_dependency

__all__ = [
    "XGBoostClassifierLearner",
    "XGBoostClassifierModel",
    "XGBoostRegressorLearner",
    "XGBoostRegressorModel",
]

try:
    from .learner import XGBoostClassifierLearner, XGBoostRegressorLearner
    from .model import XGBoostClassifierModel, XGBoostRegressorModel
except ModuleNotFoundError as error:
    raise_for_missing_optional_dependency(
        error,
        extra="xgboost",
        module="flowcean.xgboost",
        missing_dependencies={"xgboost"},
    )
