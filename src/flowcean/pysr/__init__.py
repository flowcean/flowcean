from flowcean._optional import raise_for_missing_optional_dependency

try:
    from .learner import PySRLearner, PySRModel
except ModuleNotFoundError as error:
    raise_for_missing_optional_dependency(
        error,
        extra="pysr",
        module="flowcean.pysr",
        missing_dependencies={"pysr"},
    )

__all__ = [
    "PySRLearner",
    "PySRModel",
]
