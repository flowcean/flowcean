from flowcean._optional import raise_for_missing_optional_dependency

try:
    from .learner import RiverLearner, RiverModel
except ModuleNotFoundError as error:
    raise_for_missing_optional_dependency(
        error,
        extra="river",
        module="flowcean.river",
        missing_dependencies={"river"},
    )

__all__ = [
    "RiverLearner",
    "RiverModel",
]
