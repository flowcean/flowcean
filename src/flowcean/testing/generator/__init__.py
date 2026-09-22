from typing import TYPE_CHECKING, Any

from .combination_generator import CombinationGenerator
from .generator import TestcaseGenerator
from .stochastic_generator import StochasticGenerator

if TYPE_CHECKING:
    from .ddti_generator import DDTIGenerator
    from .ddtig import ModelHandler, TestPipeline

__all__ = [
    "CombinationGenerator",
    "DDTIGenerator",
    "ModelHandler",
    "StochasticGenerator",
    "TestPipeline",
    "TestcaseGenerator",
]


def __getattr__(name: str) -> Any:
    if name == "DDTIGenerator":
        from .ddti_generator import DDTIGenerator

        return DDTIGenerator
    if name in {"ModelHandler", "TestPipeline"}:
        from . import ddtig

        return getattr(ddtig, name)
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
