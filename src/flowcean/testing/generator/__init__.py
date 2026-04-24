from typing import TYPE_CHECKING, Any

from .combination_generator import CombinationGenerator
from .generator import TestcaseGenerator
from .stochastic_generator import StochasticGenerator

if TYPE_CHECKING:
    from .ddti_generator import DDTIGenerator

__all__ = [
    "CombinationGenerator",
    "DDTIGenerator",
    "StochasticGenerator",
    "TestcaseGenerator",
]


def __getattr__(name: str) -> Any:
    if name == "DDTIGenerator":
        from .ddti_generator import DDTIGenerator

        return DDTIGenerator
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
