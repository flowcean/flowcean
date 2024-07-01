from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Self, TypeVar, override

from .base import Environment
from .incremental import IncrementalEnvironment
from .offline import OfflineEnvironment

if TYPE_CHECKING:
    from collections.abc import Iterator

    import polars as pl

    from flowcean.core import Transform

T_Environment = TypeVar("T_Environment", bound=Environment)
T_OfflineEnvironment = TypeVar(
    "T_OfflineEnvironment",
    bound=OfflineEnvironment,
)
T_IncrementalEnvironment = TypeVar(
    "T_IncrementalEnvironment",
    bound=IncrementalEnvironment,
)


class TransformedEnvironment(
    OfflineEnvironment,
    IncrementalEnvironment,
    Generic[T_Environment],
):
    """Environment with a transform.

    This class wraps an environment and a transform. It applies the transform
    to any data of the environment before returning it.

    Attributes:
        environment: The environment to wrap.
        transform: The transform to apply to the data of the environment.
    """

    environment: T_Environment
    transform: Transform

    def __init__(
        self,
        environment: T_Environment,
        transform: Transform,
    ) -> None:
        """Initialize a transformed environment.

        Args:
            environment: The environment to wrap.
            transform: The transform to apply to the data of the environment.
        """
        self.environment = environment
        self.transform = transform

    @override
    def load(self) -> Self:
        self.environment.load()
        return self

    @override
    def get_data(
        self: TransformedEnvironment[T_OfflineEnvironment],
    ) -> pl.DataFrame:
        data = self.environment.get_data()
        return self.transform.transform(data)

    @override
    def __iter__(
        self: TransformedEnvironment[T_IncrementalEnvironment],
    ) -> Iterator[pl.DataFrame]:
        return self.environment.__iter__()
