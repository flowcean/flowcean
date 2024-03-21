from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from typing_extensions import Self, override

if TYPE_CHECKING:
    from collections.abc import Generator

    import polars as pl

    from .transform import Transform


class Environment(ABC):
    """Base class for environments.

    An environment provides data to a learner. It can be a data loader, a
    simulation, etc.
    """

    @abstractmethod
    def load(self) -> Self:
        """Load the environment."""

    def with_transform(
        self,
        transform: Transform,
    ) -> TransformedEnvironment[Self]:
        """Attach a transform to the environment.

        Args:
            transform: The transform to apply to the data of the environment.

        Returns:
            The transformed environment.
        """
        return TransformedEnvironment(self, transform)


class NotLoadedError(Exception):
    """Environment not loaded.

    This exception is raised when trying to access the data of an environment
    that has not been loaded yet.
    """


class OfflineEnvironment(Environment):
    """Base class for offline environments.

    An offline environment loads data in an non-interactive way, e.g., from a
    file, a database, etc. Data can only be retrieved once.
    """

    @abstractmethod
    def get_data(self) -> pl.DataFrame:
        """Get data from the environment.

        Returns:
            The loaded dataset.
        """

    def as_stream(self, batch_size: int = 1) -> StreamingOfflineData:
        """Get a streaming interface to the data of the environment.

        Args:
            batch_size: The number of rows to yield at each iteration.

        Returns:
            A streaming offline data.
        """
        return StreamingOfflineData(self, batch_size)


class PassiveOnlineEnvironment(Environment):
    """Base class for passive online environments.

    A passive online environment loads data in an interactive way, e.g., from a
    stream, a sensor, etc. Data can be retrieved multiple times.
    """

    @abstractmethod
    def get_next_data(self) -> Generator[pl.DataFrame, None, None]:
        """Get the next data from the environment.

        Yields:
            The next batch of data.
        """


Action = TypeVar("Action")
Observation = TypeVar("Observation")


class ActiveOnlineEnvironment(Environment, Generic[Action, Observation]):
    """Base class for active online environments.

    An active online environment loads data in an interactive way, e.g., from a
    stream, a sensor, etc. Data can be retrieved multiple times. It also allows
    to act on the environment, e.g., to control a robot, a simulation, etc.
    """

    @abstractmethod
    def act(self, action: Action) -> None:
        """Act on the environment.

        Args:
            action: The action to perform.
        """

    @abstractmethod
    def step(self) -> None:
        """Advance the environment by one step."""

    @abstractmethod
    def observe(self) -> Observation:
        """Observe the environment.

        Returns:
            The observation of the environment.
        """


class StreamingOfflineData(PassiveOnlineEnvironment):
    """Streaming offline data.

    This class wraps an offline environment and provides a streaming interface
    to its data.
    """

    environment: OfflineEnvironment
    batch_size: int
    index: int
    data: pl.DataFrame | None

    def __init__(
        self,
        environment: OfflineEnvironment,
        batch_size: int = 1,
    ) -> None:
        """Initialize a streaming offline data.

        Args:
            environment: The offline environment to wrap.
            batch_size: The number of rows to yield at each iteration.
        """
        self.environment = environment
        self.batch_size = batch_size
        self.index = 0
        self.data = None

    @override
    def load(self) -> Self:
        self.environment.load()
        self.data = self.environment.get_data()
        return self

    @override
    def get_next_data(self) -> Generator[pl.DataFrame, None, None]:
        if self.data is None:
            raise NotLoadedError
        for i in range(0, len(self.data), self.batch_size):
            yield self.data.slice(i, self.batch_size)


T_Environment = TypeVar("T_Environment", bound=Environment)
T_OfflineEnvironment = TypeVar(
    "T_OfflineEnvironment",
    bound=OfflineEnvironment,
)
T_PassiveOnlineEnvironment = TypeVar(
    "T_PassiveOnlineEnvironment",
    bound=PassiveOnlineEnvironment,
)


class TransformedEnvironment(
    OfflineEnvironment,
    PassiveOnlineEnvironment,
    Generic[T_Environment],
):
    """Environment with a transform.

    This class wraps an environment and a transform. It applies the transform
    to the data of the environment before returning it.
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
    def get_next_data(
        self: TransformedEnvironment[T_PassiveOnlineEnvironment],
    ) -> Generator[pl.DataFrame, None, None]:
        return self.environment.get_next_data()
