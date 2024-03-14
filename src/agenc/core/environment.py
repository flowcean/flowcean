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
        return TransformedEnvironment(self, transform)


class NotLoadedError(Exception):
    """Environment not loaded.

    This exception is raised when trying to access the data of an environment
    that has not been loaded yet.
    """


class OfflineDataLoader(Environment):
    """Base class for data loaders.

    An offline data loader loads data from e.g., a file, a database, etc.
    Data can be retrieved once using the `get_data` method.
    """

    @abstractmethod
    def get_data(self) -> pl.DataFrame:
        """Get data from the environment.

        Returns:
            The loaded dataset.
        """

    def as_stream(self, batch_size: int = 1) -> StreamingOfflineData:
        return StreamingOfflineData(self, batch_size)


Request = TypeVar("Request")


class ActiveOnlineDataLoader(Environment, Generic[Request]):
    """Base class for active online data loaders.

    An active online data loader retrieves data given a request.
    """

    @abstractmethod
    def request_data(self, request: Request) -> pl.DataFrame:
        """Get data from the environment.

        Args:
            request: The request

        Returns:
            The data matching the request.
        """


class PassiveOnlineDataLoader(Environment):
    """Base class for passive online data loaders."""

    @abstractmethod
    def get_next_data(self) -> Generator[pl.DataFrame, None, None]:
        """Get the next data.

        Returns:
            The next data.
        """


class StreamingOfflineData(PassiveOnlineDataLoader):
    environment: OfflineDataLoader
    batch_size: int
    index: int
    data: pl.DataFrame | None

    def __init__(
        self,
        environment: OfflineDataLoader,
        batch_size: int = 1,
    ) -> None:
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


Parameters = TypeVar("Parameters")


class Simulation(Environment, Generic[Parameters]):
    """Base class for simulations."""

    @abstractmethod
    def start(self, parameters: Parameters) -> None:
        """Start the simulation.

        Args:
            parameters: The parameters to start the simulation.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop the simulation."""

    @abstractmethod
    def next_simulated_data(self) -> pl.DataFrame:
        """Get the next data from the simulation.

        Returns:
            The next data from the simulation.
        """


Action = TypeVar("Action")


class ControlledSimulation(
    Simulation[Parameters],
    Generic[Action, Parameters],
):
    """Base class for controlled simulations.

    A simulation that can be controlled by a learner.
    """

    @abstractmethod
    def act(self, action: Action) -> None:
        """Act in the simulation.

        Args:
            action: The action to take.
        """


T_Environment = TypeVar("T_Environment", bound=Environment)
T_OfflineDataLoader = TypeVar("T_OfflineDataLoader", bound=OfflineDataLoader)
T_PassiveOnlineDataLoader = TypeVar(
    "T_PassiveOnlineDataLoader",
    bound=PassiveOnlineDataLoader,
)


class TransformedEnvironment(
    OfflineDataLoader,
    PassiveOnlineDataLoader,
    Generic[T_Environment],
):
    """DataLoader that applies a transform to the data."""

    environment: T_Environment
    transform: Transform

    def __init__(
        self,
        environment: T_Environment,
        transform: Transform,
    ) -> None:
        self.environment = environment
        self.transform = transform

    @override
    def load(self) -> Self:
        self.environment.load()
        return self

    @override
    def get_data(
        self: TransformedEnvironment[T_OfflineDataLoader],
    ) -> pl.DataFrame:
        data = self.environment.get_data()
        return self.transform.transform(data)

    @override
    def get_next_data(
        self: TransformedEnvironment[T_PassiveOnlineDataLoader],
    ) -> Generator[pl.DataFrame, None, None]:
        return self.environment.get_next_data()
