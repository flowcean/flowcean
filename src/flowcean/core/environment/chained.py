from collections.abc import Iterable, Iterator

from typing_extensions import override

from flowcean.core.data import Data
from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.environment.offline import OfflineEnvironment
from flowcean.core.environment.stepable import Finished


class ChainedOfflineEnvironments(IncrementalEnvironment):
    """Chained offline environments.

    This environment chains multiple offline environments together. The
    environment will first observe the data from the first environment and then
    the data from the other environments.
    """

    _environments: Iterator[OfflineEnvironment]
    _element: OfflineEnvironment

    def __init__(self, environments: Iterable[OfflineEnvironment]) -> None:
        """Initialize the chained offline environments.

        Args:
            environments: The offline environments to chain.
        """
        self._environments = iter(environments)
        self._element = next(self._environments)
        super().__init__()

    @override
    def _observe(self) -> Data:
        return self._element.observe()

    @override
    def step(self) -> None:
        try:
            self._element = next(self._environments)
        except StopIteration:
            raise Finished from StopIteration
