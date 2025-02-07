from collections.abc import Sequence

import polars as pl
from typing_extensions import override

from flowcean.core.environment.offline import OfflineEnvironment


class JoinedEnvironments(OfflineEnvironment):
    """Joined offline environments.

    This environment joins multiple offline environments together.
    """

    _environments: Sequence[OfflineEnvironment]

    def __init__(self, *environments: OfflineEnvironment) -> None:
        """Initialize the joined offline environments.

        Args:
            environments: The offline environments to join.
        """
        self._environments = environments
        super().__init__()

    @override
    def _observe(self) -> pl.DataFrame:
        return pl.concat(
            (environment.observe() for environment in self._environments),
            how="horizontal",
        )
