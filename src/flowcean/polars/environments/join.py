import hashlib
from collections.abc import Iterable

import polars as pl
from typing_extensions import override

from flowcean.core import OfflineEnvironment


class JoinedOfflineEnvironment(OfflineEnvironment):
    """Environment that joins multiple offline environments.

    Attributes:
        environments: The offline environments to join.
    """

    environments: Iterable[OfflineEnvironment]

    def __init__(self, environments: Iterable[OfflineEnvironment]) -> None:
        """Initialize the joined offline environment.

        Args:
            environments: The offline environments to join.
        """
        self.environments = environments
        super().__init__()

    @override
    def hash(self) -> bytes:
        # The assumption here is, that if all individual environments are the
        # same, then the joined environment is the same as well.
        hasher = hashlib.sha256()
        for environment in self.environments:
            hasher.update(environment.hash())
        return hasher.digest()

    @override
    def _observe(self) -> pl.LazyFrame:
        return pl.concat(
            (environment.observe() for environment in self.environments),
            how="horizontal",
        )
