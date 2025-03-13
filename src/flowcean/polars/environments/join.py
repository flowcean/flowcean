import hashlib
from collections.abc import Iterable

import polars as pl
from typing_extensions import override

from flowcean.core.environment.offline import OfflineEnvironment


class JoinedOfflineEnvironment(OfflineEnvironment):
    """Environment that joins multiple offline environments.

    Attributes:
        environments: The offline environments to join.
    """

    environments: Iterable[OfflineEnvironment]
    _hash: bytes | None = None

    def __init__(self, environments: Iterable[OfflineEnvironment]) -> None:
        """Initialize the joined offline environment.

        Args:
            environments: The offline environments to join.
        """
        self.environments = environments
        super().__init__()

    @override
    def _observe(self) -> pl.LazyFrame:
        return pl.concat(
            (environment.observe() for environment in self.environments),
            how="horizontal",
        )

    @override
    def hash(self) -> bytes:
        if self._hash is None:
            hasher = hashlib.sha256()
            for env in self.environments:
                hasher.update(env.hash())
            self._hash = hasher.digest()
        return self._hash
