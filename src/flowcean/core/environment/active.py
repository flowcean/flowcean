from __future__ import annotations

from typing_extensions import override

from flowcean.core import (
    Actable,
    NotSupportedError,
    Stepable,
    TransformedObservable,
)
from flowcean.core.data import Data


class ActiveEnvironment(
    TransformedObservable,
    Stepable,
    Actable[Data],
):
    """Base class for active environments.

    An active environment loads data in an interactive way, e.g., from a
    simulation or real system. The environment requires actions to be taken to
    advance. Data can be retrieved by observing the environment.
    """

    def __init__(self) -> None:
        """Initialize the active environment."""
        super().__init__()

    @override
    def hash(self) -> bytes:
        raise NotSupportedError
