from __future__ import annotations

from typing import Any

from flowcean.core.environment.actable import Actable
from flowcean.core.environment.observable import (
    TransformedObservable,
)
from flowcean.core.environment.stepable import Stepable


class ActiveEnvironment(
    TransformedObservable,
    Stepable,
    Actable[Any],
):
    """Base class for active environments.

    An active environment loads data in an interactive way, e.g., from a
    simulation or real system. The environment requires actions to be taken to
    advance. Data can be retrieved by observing the environment.
    """

    def __init__(self) -> None:
        """Initialize the active environment."""
        super().__init__()
