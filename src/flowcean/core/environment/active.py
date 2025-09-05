from __future__ import annotations

from typing import Protocol, runtime_checkable

from flowcean.core.environment.actable import Actable
from flowcean.core.environment.base import Environment
from flowcean.core.environment.incremental import Stepable


@runtime_checkable
class ActiveEnvironment(Environment, Stepable, Actable, Protocol):
    """Base class for active environments.

    An active environment loads data in an interactive way, e.g., from a
    simulation or real system. The environment requires actions to be taken to
    advance. Data can be retrieved by observing the environment.
    """
