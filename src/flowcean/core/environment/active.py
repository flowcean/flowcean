from __future__ import annotations

from typing import Protocol, runtime_checkable

from flowcean.core.environment.actable import Actable
from flowcean.core.environment.base import Environment
from flowcean.core.environment.incremental import Stepable


@runtime_checkable
class ActiveEnvironment(Environment, Stepable, Actable, Protocol):
    """An environment supporting active learning through interaction.

    An active environment loads data interactively from simulations or real
    systems. The learner influences the environment by selecting actions,
    which the environment responds to with observations and rewards. This
    supports active learning strategies where the learner explores the
    environment to optimize its behavior.
    """
