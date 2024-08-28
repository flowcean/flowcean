from __future__ import annotations

import polars as pl

from flowcean.core.environment.actable import Actable
from flowcean.core.environment.observable import Observable
from flowcean.core.environment.stepable import Stepable

from .base import Environment


class ActiveEnvironment(
    Environment,
    Observable[pl.DataFrame],
    Stepable,
    Actable[pl.DataFrame],
):
    """Base class for active environments.

    An active environment loads data in an interactive way, e.g., from a
    simulation or real system. The environment requires actions to be taken to
    advance. Data can be retrieved by observing the environment.
    """
