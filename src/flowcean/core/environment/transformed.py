from __future__ import annotations

from typing import TYPE_CHECKING, override

from flowcean.core.environment.observable import Observable

if TYPE_CHECKING:
    import polars as pl

    from flowcean.core.transform import Transform
