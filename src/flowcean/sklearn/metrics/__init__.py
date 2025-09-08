from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from flowcean.core import Data, Reportable


class MultiOutputMixin:
    """Mixin to handle sklearn-style multioutput regression metrics."""

    def __init__(
        self,
        multioutput: Literal[
            "raw_values",
            "uniform_average",
            "variance_weighted",
        ] = "raw_values",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.multioutput = multioutput

    def _finalize_result(
        self,
        error: Any,
        true: Data,
    ) -> Reportable | dict[str, Reportable]:
        if isinstance(error, np.ndarray):
            return dict(zip(true.columns, error, strict=True))
        return error
