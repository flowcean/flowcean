import logging
from collections.abc import Collection

import polars as pl
from polars._typing import ColumnNameOrSelector
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Unnest(Transform):
    """Decompose struct columns into separate columns for each field.

    Example:
    ```python
    data_frame = pl.Series(
        "c",
        [
            {"a": 1, "t": 1},
            {"a": 4, "t": 2},
            {"a": 7, "t": 3},
            {"a": 10, "t": 4},
            {"a": 15, "t": 5},
        ],
    ).to_frame()
    ```
    The transformed_data will be:
    ```python
    pl.DataFrame(
        {
            "a": [1, 4, 7, 10, 15],
            "t": [1, 2, 3, 4, 5],
        },
    )
    ```
    .
    """

    def __init__(
        self,
        features: ColumnNameOrSelector | Collection[ColumnNameOrSelector],
    ) -> None:
        """Initializes the Unnest transform.

        Args:
            features: The features to unnest. Treats the selection as a
                parameter to polars `unnest` method. You can use regular
                expressions by wrapping the argument by ^ and $.
        """
        self.features = features

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        logger.debug("selecting features %s", self.features)
        return data.unnest(self.features)
