import logging
from collections.abc import Iterable

import polars as pl
from polars._typing import PolarsDataType
from typing_extensions import override

from flowcean.core import Transform

logger = logging.getLogger(__name__)


class Cast(Transform):
    """Cast features to a different datatype.

    This transform allows to change the datatype of features in a DataFrame.
    To cast all features to the same datatype, provide a single type as the
    `target_type` argument e.g.

    ```python
    transform = Cast(pl.Float64)
    ```

    By specifying the `features` keyword argument, only the selected features
    will be cast e.g.

    ```python
    transform = Cast(pl.Float64, features=["feature_a"])
    ```

    Lastly, to cast features to different datatypes, provide a dictionary with
    feature names as keys and target types as values e.g.

    ```python
    transform = Cast(
        {
            "feature_a": pl.Boolean,
            "feature_b": pl.Float64,
        },
    )
    ```
    """

    def __init__(
        self,
        target_type: PolarsDataType | dict[str, PolarsDataType],
        *,
        features: Iterable[str] | None = None,
    ) -> None:
        """Initializes the Cast transform.

        Args:
            target_type: Type to which the features will be cast.
                If a single type is provided, all features or those provided in
                the `features` keyword argument will be cast to that specific
                type. To cast features to different types, provide a dictionary
                with feature names as keys and target types as values.
            features: The features to cast. If `None` all
                features will be cast. This is the default behaviour.
        """
        self.target_type = target_type
        self.features = features

    @override
    def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
        if isinstance(self.target_type, dict):
            for feature, target_type in self.target_type.items():
                data = data.with_columns(
                    pl.col(feature).cast(target_type),
                )
            return data
        return data.with_columns(
            (
                pl.all() if self.features is None else pl.col(self.features)
            ).cast(self.target_type),
        )
