import logging
import re

import polars as pl

from agenc.core import Transform

logger = logging.getLogger(__name__)

REGEX_INDICATION_SEQUENCE = "/"


class Select(Transform):
    """Selects a subset of features from the data.

    Args:
        features (list[str]): The features to select.
                              Treats the feature name as a regular expression,
                              when it starts with '/' and select all matching
                              features.
    """

    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.feature_patterns = [
            re.compile(f"^{pattern}$")
            for pattern in [
                (
                    name[len(REGEX_INDICATION_SEQUENCE) :].strip()
                    if name.startswith(REGEX_INDICATION_SEQUENCE)
                    else re.escape(name)
                )
                for name in features
            ]
        ]

    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        logger.debug(f"selecting features {self.features}")
        return data.select(
            [
                column_name
                for column_name in data.columns
                if any(
                    [
                        pattern.match(column_name)
                        for pattern in self.feature_patterns
                    ]
                )
            ]
        )
