import logging

import polars as pl

from agenc.core import Transform

logger = logging.getLogger(__name__)


class Select(Transform):
    """Selects a subset of features from the data.

    Args:
        features (list[str]): The features to select.
    """

    def __init__(self, features: list[str]) -> None:
        self.features = features

    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        logger.debug(f"selecting features {self.features}")
        return data.select(self.features)
