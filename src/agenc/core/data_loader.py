from abc import ABC, abstractmethod

import polars as pl


class DataLoader(ABC):
    """Base class for data loaders."""

    @abstractmethod
    def load(self) -> pl.DataFrame:
        """Load the data.

        Returns:
            The loaded dataset.
        """
