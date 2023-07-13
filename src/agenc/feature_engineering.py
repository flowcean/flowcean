import logging
from typing import List

import numpy as np
import polars as pl

LOG = logging.getLogger(__name__)


def multiply(
    data: pl.DataFrame,
    new_column_name: str,
    first_factor_name: str,
    second_factor_name: str,
) -> pl.DataFrame:
    """Multiplies two columns of the dataframe and stores the result as
    a new column.

    """

    return data.with_columns(
        (pl.col(first_factor_name) * pl.col(second_factor_name)).alias(
            new_column_name
        )
    )
