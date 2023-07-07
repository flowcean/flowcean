import logging
from typing import List

import numpy as np
import polars as pl

LOG = logging.getLogger(__name__)


def mult(
    data: pl.DataFrame,
    new_col: str,
    *cols: List[str],
) -> pl.DataFrame:
    if len(cols) < 2:
        LOG.error(f"Not enough arguments for multiplication: {cols}")
        return data
    elif len(cols) > 2:
        LOG.error(f"More than two arguments are not yet supported!")
        return data

    data.with_columns((pl.col(cols[0]) * pl.col(cols[1])).alias(new_col))

    return data
