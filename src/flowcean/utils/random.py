import logging
import random

import numpy as np
import polars as pl
import torch

rng = np.random.default_rng()

logger = logging.getLogger(__name__)


def initialize_random(seed: int) -> None:
    global rng  # noqa: PLW0603
    rng = np.random.default_rng(seed)
    random.seed(seed)
    pl.set_random_seed(seed)
    torch.manual_seed(0)
    np.random.seed(seed)  # noqa: NPY002
    logger.info("Initialized random number generator with seed %d", seed)


def get_seed() -> int:
    return rng.integers(2**32 - 1)
