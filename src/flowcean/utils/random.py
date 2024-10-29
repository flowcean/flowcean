import logging
import random

import numpy as np
import polars as pl

rng = np.random.default_rng()

logger = logging.getLogger(__name__)


def initialize_random(seed: int) -> None:
    """Initialize the random number generator with the given seed.

    Args:
        seed: The seed to initialize the random number generator with.
    """
    global rng  # noqa: PLW0603
    rng = np.random.default_rng(seed)
    random.seed(get_seed())
    pl.set_random_seed(get_seed())

    np.random.seed(get_seed())  # noqa: NPY002
    try:
        import torch

        torch.manual_seed(get_seed())
    except ModuleNotFoundError:
        pass
    logger.info("Initialized random number generator with seed %d", seed)


def get_seed() -> int:
    """Generate a random seed.

    Returns:
        A random seed.
    """
    return int(rng.integers(2**32 - 1))
