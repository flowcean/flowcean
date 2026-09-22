"""Run-scoped numerical runtime thread limits."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

from threadpoolctl import threadpool_limits

if TYPE_CHECKING:
    from collections.abc import Iterator

NUMERICAL_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
DYNAMIC_THREAD_VARIABLES = (
    "OMP_DYNAMIC",
    "MKL_DYNAMIC",
)
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "MKL_DYNAMIC",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def validate_numerical_threads(value: int) -> None:
    """Require a positive built-in integer thread limit."""
    if type(value) is not int or value < 1:
        message = "numerical_threads must be a positive integer"
        raise ValueError(message)


@contextmanager
def numerical_thread_limit(count: int) -> Iterator[None]:
    """Temporarily cap loaded libraries and libraries imported by children."""
    validate_numerical_threads(count)
    original = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(NUMERICAL_THREAD_VARIABLES, str(count)))
    os.environ.update(dict.fromkeys(DYNAMIC_THREAD_VARIABLES, "FALSE"))
    try:
        with threadpool_limits(limits=count, user_api=None):
            yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
