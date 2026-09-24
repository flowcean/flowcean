"""Validation for externally driven benchmark inputs."""

import numpy as np

from ..hybrid_system import InputStream


def _input_vector(
    t: float, input_stream: InputStream, *, size: int, label: str
) -> np.ndarray:
    """Read exactly ``size`` finite input components at physical time ``t``."""
    # Do not catch exceptions from the stream: in particular the simulator's
    # missing-input error and caller-defined errors must remain distinguishable.
    raw = input_stream(t)
    message = (
        f"{label} input must contain {size} finite real numeric component(s) "
        f"as a vector of shape ({size},)"
    )
    try:
        array = np.asarray(raw)
        if np.iscomplexobj(array) or (
            array.dtype == object
            and any(np.iscomplexobj(value) for value in array.flat)
        ):
            raise ValueError(message)
        values = np.asarray(array, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(message)
    return values
