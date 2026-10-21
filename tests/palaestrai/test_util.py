"""Numeric interface conversion preserves the PalaestrAI space contract."""

import numpy as np
import pytest

pytest.importorskip("harl")
pytest.importorskip("palaestrai")

from flowcean.core import ActiveInterface, Observation
from flowcean.palaestrai.util import (
    convert_to_interface,
    convert_to_sensor_information,
)


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_sensor_interface_round_trip_preserves_numeric_space(dtype):
    original = ActiveInterface(
        uid="sensor",
        value=np.array([0, 1], dtype=dtype),
        value_min=0,
        value_max=2,
        shape=(2,),
        dtype=dtype,
    )
    information = convert_to_sensor_information(
        Observation(sensors=[original], rewards=[])
    )
    (restored,) = convert_to_interface(information)
    assert restored.uid == original.uid
    assert restored.dtype is dtype
    assert restored.shape == original.shape
    np.testing.assert_array_equal(restored.value, original.value)
    np.testing.assert_array_equal(restored.value_min, [0, 0])
    np.testing.assert_array_equal(restored.value_max, [2, 2])
