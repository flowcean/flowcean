import unittest
from collections.abc import Sequence
from typing import override

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray
from polars.testing import assert_frame_equal

from flowcean.environments.ode_environment import (
    IntegrationError,
    OdeEnvironment,
    OdeSystem,
    State,
)


class SimpleState(State):
    def __init__(self, x: float) -> None:
        self.x = x

    def as_numpy(self) -> NDArray[np.float64]:
        return np.array([self.x])

    @classmethod
    def from_numpy(cls, state: NDArray[np.float64]) -> "SimpleState":
        return SimpleState(state[0])


class SimpleSystem(OdeSystem[SimpleState]):
    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return -1 / 2 * state


class TimeDependentSystem(OdeSystem[SimpleState]):
    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return np.array([t / state[0]])


class NonIntegrableSystem(OdeSystem[SimpleState]):
    @override
    def flow(
        self,
        t: float,
        state: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return -np.sqrt(state[0])


def map_to_dataframe(
    ts: Sequence[float], states: Sequence[SimpleState]
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "t": ts,
            "y_0": [state.x for state in states],
        },
    )


class TestOdeEnvironment(unittest.TestCase):
    def test_length(self) -> None:
        environment = OdeEnvironment(
            SimpleSystem(t=0.0, state=SimpleState(x=1.0)),
            map_to_dataframe=map_to_dataframe,
        )
        loaded_data = environment.collect(11)
        assert len(loaded_data) == 21

    def test_simple_ode(self) -> None:
        ts = [0.0, 0.114893, 1.0, 1.114908, 2.0, 2.114932, 3.0, 3.114973, 4.0]
        data = pl.DataFrame(
            {
                "t": ts,
                "y_0": [np.exp(-1 / 2 * t) for t in ts],
            },
        )

        environment = OdeEnvironment(
            SimpleSystem(t=0.0, state=SimpleState(x=1.0)),
            map_to_dataframe=map_to_dataframe,
        )
        loaded_data = environment.collect(5).observe()

        assert_frame_equal(
            data,
            loaded_data,
            check_exact=False,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_time_dependent_ode(self) -> None:
        ts = [
            0.0,
            0.0001,
            0.0011,
            0.0111,
            0.1111,
            1.0,
            1.114885,
            2.0,
            2.120122,
            3.0,
        ]
        data = pl.DataFrame(
            {
                "t": ts,
                "y_0": [np.sqrt(np.power(t, 2) + 1) for t in ts],
            }
        )

        environment = OdeEnvironment(
            TimeDependentSystem(t=0.0, state=SimpleState(x=1.0)),
            map_to_dataframe=map_to_dataframe,
        )
        loaded_data = environment.collect(4).observe()

        assert_frame_equal(
            data,
            loaded_data,
            check_exact=False,
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
    def test_integration_error(self) -> None:
        environment = OdeEnvironment(
            NonIntegrableSystem(t=0.0, state=SimpleState(x=1.0)),
            map_to_dataframe=map_to_dataframe,
        )
        with pytest.raises(IntegrationError):
            environment.collect(10)


if __name__ == "__main__":
    unittest.main()
