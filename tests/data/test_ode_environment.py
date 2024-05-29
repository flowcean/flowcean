import unittest
from functools import reduce
from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy.linalg import expm

from flowcean.environments.ode_environment import (
    IntegrationError,
    OdeEnvironment,
)


class TestOdeEnvironment(unittest.TestCase):
    def test_length(self) -> None:
        environment = OdeEnvironment(
            lambda _, x: np.array([-1 / 2 * x[0]]),
            np.array([1]),
        )
        # For the OdeEnvironment this call does nothing, but by convention
        # `load` should still be called
        environment.load()
        loaded_data = environment.take(10)
        assert len(loaded_data) == 10

    def test_simple_ode(self) -> None:
        data = pl.DataFrame(
            {
                "y_0": [np.exp(-1 / 2 * t) for t in range(10)],
            },
        )

        environment = OdeEnvironment(
            lambda _, x: np.array([-1 / 2 * x[0]]),
            np.array([1]),
        )
        environment.load()
        loaded_data = environment.take(10)

        assert_frame_equal(
            data,
            loaded_data,
            check_exact=False,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_time_dependent_ode(self) -> None:
        data = pl.DataFrame(
            {
                "y_0": [np.sqrt(np.power(t, 2) + 1) for t in range(10)],
            },
        )

        environment = OdeEnvironment(
            lambda t, x: np.array([t / x[0]]),
            np.array([1]),
        )
        environment.load()
        loaded_data = environment.take(10)

        assert_frame_equal(
            data,
            loaded_data,
            check_exact=False,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_named_ode(self) -> None:
        data = pl.DataFrame(
            {
                "result": [np.exp(-1 / 2 * t) for t in range(10)],
            },
        )
        environment = OdeEnvironment(
            lambda _, x: np.array([-1 / 2 * x[0]]),
            np.array([1]),
            output_names=["result"],
        )
        environment.load()
        loaded_data = environment.take(10)

        assert_frame_equal(
            data,
            loaded_data,
            check_exact=False,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_output_function(self) -> None:
        data = pl.DataFrame(
            {
                "t": [float(x) for x in range(10)],
                "y": [np.exp(-1 / 2 * t) * 2 for t in range(10)],
            },
        )
        environment = OdeEnvironment(
            lambda _, x: np.array([-1 / 2 * x[0]]),
            np.array([1]),
            g=lambda t, x: np.array([t, x[0] * 2]),
            output_names=["t", "y"],
        )
        environment.load()
        loaded_data = environment.take(10)

        assert_frame_equal(
            data,
            loaded_data,
            check_exact=False,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_vector_ode(self) -> None:
        A = np.array([[0, 1], [-10, 0]])  # noqa: N806
        x0 = np.array([0, 1])

        def analytical_fnc(t: float) -> npt.NDArray[np.float64]:
            return cast(npt.NDArray[np.float64], np.matmul(expm(A * t), x0))

        data = pl.DataFrame(
            reduce(
                lambda x, y: np.vstack((x, y)),
                [analytical_fnc(t) for t in range(10)],
            ),
            schema=["y_0", "y_1"],
        )

        environment = OdeEnvironment(
            lambda _, x: np.matmul(A, x),
            x0,
            implicit_integration=True,
        )
        environment.load()
        loaded_data = environment.take(10)

        assert_frame_equal(
            data,
            loaded_data,
            check_exact=False,
            atol=1e-2,
            rtol=1e-2,
        )

    def test_integration_error(self) -> None:
        environment = OdeEnvironment(
            lambda _, x: -np.array(np.sqrt(x[0])),
            np.array([1]),
        )
        environment.load()
        with pytest.raises(IntegrationError):
            environment.take(10)


if __name__ == "__main__":
    unittest.main()
