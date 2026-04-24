import numpy as np
from pysr import PySRRegressor


def test_real_pysr_backend_smoke() -> None:
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)

    model = PySRRegressor(
        progress=False,
        verbosity=0,
        niterations=1,
        populations=2,
        population_size=20,
        maxsize=7,
        warm_start=False,
    )

    model.fit(x, y)

    predictions = model.predict(x)
    assert predictions.shape == (4,)
