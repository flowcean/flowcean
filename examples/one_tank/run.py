import logging
from datetime import UTC, datetime
from typing import NamedTuple

import numpy as np

import flowcean.cli
from flowcean.data.dataset import Dataset
from flowcean.data.ode_environment import ODEEnvironment
from flowcean.data.train_test_split import TrainTestSplit
from flowcean.learners.lightning import LightningLearner, MultilayerPerceptron
from flowcean.learners.regression_tree import RegressionTree
from flowcean.metrics import MeanAbsoluteError, MeanSquaredError, evaluate
from flowcean.strategies.offline import learn_offline
from flowcean.transforms.sliding_window import SlidingWindow

logger = logging.getLogger(__name__)


def main() -> None:
    flowcean.cli.initialize_logging()

    # This example is based on https://de.mathworks.com/help/slcontrol/ug/watertank-simulink-model.html.
    class Parameters(NamedTuple):
        A: float  # Area of the tank
        a: float  # Constant related to the out-flow of the tank
        b: float  # Constant related to the in-flow of the tank

    parameters = Parameters(
        A=5,
        a=0.5,
        b=2,
    )
    x0 = np.array([1])
    tstep = 0.1

    V = lambda t: np.max([0, np.sin(2 * np.pi * 1 / 10 * t)])  # noqa: E731, N806

    data_incremental = ODEEnvironment(
        lambda t, x, parameters=parameters: np.array(
            [
                (parameters.b * V(t) - parameters.a * np.sqrt(x[0]))
                / parameters.A,
            ],
        ),
        x0,
        g=lambda t, x: np.array([x[0], V(t)]),
        tstep=tstep,
        output_names=["h", "V"],
    )

    data = Dataset(data_incremental.load().take(250)).with_transform(
        SlidingWindow(window_size=3),
    )
    data = data.load()

    train, test = TrainTestSplit(ratio=0.8, shuffle=True).split(data)

    inputs = ["h_0", "h_1", "V_0", "V_1", "V_2"]
    outputs = ["h_2"]

    for learner in [
        RegressionTree(max_depth=5),
        LightningLearner(
            module=MultilayerPerceptron(
                learning_rate=1e-3,
                input_size=len(inputs),
                output_size=len(outputs),
                hidden_dimensions=[10, 10],
            ),
            max_epochs=100,
        ),
    ]:
        t_start = datetime.now(tz=UTC)
        model = learn_offline(
            train,
            learner,
            inputs,
            outputs,
        )
        delta_t = datetime.now(tz=UTC) - t_start
        print(f"Learning took {np.round(delta_t.microseconds / 1000, 1)} ms")

        report = evaluate(
            model,
            test,
            inputs,
            outputs,
            [MeanAbsoluteError(), MeanSquaredError()],
        )
        print(report)


if __name__ == "__main__":
    main()
