from pathlib import Path

from tqdm import tqdm

import flowcean.cli
from flowcean.core.environment.chain import ChainEnvironment
from flowcean.environments.train_test_split import TrainTestSplit
from flowcean.environments.uri import UriDataLoader
from flowcean.learners.grpc_passive_automata.learner import (
    GrpcPassiveAutomataLearner,
)
from flowcean.metrics import MeanAbsoluteError, MeanSquaredError
from flowcean.strategies.offline import evaluate_offline, learn_offline
from flowcean.transforms.explode import Explode
from flowcean.transforms.select import Select
from flowcean.transforms.unnest import Unnest


def main() -> None:
    flowcean.cli.initialize_logging()

    data = ChainEnvironment(
        *[
            UriDataLoader("file:" + path.as_posix()).load().to_time_series("t")
            for path in tqdm(
                list(Path("./data").glob("*.csv")),
                desc="Loading environments",
            )
        ]
    )
    print(data.get_data().head())
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    learner = GrpcPassiveAutomataLearner.with_address(address="localhost:8080")
    inputs = ["input"]
    outputs = ["output"]

    model = learn_offline(
        train,
        learner,
        inputs,
        outputs,
    )

    report = evaluate_offline(
        model,
        test,
        inputs,
        outputs,
        [MeanAbsoluteError(), MeanSquaredError()],
        Explode("output") | Unnest("output") | Select(["value"]),
    )
    print(report)


if __name__ == "__main__":
    main()
