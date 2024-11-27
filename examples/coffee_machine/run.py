from pathlib import Path

from tqdm import tqdm

import flowcean.cli
from flowcean.core.environment.chained import ChainedOfflineEnvironments
from flowcean.environments.train_test_split import TrainTestSplit
from flowcean.environments.uri import UriDataLoader
from flowcean.learners.grpc_passive_automata.learner import (
    GrpcPassiveAutomataLearner,
)
from flowcean.metrics.regression import MeanAbsoluteError, MeanSquaredError
from flowcean.strategies.offline import evaluate_offline, learn_offline
from flowcean.transforms.explode import Explode
from flowcean.transforms.select import Select
from flowcean.transforms.to_time_series import ToTimeSeries
from flowcean.transforms.unnest import Unnest


def main() -> None:
    flowcean.cli.initialize_logging()

    data = ChainedOfflineEnvironments(
        [
            UriDataLoader("file:" + path.as_posix()).with_transform(
                ToTimeSeries("t")
            )
            for path in tqdm(
                list(Path("./data").glob("*.csv")),
                desc="Loading environments",
            )
        ]
    )
    print(data.observe().head())
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(
        data.collect()
    )

    learner = GrpcPassiveAutomataLearner.run_docker(
        image="ghcr.io/flowcean/flowcean/java-automata-learner:latest",
        pull=True,
    )
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
        Explode(["output"]) | Unnest(["output"]) | Select(["value"]),
    )
    print(report)


if __name__ == "__main__":
    main()
