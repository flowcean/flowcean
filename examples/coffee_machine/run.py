import flowcean.cli
from flowcean.environments.train_test_split import TrainTestSplit
from flowcean.environments.uri import UriDataLoader
from flowcean.learners.grpc.learner import GrpcLearner
from flowcean.metrics import MeanAbsoluteError, MeanSquaredError, evaluate
from flowcean.strategies.offline import learn_offline


def main() -> None:
    flowcean.cli.initialize_logging()

    data = UriDataLoader(uri="file:./data/coffee_data.csv")
    data.load()
    train, test = TrainTestSplit(ratio=0.8, shuffle=False).split(data)

    learner = GrpcLearner.run_docker(
        image="collaborating.tuhh.de:5005/w-6/agenc/agenc/java-automata-learner:latest",
    )
    inputs = ["^i.*$", "^o\\d$", "^o1[0-8]$"]
    outputs = ["o19"]

    model = learn_offline(
        train,
        learner,
        inputs,
        outputs,
    )

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
