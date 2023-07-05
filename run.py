import argparse
from pathlib import Path

from agenc.data import Dataset
from agenc.experiment import Experiment
from agenc.metrics import mae, rmse


def instantiate_learner(experiment: Experiment):
    class_module, class_name = experiment.learner.class_path.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)

    return args_class(**experiment.learner.parameters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Path to experiment file",
    )
    arguments = parser.parse_args()

    experiment = Experiment.load_from_path(arguments.experiment)

    dataset = Dataset.from_experiment(experiment)

    train_data, test_data = dataset.train_test_split(
        experiment.data.train_test_split,
        experiment.random_state,
    )

    learner = instantiate_learner(experiment)
    learner.train(train_data)

    predictions = learner.predict(test_data)

    print(f"RMSE: {rmse([outputs for _, outputs in test_data], predictions)}")
    print(f"MAE: {mae([outputs for _, outputs in test_data], predictions)}")


if __name__ == "__main__":
    main()
