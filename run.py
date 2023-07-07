import argparse
from functools import reduce
from pathlib import Path

from agenc.data import Dataset
from agenc.experiment import Experiment


def _instantiate_class(class_path, init_arguments):
    class_module, class_name = class_path.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)

    return args_class(**init_arguments)


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

    

    preprocessors = [
        _instantiate_class(
            preprocessor.class_path, preprocessor.init_arguments
        )
        for preprocessor in experiment.data.preprocessors
    ]
    learner = _instantiate_class(
        experiment.learner.class_path,
        experiment.learner.init_arguments,
    )
    metrics = [
        _instantiate_class(metric.class_path, metric.init_arguments)
        for metric in experiment.metrics
    ]

    dataset = reduce(
        lambda dataset, transform: transform(dataset),
        preprocessors,
        dataset,
    )

    train_data, test_data = dataset.train_test_split(
        experiment.data.train_test_split,
        experiment.random_state,
    )

    learner.train(train_data)

    # TODO: the prediction should not get the ground truth
    predictions = learner.predict(test_data)

    for metric in metrics:
        print(f"{metric.name}: {metric(test_data.outputs(), predictions)}")


if __name__ == "__main__":
    main()
