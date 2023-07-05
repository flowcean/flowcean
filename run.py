from agenc.data import Dataset
from agenc.experiment import Experiment
from agenc.metrics import rmse, mae
from agenc.learner import Learner


def main():
    experiment = Experiment.load_from_path("experiments/failure_time_prediction.yaml")

    dataset = Dataset.from_experiment(experiment)

    train_data, test_data = dataset.train_test_split(
        experiment.data.train_test_split,
        experiment.random_state,
    )

    learner = Learner(**experiment.learner.parameters)

    learner.train(train_data)

    predictions = learner.predict(test_data)

    print(f"RMSE: {rmse([outputs for _, outputs in test_data], predictions)}")
    print(f"MAE: {mae([outputs for _, outputs in test_data], predictions)}")


if __name__ == "__main__":
    main()
