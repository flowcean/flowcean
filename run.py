from agenc.data import Dataset
from agenc.experiment import Experiment
from agenc.metrics import rmse, mae
from agenc.learner import Learner


def main():
    experiment = Experiment.load_from_path("experiments/failure_time_prediction.yaml")

    dataset = Dataset.from_experiment(experiment)

    learner = Learner(experiment.learner.parameters)

    learner.train(dataset.train_data)

    predictions = learner.predict(dataset.test_data)

    print(f"RMSE: {rmse([outputs for _, outputs in dataset.test_data], predictions)}")
    print(f"MAE: {mae([outputs for _, outputs in dataset.test_data], predictions)}")


if __name__ == "__main__":
    main()
