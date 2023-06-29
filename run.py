from agenc.data import Data
from agenc.experiment import Experiment
from agenc.metrics import rmse, mae
from agenc.learner import Learner


def main():
    experiment = Experiment.load_from_path("experiments/failure_time_prediction.yaml")

    train_data = Data.from_experiment(experiment, "train")
    test_data = Data.from_experiment(experiment, "test")

    learner = Learner(experiment.learner.parameters)

    learner.train(train_data)

    predictions = learner.predict(test_data)

    print(f"RMSE: {rmse([outputs for _, outputs in test_data], predictions)}")
    print(f"MAE: {mae([outputs for _, outputs in test_data], predictions)}")


if __name__ == "__main__":
    main()
