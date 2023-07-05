import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from agenc.experiment import Experiment


class DataSplit:
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray]:
        return self.inputs[item], self.outputs[item]


class Dataset:
    def __init__(self, train_data: DataSplit, test_data: DataSplit):
        self.train_data = train_data
        self.test_data = test_data

    @classmethod
    def from_experiment(cls, experiment: Experiment) -> "Dataset":
        columns = [column.name for column in experiment.metadata.columns]

        data = pl.read_csv(experiment.metadata.data_path, columns=columns)
        inputs = data.select(experiment.data.inputs).to_numpy()
        outputs = data.select(experiment.data.outputs).to_numpy()

        assert len(inputs) == len(outputs)

        inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
            inputs,
            outputs,
            train_size=experiment.data.train_test_split,
            random_state=experiment.random_state,
        )

        train_data = DataSplit(inputs_train, outputs_train)
        test_data = DataSplit(inputs_test, outputs_test)

        return cls(
            train_data=train_data,
            test_data=test_data,
        )
