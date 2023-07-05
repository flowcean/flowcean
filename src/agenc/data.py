from typing import Tuple, cast

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from agenc.experiment import Experiment


class Dataset:
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = inputs
        self.outputs = outputs

    @classmethod
    def from_experiment(cls, experiment: Experiment) -> "Dataset":
        columns = [column.name for column in experiment.metadata.columns]

        data = pl.read_csv(experiment.metadata.data_path, columns=columns)
        inputs = data.select(experiment.data.inputs).to_numpy()
        outputs = data.select(experiment.data.outputs).to_numpy()

        assert len(inputs) == len(outputs)

        return cls(
            inputs=inputs,
            outputs=outputs,
        )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray]:
        return self.inputs[item], self.outputs[item]

    def train_test_split(
        self,
        train_size: float,
        random_state: int,
    ) -> Tuple["Dataset", "Dataset"]:
        inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
            self.inputs,
            self.outputs,
            train_size=train_size,
            random_state=random_state,
        )

        train_data = Dataset(
            cast(np.ndarray, inputs_train),
            cast(np.ndarray, outputs_train),
        )
        test_data = Dataset(
            cast(np.ndarray, inputs_test),
            cast(np.ndarray, outputs_test),
        )
        return train_data, test_data
