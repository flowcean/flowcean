import polars as pl
import numpy as np

from agenc.experiment import Experiment


class Data:
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = inputs
        self.outputs = outputs

    @classmethod
    def from_experiment(cls, experiment: Experiment, split: str) -> "Data":
        columns = [column.name for column in experiment.metadata.columns]
        if split == "train":
            path = experiment.metadata.train_data
        elif split == "test":
            path = experiment.metadata.test_data
        else:
            raise ValueError(f"unknown split: `{split}`")

        data = pl.read_csv(path, columns=columns)
        inputs = data.select(experiment.inputs).to_numpy()
        outputs = data.select(experiment.outputs).to_numpy()

        assert len(inputs) == len(outputs)

        return cls(
            inputs=inputs,
            outputs=outputs,
        )

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray]:
        return self.inputs[item], self.outputs[item]
