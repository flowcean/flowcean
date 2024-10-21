from typing import override

import polars as pl
import torch
from torch import nn
from torch.optim.sgd import SGD

from flowcean.core.learner import SupervisedIncrementalLearner
from flowcean.models.pytorch import PyTorchModel


class LinearRegression(SupervisedIncrementalLearner):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float = 1e-3,
        loss: nn.Module | None = None,
    ) -> None:
        self.model = nn.Linear(input_size, output_size)
        self.loss = loss or nn.MSELoss()
        self.optimizer = SGD(
            self.model.parameters(),
            lr=learning_rate,
        )

    @override
    def learn_incremental(
        self,
        inputs: pl.DataFrame,
        outputs: pl.DataFrame,
    ) -> PyTorchModel:
        self.optimizer.zero_grad()
        features = torch.from_numpy(inputs.to_numpy(writable=True))
        labels = torch.from_numpy(outputs.to_numpy(writable=True))
        prediction = self.model(features)
        loss = self.loss(prediction, labels)
        loss.backward()
        self.optimizer.step()
        return PyTorchModel(self.model, outputs.columns)
