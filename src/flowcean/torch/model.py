from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from typing_extensions import override

from flowcean.core import Model

from .dataset import TorchDataset


class PyTorchModel(Model):
    """PyTorch model wrapper."""

    def __init__(
        self,
        module: Module,
        output_names: list[str],
        batch_size: int = 32,
        num_workers: int = 1,
    ) -> None:
        """Initialize the model.

        Args:
            module: The PyTorch module.
            output_names: The names of the output columns.
            batch_size: The batch size to use for predictions.
            num_workers: The number of workers to use for the DataLoader.
        """
        self.module = module
        self.output_names = output_names
        self.batch_size = batch_size
        self.num_workers = num_workers

    @override
    def predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        dataloader = DataLoader(
            TorchDataset(input_features.collect()),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        predictions = []
        for batch in dataloader:
            inputs, _ = batch
            predictions.append(self.module(inputs).detach().numpy())
        predictions = np.concatenate(predictions, axis=0)
        return pl.DataFrame(predictions, self.output_names).lazy()

    @override
    def save(self, path: Path) -> None:
        torch.save(self.module.state_dict(), path)

    @override
    def load(self, path: Path) -> None:
        self.module.load_state_dict(torch.load(path))
