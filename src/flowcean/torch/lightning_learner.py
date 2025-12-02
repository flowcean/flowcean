import os
import platform

import lightning
import polars as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from typing_extensions import override

from flowcean.core import SupervisedLearner

from .dataset import TorchDataset
from .model import PyTorchModel


class LightningLearner(SupervisedLearner):
    """A learner that uses PyTorch Lightning."""

    def __init__(
        self,
        module: lightning.LightningModule,
        num_workers: int | None = None,
        batch_size: int = 32,
        max_epochs: int = 100,
        accelerator: str = "auto",
    ) -> None:
        """Initialize the learner.

        Args:
            module: The PyTorch Lightning module.
            num_workers: The number of workers to use for the DataLoader.
            batch_size: The batch size to use for training.
            max_epochs: The maximum number of epochs to train for.
            accelerator: The accelerator to use.
        """
        self.module = module
        self.num_workers = num_workers or os.cpu_count() or 0
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = None
        self.accelerator = accelerator

    @override
    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> PyTorchModel:
        dfs = pl.collect_all([inputs, outputs])
        collected_inputs = dfs[0]
        collected_outputs = dfs[1]
        dataset = TorchDataset(collected_inputs, collected_outputs)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=platform.system() == "Windows",
        )
        trainer = lightning.Trainer(
            accelerator=self.accelerator,
            max_epochs=self.max_epochs,
            callbacks=[
                EarlyStopping(
                    monitor="train_loss",
                    patience=10,
                    mode="min",
                ),
            ],
        )
        self.module.example_input_array = dataset[0][0]
        trainer.fit(self.module, dataloader)
        return PyTorchModel(self.module, collected_outputs.columns)
