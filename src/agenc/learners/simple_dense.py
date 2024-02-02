import os
from pathlib import Path
from typing import Any

import lightning
import numpy as np
import polars as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from typing_extensions import override

from agenc.core import Learner


class TorchDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        inputs: NDArray[Any],
        outputs: NDArray[Any] | None = None,
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor]:
        inputs = Tensor(self.inputs[item])
        if self.outputs is None:
            outputs = Tensor([])
        else:
            outputs = Tensor(self.outputs[item])
        return inputs, outputs


class SimpleDense(Learner):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float = 1e-3,
        hidden_dimensions: list[int] | None = None,
        batch_size: int = 32,
        num_workers: int | None = None,
        max_epochs: int = 100,
    ) -> None:
        if hidden_dimensions is None:
            hidden_dimensions = []
        if num_workers is None:
            num_workers = os.cpu_count() or 1
        self.model = MultilayerPerceptron(
            learning_rate=learning_rate,
            input_size=input_size,
            output_size=output_size,
            hidden_dimensions=hidden_dimensions,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.trainer: None | lightning.Trainer = None

    @override
    def train(
        self,
        input_features: pl.DataFrame,
        output_features: pl.DataFrame,
    ) -> None:
        input_data = input_features.to_numpy()
        output_data = output_features.to_numpy()
        train_len = int(0.8 * len(input_data))
        validation_len = len(input_data) - train_len
        train_dataset, val_dataset = random_split(
            TorchDataset(
                input_data,
                output_data,
            ),
            [train_len, validation_len],
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        validation_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        early_stopping = EarlyStopping(
            monitor="validate/loss",
            patience=5,
            mode="min",
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="validate/loss",
            mode="min",
            save_last=True,
        )
        logger = TensorBoardLogger(Path.cwd(), default_hp_metric=False)
        self.trainer = lightning.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[early_stopping, checkpoint_callback],
            logger=logger,
            enable_checkpointing=True,
        )
        self.trainer.fit(self.model, train_dataloader, validation_dataloader)

        self.model = MultilayerPerceptron.load_from_checkpoint(
            checkpoint_callback.best_model_path,
        )

    @override
    def predict(self, input_features: pl.DataFrame) -> NDArray[Any]:
        dataloader = DataLoader(
            TorchDataset(input_features.to_numpy()),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
        predictions = []
        for batch in dataloader:
            inputs, _ = batch
            predictions.append(self.model(inputs).detach().numpy())
        return np.concatenate(predictions)

    @override
    def save(self, path: Path) -> None:
        if self.trainer is None:
            raise RuntimeError(
                "this learner has not been trained yet, cannot save model"
                " before training"
            )
        self.trainer.save_checkpoint(path)

    @override
    def load(self, path: Path) -> None:
        self.model = MultilayerPerceptron.load_from_checkpoint(path)


class MultilayerPerceptron(lightning.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        input_size: int,
        output_size: int,
        hidden_dimensions: list[int] | None = None,
    ) -> None:
        if hidden_dimensions is None:
            hidden_dimensions = []
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        layers: list[Module] = []
        hidden_size = input_size
        for dimension in hidden_dimensions:
            layers.append(torch.nn.Linear(hidden_size, dimension))
            layers.append(torch.nn.LeakyReLU())
            hidden_size = dimension
        layers.append(torch.nn.Linear(hidden_size, output_size))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        tensor: Tensor = self.model(x)
        return tensor

    def _shared_eval_step(self, batch: Tensor) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        return torch.nn.functional.mse_loss(outputs, targets)

    def training_step(self, batch: Tensor, _batch_index: int) -> Tensor:
        loss = self._shared_eval_step(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Tensor, _batch_index: int) -> Tensor:
        loss = self._shared_eval_step(batch)
        self.log("validate/loss", loss)
        return loss

    def configure_optimizers(
        self,
    ) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=40,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
