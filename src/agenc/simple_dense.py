import os
from pathlib import Path

import lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from torch.utils.data.dataloader import DataLoader

from agenc.learner import Learner


class TorchDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray | None = None,
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.Tensor(self.inputs[item])
        if self.outputs is None:
            outputs = torch.Tensor([])
        else:
            outputs = torch.Tensor(self.outputs[item])
        return inputs, outputs


class SimpleDense(Learner):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        learning_rate: float = 1e-3,
        hidden_dimensions: list[int] = [],
        batch_size: int = 32,
        num_workers: int = os.cpu_count() or 1,
        max_epochs: int = 100,
    ) -> None:
        self.model = MultilayerPerceptron(
            learning_rate=learning_rate,
            input_size=input_size,
            output_size=output_size,
            hidden_dimensions=hidden_dimensions,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs

    def train(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        train_len = int(0.8 * len(inputs))
        validation_len = len(inputs) - train_len
        train_dataset, val_dataset = random_split(
            TorchDataset(inputs, outputs),
            [train_len, validation_len],
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        validation_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
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
        trainer = lightning.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[early_stopping, checkpoint_callback],
            logger=logger,
            enable_checkpointing=True,
        )
        trainer.fit(self.model, train_dataloader, validation_dataloader)

        self.model = MultilayerPerceptron.load_from_checkpoint(
            checkpoint_callback.best_model_path,
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        dataloader = DataLoader(
            TorchDataset(inputs),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        predictions = []
        for batch in dataloader:
            inputs, _ = batch
            predictions.append(self.model(inputs).detach().numpy())
        return np.concatenate(predictions)


class MultilayerPerceptron(lightning.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        input_size: int,
        output_size: int,
        hidden_dimensions: list[int] = [],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        layers = []
        hidden_size = input_size
        for dimension in hidden_dimensions:
            layers.append(torch.nn.Linear(hidden_size, dimension))
            layers.append(torch.nn.LeakyReLU())
            hidden_size = dimension
        layers.append(torch.nn.Linear(hidden_size, output_size))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def _shared_eval_step(self, batch):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        return loss

    def training_step(self, batch, _):
        loss = self._shared_eval_step(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, _):
        loss = self._shared_eval_step(batch)
        self.log("validate/loss", loss)
        return loss

    def test_step(self, batch, _):
        loss = self._shared_eval_step(batch)
        self.log("test/loss", loss)
        return loss

    def configure_optimizers(self):
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
