import lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import torch
from torch.utils.data import Dataset as _Dataset, random_split
from torch.utils.data.dataloader import DataLoader

from agenc.data import Data


class Dataset(_Dataset):
    def __init__(self, data: Data):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, outputs = self.data[item]
        return torch.Tensor(inputs), torch.Tensor(outputs)


class Learner:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.model = MLP(learning_rate=parameters["learning_rate"])

    def train(self, dataset: Data):
        train_len = int(0.8 * len(dataset))
        validation_len = len(dataset) - train_len
        train_dataset, val_dataset = random_split(
            Dataset(dataset),
            [train_len, validation_len],
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.parameters["batch_size"],
            shuffle=True,
            num_workers=self.parameters["num_workers"],
        )
        validation_dataloader = DataLoader(
            val_dataset,
            batch_size=self.parameters["batch_size"],
            num_workers=self.parameters["num_workers"],
        )
        early_stopping = EarlyStopping(
            monitor="validate/loss",
            patience=5,
            mode="min",
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",
            filename="model-{epoch:02d}",
            monitor="validate/loss",
            save_last=True,
        )
        logger = TensorBoardLogger(save_dir="logs/", default_hp_metric=False)
        trainer = lightning.Trainer(
            max_epochs=self.parameters["max_epochs"],
            callbacks=[early_stopping, checkpoint_callback],
            logger=logger,
            enable_checkpointing=True,
        )
        trainer.fit(self.model, train_dataloader, validation_dataloader)

        self.model = MLP.load_from_checkpoint(
            checkpoint_callback.best_model_path,
        )

    def predict(self, dataset: Data) -> np.ndarray:
        dataloader = DataLoader(
            Dataset(dataset),
            batch_size=self.parameters["batch_size"],
            num_workers=self.parameters["num_workers"],
        )
        predictions = []
        for batch in dataloader:
            inputs, _ = batch
            predictions.append(self.model(inputs).detach().numpy())
        return np.concatenate(predictions)


class MLP(lightning.LightningModule):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, 25),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(25, 25),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(25, 1),
        )

    def forward(self, x):
        input_mean = torch.Tensor([0.12495715, 0.10395051, 0.02667484])
        input_std = torch.Tensor([0.17149029, 0.11083332, 0.01697188])
        output_mean = torch.Tensor([3725.85228508])
        output_std = torch.Tensor([3710.73972826])
        return self.model((x - input_mean) / input_std) * output_std + output_mean

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
