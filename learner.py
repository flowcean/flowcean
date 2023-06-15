from pathlib import Path

import lightning
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class AccelerometerDataset(Dataset):
    def __init__(self, root: Path | str, train: bool = True):
        root = Path(root)
        if train:
            filename = "train.csv"
        else:
            filename = "test.csv"
        data = pl.read_csv(root / filename)
        print(data)
        self.inputs = data.select(
            "y-Amplitude",
            "z-Amplitude",
            "Growth-rate",
        ).to_numpy()
        self.targets = data.select("Estimated-Failure-Time").to_numpy()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        inputs = torch.tensor(self.inputs[idx], dtype=torch.float32)
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        return inputs, targets


class AccelerometerData(lightning.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate" or stage is None:
            full_training_dataset = AccelerometerDataset(
                "data/accelerometer/",
                train=True,
            )
            train_len = int(0.8 * len(full_training_dataset))
            val_len = len(full_training_dataset) - train_len
            self.train_dataset, self.val_dataset = random_split(
                full_training_dataset,
                [train_len, val_len],
            )
        if stage == "test" or stage is None:
            self.test_dataset = AccelerometerDataset("data/accelerometer", train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=64)


class MLP(lightning.LightningModule):
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, 25),
            torch.nn.Sigmoid(),
            torch.nn.Linear(25, 1),
            torch.nn.ReLU(),
        )

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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    data = AccelerometerData()
    model = MLP()
    trainer = lightning.Trainer(max_epochs=10)
    trainer.fit(model, data)
