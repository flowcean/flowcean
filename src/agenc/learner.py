import lightning
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
        trainer = lightning.Trainer(max_epochs=self.parameters["max_epochs"])
        trainer.fit(self.model, train_dataloader, validation_dataloader)

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
