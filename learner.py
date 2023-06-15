import lightning
import pandas as pd
import torch


class AccelerometerDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        # Load data
        data = pd.read_csv(csv_file)
        # 2: y-amplitude, 4: z-amplitude, 6: growth rate
        self.inputs = data.iloc[:, [2,4,6]].values 
        self.targets = data.iloc[:, -1].values

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
            full_training_dataset = AccelerometerDataset("train.csv")
            train_len = int(0.8 * len(full_training_dataset))
            val_len = len(full_training_dataset) - train_len
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_training_dataset, [train_len, val_len])
        if stage == "test" or stage is None:
            self.test_dataset = AccelerometerDataset("test.csv")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=32)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=32)
    

class MLP(lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, 25),
            torch.nn.Sigmoid(),
            torch.nn.Linear(25, 1),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log("validate/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    

if __name__ == "__main__":
    data = AccelerometerData()
    model = MLP()
    trainer = lightning.Trainer(max_epochs=10)
    trainer.fit(model, data)
