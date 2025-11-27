import lightning
import torch
from torch import nn


class CNN(lightning.LightningModule):
    def __init__(
        self,
        *,
        in_channels: int,
        learning_rate: float,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.learning_rate = learning_rate

        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 → 128
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 → 64
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 → 32
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 → 16
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        feature_dim = 256

        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.gap(x).flatten(1)
        return self.fc_layers(x)

    def training_step(self, batch, _):
        inputs, targets = batch

        # Remove extra dimensions, ensure shape [B,1]
        targets = targets.float().squeeze()
        targets = targets.view(-1, 1)

        outputs = self(inputs)

        loss = nn.functional.binary_cross_entropy_with_logits(
            outputs,
            targets,
            pos_weight=self.pos_weight,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
