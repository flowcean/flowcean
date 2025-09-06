import lightning
import torch
from torch import nn


class CNN(lightning.LightningModule):
    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int,
        learning_rate: float,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.learning_rate = learning_rate

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Make it resolution-agnostic
        self.gap = nn.AdaptiveAvgPool2d(1)  # -> [B, 64, 1, 1]
        feature_dim = 64

        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)  # [B, 64, H/4, W/4]
        x = self.gap(x).flatten(1)  # [B, 64]
        x = self.fc_layers(x)  # [B, 1]
        return torch.sigmoid(x)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        _batch_idx: int,
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        loss = nn.functional.binary_cross_entropy(outputs, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
