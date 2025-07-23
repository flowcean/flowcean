import lightning
import torch
from torch import nn


class CNN(lightning.LightningModule):
    def __init__(
        self,
        image_size: int = 36,
        in_channels: int = 3,  # New parameter for input channels
        learning_rate: float = 0.001,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels  # Store the input channels
        self.learning_rate = learning_rate

        # Define convolutional layers with configurable input channels
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                padding=1,
            ),  # Input: in_channels, Output: 32 channels
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduces 36x36 to 18x18
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),  # Input: 32 channels, Output: 64 channels
            nn.ReLU(),
            nn.MaxPool2d(2),  # Reduces 18x18 to 9x9
        )

        # Compute flattened size dynamically
        dummy_input = torch.zeros(
            1,
            in_channels,
            image_size,
            image_size,
        )  # Update dummy input
        conv_output = self.conv_layers(dummy_input)
        flattened_size = conv_output.view(1, -1).size(1)

        # Define fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze the extra dimension
        # (e.g., [4, 1, 2, 200, 200] -> [4, 2, 200, 200])
        x = x.squeeze(1)  # Remove dimension at index 1 (the extra 1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return torch.sigmoid(x)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        _batch_idx: int,
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        if targets.dim() == 1:
            targets = targets.unsqueeze(
                1,
            )  # Ensure targets are [batch_size, 1]
        targets = targets.float()
        loss = nn.functional.binary_cross_entropy(outputs, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
