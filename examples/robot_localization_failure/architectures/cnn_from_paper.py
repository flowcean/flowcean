import lightning
import torch
from torch import nn


class CNN(lightning.LightningModule):
    def __init__(
        self,
        *,
        in_channels: int,
        learning_rate: float,
        pos_weight: torch.Tensor | None = None,  # <- preserved
    ) -> None:
        super().__init__()
        self.pos_weight = pos_weight  # kept even though unused in MSE
        self.learning_rate = learning_rate

        # ========= PAPER ARCHITECTURE =========
        # 4× (Conv2D + ReLU + MaxPool)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 100 → 50
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 50 → 25
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25 → 12
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12 → 6
        )

        # final feature size = 256 * 6 * 6
        feature_dim = 256 * 6 * 6

        # Fully connected block from paper (1 hidden + tanh output)
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),  # required by paper
        )

        # Weight init: normal distribution (paper)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ========= FORWARD =========
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.flatten(1)
        return self.fc_layers(x)

    # ========= TRAINING STEP =========
    # Using squared error (paper)
    def training_step(self, batch, _):
        inputs, targets = batch

        targets = targets.float().squeeze().view(-1, 1)

        outputs = self(inputs)
        loss = nn.functional.mse_loss(outputs, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    # ========= OPTIMIZER =========
    # Paper used Adadelta instead of Adam
    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=self.learning_rate)
