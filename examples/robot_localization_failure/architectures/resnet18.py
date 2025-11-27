# architectures/resnet18.py

import lightning
import torch
from torch import nn
from torchvision import models


class ResNet18(lightning.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])
        self.learning_rate = learning_rate
        self.pos_weight = pos_weight

        # Load pretrained backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove classifier head
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # Custom classification head
        self.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # ImageNet normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x):
        # x is [B,3,H,W] with values in [0,1]
        x = (x - self.mean) / self.std
        features = self.backbone(x)
        return self.fc(features)

    def training_step(self, batch, _):
        inputs, targets = batch
        targets = targets.float().view(-1, 1)

        outputs = self(inputs)

        loss = nn.functional.binary_cross_entropy_with_logits(
            outputs,
            targets,
            pos_weight=self.pos_weight,
        )

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
