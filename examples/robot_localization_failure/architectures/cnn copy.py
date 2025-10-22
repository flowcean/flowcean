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
        pos_weight: torch.Tensor | None = None,  ############# New param added
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        # self.pos_weight = torch.tensor(3.0)  #################
        ################################################
        default_pos_weight = torch.tensor([1.0])  # must be a 1D tensor
        self.register_buffer(
            "pos_weight",
            pos_weight if pos_weight is not None else default_pos_weight,
        )
        ################################################
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
        return self.fc_layers(x)  # [B, 1]

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        _batch_idx: int,
    ) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        #####
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            # print(
            #     f"Batch stats → logits: mean={outputs.mean().item():.3f}, "
            #     f"min={outputs.min().item():.3f}, max={outputs.max().item():.3f}, "
            #     f"sigmoid_mean={probs.mean().item():.3f}",
            # )
            print(
                f"Positives>0.5: {(probs > 0.5).float().mean().item():.3f}, "
                f"Sigmoid mean={probs.mean().item():.3f}",
            )
        ###
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        # print(
        #     f"Device check — outputs: {outputs.device}, pos_weight: {self.pos_weight.device}"
        # )  #############
        loss = nn.functional.binary_cross_entropy_with_logits(
            outputs,
            targets,
            pos_weight=self.pos_weight * 10.0,  ####
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
