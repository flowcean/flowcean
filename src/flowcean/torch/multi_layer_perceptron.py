from typing import TYPE_CHECKING, Any

import lightning
import torch
from torch import Tensor
from torch.optim.adam import Adam
from typing_extensions import override

if TYPE_CHECKING:
    from torch.nn import Module


class MultilayerPerceptron(lightning.LightningModule):
    """A multilayer perceptron."""

    def __init__(
        self,
        learning_rate: float,
        output_size: int,
        hidden_dimensions: list[int] | None = None,
        *,
        activation_function: type[torch.nn.Module] | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            learning_rate: The learning rate.
            output_size: The size of the output.
            hidden_dimensions: The dimensions of the hidden layers.
            activation_function: The activation function to use.
                Defaults to ReLU if not provided.
        """
        super().__init__()
        if hidden_dimensions is None:
            hidden_dimensions = []
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        layers: list[Module] = []
        for dimension in hidden_dimensions:
            layers.extend(
                (
                    torch.nn.LazyLinear(dimension),
                    activation_function()
                    if activation_function
                    else torch.nn.ReLU(),
                ),
            )
        layers.append(torch.nn.LazyLinear(output_size))
        self.model = torch.nn.Sequential(*layers)

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        y: Tensor = self.model(*args, **kwargs)
        return y

    @override
    def training_step(self, batch: Any) -> Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
        )
        return loss

    @override
    def configure_optimizers(self) -> Any:
        optimizer = Adam(
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
