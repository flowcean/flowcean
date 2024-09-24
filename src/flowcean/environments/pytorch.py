import polars as pl
from torch import Tensor
from torch.utils.data import Dataset


class TorchDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset for PyTorch."""

    def __init__(
        self,
        inputs: pl.DataFrame,
        outputs: pl.DataFrame | None = None,
    ) -> None:
        """Initialize the TorchDataset.

        Args:
            inputs: The input data.
            outputs: The output data. Defaults to None.
        """
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.inputs)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor]:
        """Return the item at the given index.

        Args:
            item: The index of the item to return.

        Returns:
            The inputs and outputs at the given index.
        """
        inputs = Tensor(self.inputs.row(item))
        if self.outputs is None:
            outputs = Tensor([])
        else:
            outputs = Tensor(self.outputs.row(item))
        return inputs, outputs
