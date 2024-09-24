import polars as pl
from torch import Tensor
from torch.utils.data import Dataset


class TorchDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        inputs: pl.DataFrame,
        outputs: pl.DataFrame | None = None,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor]:
        inputs = Tensor(self.inputs.row(item))
        if self.outputs is None:
            outputs = Tensor([])
        else:
            outputs = Tensor(self.outputs.row(item))
        return inputs, outputs
