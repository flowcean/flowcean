from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
from typing_extensions import override

from flowcean.core import Model

if TYPE_CHECKING:
    from torch.nn import Module


class PyTorchModel(Model):
    """PyTorch model wrapper."""

    def __init__(
        self,
        module: Module,
        output_names: list[str],
        batch_size: int = 32,
        num_workers: int = 1,
    ) -> None:
        """Initialize the model.

        Args:
            module: The PyTorch module.
            output_names: The names of the output columns.
            batch_size: The batch size to use for predictions.
            num_workers: Retained for backward compatibility.
        """
        self.module = module
        self.output_names = output_names
        self.batch_size = batch_size
        self.num_workers = num_workers

    @override
    def _predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        collected_inputs = input_features.collect()
        if collected_inputs.height == 0:
            return pl.DataFrame(
                {
                    name: pl.Series(name, [], dtype=pl.Float32)
                    for name in self.output_names
                },
            ).lazy()

        inputs = torch.as_tensor(
            collected_inputs.to_numpy(),
            dtype=torch.float32,
        )
        self.module.eval()
        module_device = self._module_device()
        inputs = inputs.to(module_device)

        predictions = []
        with torch.inference_mode():
            for input_batch in self._iter_input_batches(inputs):
                output_batch = self.module(input_batch)
                predictions.append(output_batch.detach().cpu().numpy())

        prediction_array = np.concatenate(predictions, axis=0)
        return pl.DataFrame(prediction_array, schema=self.output_names).lazy()

    def _module_device(self) -> torch.device:
        """Determine the device where the wrapped module expects inputs."""
        first_parameter = next(self.module.parameters(), None)
        if first_parameter is not None:
            return first_parameter.device

        first_buffer = next(self.module.buffers(), None)
        if first_buffer is not None:
            return first_buffer.device

        return torch.device("cpu")

    def _iter_input_batches(
        self,
        inputs: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Yield micro-batches to balance throughput and memory use."""
        if self.batch_size <= 0 or inputs.shape[0] <= self.batch_size:
            return [inputs]

        return list(inputs.split(self.batch_size))
