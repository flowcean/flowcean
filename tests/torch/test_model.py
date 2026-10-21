from collections.abc import Callable
from io import BytesIO

import polars as pl
import pytest
import torch
from polars.testing import assert_frame_equal

from flowcean.torch import PyTorchModel
from flowcean.torch.dataset import TorchDataset


class Double(torch.nn.Module):
    def __init__(self, *, vector: bool = False) -> None:
        super().__init__()
        self.vector = vector

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = inputs * 2
        return result[:, 0] if self.vector else result


@pytest.mark.parametrize(
    ("vector", "width"), [(True, 1), (False, 1), (False, 2)]
)
@pytest.mark.parametrize("batch_size", [1, 2, 32])
def test_predictions_preserve_batch_rows_and_target_width(
    vector: bool,
    width: int,
    batch_size: int,
) -> None:
    inputs = pl.DataFrame({f"x{i}": [1.0, 2.0, 3.0] for i in range(width)})
    names = [f"y{i}" for i in range(width)]
    model = PyTorchModel(Double(vector=vector), names, batch_size=batch_size)
    expected = pl.DataFrame({name: [2.0, 4.0, 6.0] for name in names}).cast(
        pl.Float32
    )
    assert_frame_equal(model.predict(inputs.lazy()).collect(), expected)
    assert model._module_device() == torch.device("cpu")
    blob = BytesIO()
    model.save(blob)
    blob.seek(0)
    assert_frame_equal(
        PyTorchModel.load(blob).predict(inputs.lazy()).collect(), expected
    )


class Malformed(torch.nn.Module):
    def __init__(self, output: Callable[[torch.Tensor], object]) -> None:
        super().__init__()
        self.output = output

    def forward(self, inputs: torch.Tensor) -> object:
        return self.output(inputs)


@pytest.mark.parametrize(
    "output",
    [
        lambda x: x[0, 0],
        lambda x: x.unsqueeze(0),
        lambda x: x[:1],
        lambda x: torch.cat([x, x], dim=1),
    ],
)
def test_invalid_output_shape_is_rejected(
    output: Callable[[torch.Tensor], object],
) -> None:
    model = PyTorchModel(Malformed(output), ["y"])
    with pytest.raises(ValueError, match="Prediction shape"):
        model.predict(pl.LazyFrame({"x": [1.0, 2.0]}))


def test_nontensor_output_is_rejected() -> None:
    model = PyTorchModel(Malformed(lambda x: (x, x)), ["y"])
    with pytest.raises(TypeError, match="tensors"):
        model.predict(pl.LazyFrame({"x": [1.0, 2.0]}))


def test_empty_inputs_preserve_output_schema_without_calling_module() -> None:
    def unexpected_call(inputs: torch.Tensor) -> object:
        pytest.fail("Empty prediction should not invoke the module")

    model = PyTorchModel(Malformed(unexpected_call), ["y", "z"])
    result = model.predict(pl.LazyFrame(schema={"x": pl.Float32})).collect()
    assert result.shape == (0, 2)
    assert result.schema == {"y": pl.Float32, "z": pl.Float32}


def test_dataset_rejects_unpaired_rows() -> None:
    with pytest.raises(ValueError, match="equal row counts"):
        TorchDataset(
            pl.DataFrame({"x": [1.0, 2.0]}), pl.DataFrame({"y": [1.0]})
        )
