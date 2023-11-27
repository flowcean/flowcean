from typing import Any

from numpy.typing import NDArray


class Dataset:
    input_columns: list[str]
    output_columns: list[str]
    data: NDArray[Any]

    def __init__(
        self,
        input_columns: list[str],
        output_columns: list[str],
        data: NDArray[Any],
    ):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.data = data
