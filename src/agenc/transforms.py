import polars as pl

from agenc.data import Dataset


class Transform:
    def __call__(self, data: Dataset) -> Dataset:
        raise NotImplementedError


class StandardScaler(Transform):
    def __call__(self, data: Dataset) -> Dataset:
        return Dataset(
            data=self._transform(data.data),
            input_columns=data.input_columns,
            output_columns=data.output_columns,
        )

    def _transform(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.select((pl.all() - pl.all().mean()) / pl.all().std())
