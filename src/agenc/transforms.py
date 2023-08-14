import polars as pl


class Transform:
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError


class StandardScaler(Transform):
    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.select((pl.all() - pl.all().mean()) / pl.all().std())
