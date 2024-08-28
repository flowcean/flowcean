import random
from typing import Self, override

import polars as pl

from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.environment.observable import Observable
from flowcean.core.environment.stepable import Finished, Stepable
from flowcean.core.transform import Transform


# class MyObservable(Observable[pl.DataFrame], Stepable):
#     @override
#     def observe(self) -> pl.DataFrame:
#         return pl.DataFrame(
#             {
#                 "x": pl.arange(0, 10, eager=True).cast(pl.Float32) / 10,
#                 "y": pl.arange(10, 0, -1, eager=True).cast(pl.Float32) / 10,
#             },
#         )
#
#     @override
#     def step(self) -> None:
#         print("step")
#
#     def asdf(self) -> int:
#         return 42
#
#
# class MyTransform(Transform[pl.DataFrame, int]):
#     @override
#     def transform(self, data: pl.DataFrame) -> int:
#         return len(data.select("x"))
#
#
# foo = MyObservable()
# my_transform = MyTransform()
# bar = foo.with_transform(my_transform)
#
# data = bar.observe()
# foo.step()
#
# print(data)
#
# exit()
#
# N = 10
#
# a = [1, 2, 3]
# len(map(lambda x: x * 2, a))
#
# data = pl.DataFrame(
#     {
#         "x": pl.arange(0, N, eager=True).cast(pl.Float32) / N,
#         "y": pl.arange(N, 0, -1, eager=True).cast(pl.Float32) / N,
#     },
# )


class MyEnvironment(IncrementalEnvironment):
    def __init__(self, n: int) -> None:
        self.n = n
        self.state = 42.0

    @override
    def load(self) -> Self:
        print("Loading the environment.")
        return self

    @override
    def observe(self) -> float:
        print("Observing the environment.")
        return self.state

    @override
    def step(self) -> None:
        print("Advancing the environment by one step.")
        self.n -= 1
        if self.n == 0:
            raise Finished
        self.state = random.random()


if __name__ == "__main__":
    env = MyEnvironment(n=10)
    env.load()
    for data in env:
        print(f"Data: {data}")
