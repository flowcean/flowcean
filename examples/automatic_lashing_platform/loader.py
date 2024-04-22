import json
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, Self, TypedDict

import polars as pl
import polars.selectors as cs
from typing_extensions import override

from flowcean.core import OfflineEnvironment
from flowcean.core.environment import NotLoadedError


@dataclass
class AlpSample:
    timeseries: pl.DataFrame
    parameters: dict[str, Any]


class AlpDataLoader(OfflineEnvironment):
    path: Path
    data: pl.DataFrame | None = None

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.data = None

    @override
    def load(self) -> Self:
        sample_names = [
            Path(item).with_suffix("")
            for item in Path.rglob(self.path, "*.json")
        ]

        datas = [
            AlpSample(
                timeseries=pl.read_parquet(name.with_suffix(".parquet")),
                parameters=_melt_list_of_dicts(
                    json.load(name.with_suffix(".json").open()),
                ),
            )
            for name in sample_names
        ]
        self.data = reduce(
            lambda x, y: x.vstack(y),
            [
                sample.timeseries.gather_every(100)
                .unstack(
                    1,
                    columns=~cs.by_name("t"),
                )
                .hstack(
                    pl.DataFrame(
                        {
                            "containerWeight": sample.parameters[
                                "containerWeight"
                            ],
                            "p_initial": sample.parameters["p_initial"],
                            "valveState0": sample.parameters["valveState"][0],
                            "valveState1": sample.parameters["valveState"][1],
                            "valveState2": sample.parameters["valveState"][2],
                            "valveState3": sample.parameters["valveState"][3],
                            "activeValveCount": sample.parameters[
                                "activeValveCount"
                            ],
                        },
                    ),
                )
                for sample in datas
            ],
        )
        return self

    @override
    def get_data(self) -> pl.DataFrame:
        if self.data is None:
            raise NotLoadedError
        return self.data


class _ParameterEntry(TypedDict):
    name: str
    value: Any


def _melt_list_of_dicts(data: list[_ParameterEntry]) -> dict[str, Any]:
    return {entry["name"]: entry["value"] for entry in data}
