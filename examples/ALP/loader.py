import glob
import json
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any, TypedDict

import polars as pl
import polars.selectors as cs
from agenc.core import DataLoader


@dataclass
class AlpSample:
    timeseries: pl.DataFrame
    parameters: dict[str, Any]


class AlpDataLoader(DataLoader):
    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> pl.DataFrame:
        sample_names = [
            Path(item).with_suffix("")
            for item in glob.glob(str(self.path / "*.json"))
        ]

        datas = [
            AlpSample(
                timeseries=pl.read_parquet(name.with_suffix(".parquet")),
                parameters=_melt_list_of_dicts(
                    json.load(open(name.with_suffix(".json")))
                ),
            )
            for name in sample_names
        ]
        return reduce(
            lambda x, y: x.vstack(y),
            [
                sample.timeseries.gather_every(100)
                .unstack(
                    1,
                    columns=~cs.by_name("t"),
                )
                .hstack(
                    pl.DataFrame({
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
                    })
                )
                for sample in datas
            ],
        )


class _ParameterEntry(TypedDict):
    name: str
    value: Any


def _melt_list_of_dicts(data: list[_ParameterEntry]) -> dict[str, Any]:
    return {entry["name"]: entry["value"] for entry in data}
