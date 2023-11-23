from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._dynamic_loader import load_instance


@dataclass
class InstanceSpecification:
    class_path: str
    arguments: dict[str, Any]

    @classmethod
    def from_string(cls, class_path: str) -> "InstanceSpecification":
        return cls(class_path=class_path, arguments={})

    @classmethod
    def from_dict_or_string(
        cls, data: dict[str, Any] | str
    ) -> "InstanceSpecification":
        if isinstance(data, str):
            return cls.from_string(data)
        return cls(
            class_path=data["class_path"], arguments=data.get("arguments", {})
        )

    def load(self) -> Any:
        return load_instance(self.class_path, self.arguments)


@dataclass
class LearnerSpecification(InstanceSpecification):
    name: str
    save_path: Path | None
    load_path: Path | None
    train: bool

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "LearnerSpecification":
        if "save_path" in data:
            save_path = Path(data["save_path"])
        else:
            save_path = None
        if "load_path" in data:
            load_path = Path(data["load_path"])
        else:
            load_path = None

        return cls(
            name=data["name"],
            class_path=data["class_path"],
            arguments=data.get("arguments", {}),
            save_path=save_path,
            load_path=load_path,
            train=data.get("train", True),
        )


@dataclass
class Experiment:
    seed: int
    data_loader: InstanceSpecification
    test_data_loader: InstanceSpecification
    inputs: list[str]
    outputs: list[str]
    transforms: list[InstanceSpecification]
    learners: list[LearnerSpecification]
    metrics: list[InstanceSpecification]
