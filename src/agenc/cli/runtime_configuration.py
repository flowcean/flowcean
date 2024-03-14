from __future__ import annotations

from pathlib import Path
from typing import Any

import ruamel.yaml
from typing_extensions import Self


class RuntimeConfiguration:
    _config: dict[str, Any]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        return key in self._config

    def update(self, new_values: dict[str, Any]) -> None:
        self._config.update(new_values)

    @classmethod
    def load_from_file(cls, path: Path | str) -> Self:
        with Path(path).open() as file:
            yaml = ruamel.yaml.YAML(typ="safe").load(file)
        return cls(yaml)
