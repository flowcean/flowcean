from __future__ import annotations

import logging
import logging.config
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import ruamel.yaml

logger = logging.getLogger(__name__)

DEFAULT_BASE_CONFIG = {
    "logging": {
        "version": 1,
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(name)s][%(levelname)s] %(message)s",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["default"],
        },
    }
}


class RuntimeConfiguration:
    _config: MutableMapping[str, Any]

    def __init__(self) -> None:
        self._config = deepcopy(DEFAULT_BASE_CONFIG)

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def update(self, new_values: Mapping[str, Any]) -> None:
        self._config.update(new_values)


_configuration = RuntimeConfiguration()


def get_configuration() -> RuntimeConfiguration:
    return _configuration


def load_from_file(path: Path) -> None:
    with open(path) as file:
        yaml = ruamel.yaml.YAML(typ="safe").load(file)
    get_configuration().update(yaml)
