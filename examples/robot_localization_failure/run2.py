from __future__ import annotations

import logging
import os
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf


def _path_from_env(env_var: str, default: Path) -> Path:
    env_value = os.environ.get(env_var)
    if not env_value:
        return default

    return Path(env_value)


def xdg_config_home() -> Path:
    """Get the XDG configuration home directory.

    This function checks the `XDG_CONFIG_HOME` environment variable and returns
    its value if set. If the variable is not set, it defaults to `~/.config`.
    """
    return _path_from_env("XDG_CONFIG_HOME", Path.home() / ".config")


def load_config(path: Path) -> DictConfig | ListConfig:
    return OmegaConf.load(path) if path.exists() else OmegaConf.create()


DEFAULT_CONFIG = {
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s [%(name)s][%(levelname)s] %(message)s",
    },
}


def load_experiment_config() -> DictConfig | ListConfig:
    config_sources = [
        OmegaConf.create(DEFAULT_CONFIG),
        load_config(xdg_config_home() / "flowcean" / "config.yaml"),
        load_config(Path.cwd() / "config.yaml"),
        OmegaConf.from_cli(),
    ]
    return OmegaConf.unsafe_merge(*config_sources)


logging.basicConfig(**conf.logging)

logger = logging.getLogger(__name__)
logger.info("Logging initialized with configuration")
logger.debug(OmegaConf.to_yaml(conf))
