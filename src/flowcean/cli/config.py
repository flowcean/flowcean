from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

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


def _load_config(path: Path) -> DictConfig | ListConfig:
    return OmegaConf.load(path) if path.exists() else OmegaConf.create()


DEFAULT_CONFIG = {
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s [%(name)s][%(levelname)s] %(message)s",
    },
}


def load_experiment_config(**script_config: Any) -> DictConfig | ListConfig:
    """Load and merge experiment configuration.

    Merges configuration sources into a single object, with the following
    precedence (lowest to highest): default settings, user config from XDG
    directory, project config (`config.yaml` or specified via CLI), CLI
    arguments, and script-provided overrides.

    Args:
        **script_config: Additional configuration overrides.

    Returns:
        The merged configuration object.
    """
    cli = OmegaConf.from_cli()
    if "conf" in cli:
        experiment = OmegaConf.load(cli.pop("conf"))
    else:
        experiment = _load_config(Path.cwd() / "config.yaml")

    return OmegaConf.unsafe_merge(
        OmegaConf.create(DEFAULT_CONFIG),
        _load_config(xdg_config_home() / "flowcean" / "config.yaml"),
        experiment,
        cli,
        OmegaConf.create(script_config),
    )


def initialize(**kwargs: Any) -> DictConfig | ListConfig:
    """Initialize the experiment environment.

    Loads the configuration and sets up logging according to its settings.

    Args:
        **kwargs: Additional configuration overrides.

    Returns:
        The initialized configuration object.
    """
    conf = load_experiment_config(**kwargs)
    logging.basicConfig(**conf.logging)
    return conf
