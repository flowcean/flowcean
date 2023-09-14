"""AGenC runtime config"""
from __future__ import annotations

import logging
import logging.config
import os
from copy import deepcopy
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, TextIO, Union

import appdirs
import ruamel.yaml

LOG = logging.getLogger(__name__)

CONFIG_FILE_NAME = "agenc-runtime-conf.yml"
CONFIG_FILE_PATHS = [
    f"{appdirs.user_config_dir('agenc')}/{CONFIG_FILE_NAME}",
    f"{os.getcwd()}/{CONFIG_FILE_NAME}",
]
DEFAULT_BASE_CONFIG = {
    "logging": {
        "version": 1,
        "formatters": {
            "simple": {
                "format": "%(asctime)s [%(name)s][%(levelname)s] %(message)s"
            },
            "debug": {
                "format": (
                    "%(asctime)s [%(name)s (%(process)d)][%(levelname)s] "
                    "%(message)s (%(module)s.%(funcName)s in %(filename)s"
                    ":%(lineno)d)"
                )
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "console-debug": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "debug",
                "stream": "ext://sys.stdout",
            },
            "logfile": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "simple",
                "filename": "agenc.log",
                "mode": "w",
            },
            "logfile-debug": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "debug",
                "filename": "agenc_debug.log",
                "mode": "w",
            },
        },
        "loggers": {"agenc": {"level": "WARNING"}},
        "root": {"level": "ERROR", "handlers": ["console", "logfile-debug"]},
    }
}


class _RuntimeConfig:
    """Application-wide runtime configuration.

    This singleton class provides an application-wide runtime
    configuration and transparently hides all sources from the rest of
    the application.
    """

    DEFAULT_CONFIG = deepcopy(DEFAULT_BASE_CONFIG)
    _instance = None

    def __init__(self):
        self._config_file_path = None
        self.config_search_path = CONFIG_FILE_PATHS

        # The loaded configuration is what RuntimeConfig.load gave us.
        # It remeains immutable after loading.
        self._loaded_configuration = dict()

    def _get(self, key: str, default=None, exception=None) -> Any:
        """Retrieves a config key.

        Retrieves any config key; if not set, it queries the config
        dictionary; if it isn't present there, it returns the given
        default value. It also sets the value in the current object
        as side-effect.

        """
        lkey = f"_{key}"
        if lkey not in self.__dict__:
            try:
                self.__dict__[lkey] = self._loaded_configuration[key]
            except KeyError:
                if default:
                    self.__dict__[lkey] = default
                else:
                    self.__dict__[lkey] = _RuntimeConfig.DEFAULT_CONFIG[key]
                if exception:
                    raise KeyError(exception)
        return self.__dict__[lkey]

    def reset(self):
        """Resets the runtime configuration to empty state."""
        for key in list(self._loaded_configuration.keys()) + list(
            _RuntimeConfig.DEFAULT_CONFIG.keys()
        ):
            try:
                del self.__dict__[f"_{key}"]
            except KeyError:
                pass
        self._loaded_configuration = dict()
        self._config_file_path = None

    @property
    def logging(self) -> Dict:
        """Configuration of all subsystem loggers.

        Returns
        -------
        dict
            A logging configuration that can be fed into
            `logging.DictConfig`.

        """
        return self._get(
            "logging", exception="Sorry, no logging config in the config file."
        )

    def load(
        self, stream_or_dict: Union[dict, TextIO, str, Path, None] = None
    ):
        """Load the configuration from an external source.

        The runtime configuration is initialized from the default
        configuration in ::`_RuntimeConfig.DEFAULT_CONFIG`. This method
        then iterates through the list in
        ::`_RuntimeConfig.CONFIG_FILE_PATHS`, subsequently updating the
        existing configuration with new values found. Finally, the
        given ::`stream_or_dict` parameter is used if present,
        ultimately taking preference over all other values.

        That means that each config file can contain only a portion of
        the overall configuration; it gets updated subsequently.

        Parameters
        ----------
        stream_or_dict: Union[dict, TextIO, str, Path, None]
            Loads the runtime configuration directly from a dictionary
            or as YAML-encoded stream. If no stream is given, the
            default files in ::`.CONFIG_FILE_PATHS` will be tried as
            described.

        """
        if not isinstance(self._loaded_configuration, dict):
            self._loaded_configuration = dict()
        if stream_or_dict is None and self._loaded_configuration:
            # if not stream_or_dict and len(self._loaded_configuration) > 0:
            # Don't load a default config if we already have something;
            # use reset() instead.
            return

        # yml = ruamel.yaml.YAML(typ="safe")
        has_seen_nondefault_config = False
        self._loaded_configuration.update(_RuntimeConfig.DEFAULT_CONFIG)
        yaml = ruamel.yaml.YAML()
        yaml.preserve_quotes = True
        for fil in CONFIG_FILE_PATHS:
            try:
                LOG.debug("Trying to open configuration file: '%s'", fil)
                with open(fil, "r") as fp:
                    deserialized = yaml.load(fp)
                    if not isinstance(deserialized, dict):
                        LOG.warning(
                            (
                                "The contents of '%s' could not be"
                                " deserialized to dict, skipping it."
                            ),
                            fil,
                        )
                        continue
                    self._loaded_configuration.update(deserialized)
                    self._config_file_path = fil
                    has_seen_nondefault_config = True
            except IOError:
                continue

        if isinstance(stream_or_dict, dict):
            self._loaded_configuration.update(stream_or_dict)
            self._config_file_path = "(dict)"
            return

        if isinstance(stream_or_dict, str):
            stream_or_dict = Path(stream_or_dict)
        if isinstance(stream_or_dict, Path):
            try:
                stream_or_dict = open(stream_or_dict, "r")
            except OSError:
                LOG.warning(
                    (
                        "Failed to load runtime configuration from file at"
                        " '%s', ignoring"
                    ),
                    stream_or_dict,
                )
        if stream_or_dict is not None:
            try:
                deserialized = yaml.load(stream_or_dict)
                if not isinstance(deserialized, dict):
                    raise TypeError
                self._loaded_configuration.update(deserialized)
                try:
                    self._config_file_path = stream_or_dict.name
                except AttributeError:
                    self._config_file_path = str(stream_or_dict)
                has_seen_nondefault_config = True
            except TypeError:
                LOG.warning(
                    (
                        "Failed to load runtime configuration from stream"
                        " at '%s', ignoring it"
                    ),
                    repr(stream_or_dict),
                )
            finally:
                if isinstance(stream_or_dict, TextIOWrapper):
                    stream_or_dict.close()

        if not has_seen_nondefault_config:
            LOG.info(
                "No runtime configuration given, loaded built-in default."
            )
            self._config_file_path = "(DEFAULT)"

    def to_dict(self) -> Dict:
        return {key: self._get(key) for key in _RuntimeConfig.DEFAULT_CONFIG}

    def __str__(self):
        return f"<RuntimeConfig id=0x{id(self)}> at {self._config_file_path}"

    def __repr__(self):
        return str(self.to_dict())


def RuntimeConfig():
    if _RuntimeConfig._instance is None:
        _RuntimeConfig._instance = _RuntimeConfig()
        try:
            _RuntimeConfig._instance.load()
        except FileNotFoundError:
            from copy import deepcopy

            _RuntimeConfig._instance._loaded_configuration = deepcopy(
                _RuntimeConfig.DEFAULT_CONFIG
            )
    return _RuntimeConfig._instance


def init_logger(verbose: int):
    """Init logger with config from either RuntimeConfig or a default."""
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    log_level = levels[verbose if verbose < len(levels) else len(levels) - 1]

    try:
        logging.config.dictConfig(RuntimeConfig().logging)
        logging.debug(
            "Initialized logging from RuntimeConfig(%s)",
            RuntimeConfig(),
        )
    except (KeyError, ValueError) as err:
        logging.basicConfig(level=log_level)
        logging.warning(
            "Could not load logging config (%s), continuing with defaults.",
            err,
        )

    if verbose != 0:
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(log_level)
            RuntimeConfig().logging["loggers"].update(
                {name: {"level": str(logging._levelToName[log_level])}}
            )
