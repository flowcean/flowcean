from __future__ import annotations

import logging
import logging.config

from agenc.cli.runtime_configuration import get_configuration


def initialize(level: logging._Level | None = None) -> None:
    """Initialize logger from RuntimeConfiguration."""
    logging.config.dictConfig(get_configuration()["logging"])
    logging.debug("initialized logging from RuntimeConfiguration")

    if level is not None:
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(level)
            get_configuration()["logging"].update(
                {
                    "loggers": {
                        name: {
                            "level": level,
                        }
                    }
                }
            )
