"""Logging callbacks for learner progress feedback.

Example:
    >>> from flowcean.core.callbacks import LoggingCallback
    >>> callback = LoggingCallback()
"""

import logging
from typing import Any

from flowcean.core.callbacks.base import LearnerCallback
from flowcean.core.model import Model
from flowcean.core.named import Named


class LoggingCallback(LearnerCallback):
    """Standard Python logging callback."""

    def __init__(
        self,
        logger: logging.Logger | None = None,
        level_start: int = logging.INFO,
        level_progress: int = logging.DEBUG,
        level_end: int = logging.INFO,
        level_error: int = logging.ERROR,
    ) -> None:
        """Initialize the logging callback."""
        self.logger = logger or logging.getLogger("flowcean.learner")
        self.level_start = level_start
        self.level_progress = level_progress
        self.level_end = level_end
        self.level_error = level_error

    def on_learning_start(
        self,
        learner: Named,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Log learning start event."""
        message = f"[{learner.name}] Learning started"
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            message += f" ({context_str})"
        self.logger.log(self.level_start, message)

    def on_learning_progress(
        self,
        learner: Named,
        progress: float | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Log learning progress."""
        message = f"[{learner.name}] Learning in progress"
        if progress is not None:
            message += f" ({progress * 100:.1f}%)"
        if metrics:
            metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
            message += f" - {metrics_str}"
        self.logger.log(self.level_progress, message)

    def on_learning_end(
        self,
        learner: Named,
        model: Model,  # noqa: ARG002
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Log learning completion."""
        message = f"[{learner.name}] Learning finished"
        if metrics:
            metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
            message += f" ({metrics_str})"
        self.logger.log(self.level_end, message)

    def on_learning_error(
        self,
        learner: Named,
        error: Exception,
    ) -> None:
        """Log learning error."""
        self.logger.log(
            self.level_error,
            "[%s] Learning failed: %s",
            learner.name,
            error,
            exc_info=error,
        )
