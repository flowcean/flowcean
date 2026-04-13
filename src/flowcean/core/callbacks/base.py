"""Core callback protocols and callback manager.

Example:
    >>> from flowcean.core.callbacks import (
    ...     CallbackManager,
    ...     LoggingCallback,
    ...     RichCallback,
    ... )
    >>> manager = CallbackManager([RichCallback(), LoggingCallback()])
    >>> manager.on_learning_start(learner)
"""

from abc import abstractmethod
from typing import Any, Protocol

from flowcean.core.model import Model
from flowcean.core.named import Named


class LearnerCallback(Protocol):
    """Protocol for learner callbacks."""

    @abstractmethod
    def on_learning_start(
        self,
        learner: Named,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Called when learning starts."""

    @abstractmethod
    def on_learning_progress(
        self,
        learner: Named,
        progress: float | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Called during learning with progress updates."""

    @abstractmethod
    def on_learning_end(
        self,
        learner: Named,
        model: Model,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Called when learning completes successfully."""

    @abstractmethod
    def on_learning_error(
        self,
        learner: Named,
        error: Exception,
    ) -> None:
        """Called if learning fails with an error."""


class CallbackManager:
    """Manage multiple callbacks through a single interface."""

    def __init__(self, callbacks: list[LearnerCallback] | None = None) -> None:
        """Initialize the callback manager."""
        self.callbacks = callbacks or []

    def on_learning_start(
        self,
        learner: Named,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Notify all callbacks that learning has started."""
        for callback in self.callbacks:
            callback.on_learning_start(learner, context)

    def on_learning_progress(
        self,
        learner: Named,
        progress: float | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Notify all callbacks of learning progress."""
        for callback in self.callbacks:
            callback.on_learning_progress(learner, progress, metrics)

    def on_learning_end(
        self,
        learner: Named,
        model: Model,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Notify all callbacks that learning has completed."""
        for callback in self.callbacks:
            callback.on_learning_end(learner, model, metrics)

    def on_learning_error(
        self,
        learner: Named,
        error: Exception,
    ) -> None:
        """Notify all callbacks that learning has failed."""
        for callback in self.callbacks:
            callback.on_learning_error(learner, error)


class SilentCallback:
    """A callback that intentionally produces no output."""

    def on_learning_start(
        self,
        learner: Named,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Do nothing when learning starts."""

    def on_learning_progress(
        self,
        learner: Named,
        progress: float | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Do nothing during learning progress."""

    def on_learning_end(
        self,
        learner: Named,
        model: Model,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Do nothing when learning ends."""

    def on_learning_error(
        self,
        learner: Named,
        error: Exception,
    ) -> None:
        """Do nothing when learning errors."""
