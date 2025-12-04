"""Callback system for learner progress feedback.

This module provides a standardized way to get progress feedback from learners
during the learning process. Callbacks can be used to log progress, display
progress bars, or integrate with monitoring systems.

Example:
    >>> from flowcean.core.callbacks import RichCallback
    >>> from flowcean.sklearn import RandomForestRegressorLearner
    >>>
    >>> learner = RandomForestRegressorLearner(callbacks=[RichCallback()])
    >>> model = learner.learn(inputs, outputs)
    # Shows: [RandomForestRegressorLearner] Learning ⠋ (with spinner)
"""

from abc import abstractmethod
from typing import Any, Protocol

from .model import Model
from .named import Named


class LearnerCallback(Protocol):
    """Protocol for learner callbacks.

    Callbacks receive notifications about the learning process and can
    provide feedback to users or logging systems.
    """

    @abstractmethod
    def on_learning_start(
        self,
        learner: Named,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Called when learning starts.

        Args:
            learner: The learner that is starting to learn.
            context: Optional context information (e.g., data shapes,
                hyperparams).
        """

    @abstractmethod
    def on_learning_progress(
        self,
        learner: Named,
        progress: float | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Called during learning with progress updates.

        Args:
            learner: The learner that is learning.
            progress: Progress value between 0.0 and 1.0, or None if
                indeterminate.
            metrics: Optional metrics (e.g., loss, accuracy, iteration number).
        """

    @abstractmethod
    def on_learning_end(
        self,
        learner: Named,
        model: Model,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Called when learning completes successfully.

        Args:
            learner: The learner that finished learning.
            model: The trained model.
            metrics: Optional final metrics.
        """

    @abstractmethod
    def on_learning_error(
        self,
        learner: Named,
        error: Exception,
    ) -> None:
        """Called if learning fails with an error.

        Args:
            learner: The learner that encountered an error.
            error: The exception that was raised.
        """


class CallbackManager:
    """Manages multiple callbacks and dispatches events to them.

    This class simplifies working with multiple callbacks by providing
    a single interface that forwards calls to all registered callbacks.

    Example:
        >>> from flowcean.core.callbacks import (
        ...     CallbackManager,
        ...     RichCallback,
        ...     LoggingCallback,
        ... )
        >>> manager = CallbackManager([RichCallback(), LoggingCallback()])
        >>> manager.on_learning_start(learner)
    """

    def __init__(self, callbacks: list[LearnerCallback] | None = None) -> None:
        """Initialize the callback manager.

        Args:
            callbacks: List of callbacks to manage. Defaults to empty list.
        """
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
    """A callback that does nothing.

    Useful for disabling progress feedback or as a base class.
    """

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


__all__ = [
    "CallbackManager",
    "LearnerCallback",
    "SilentCallback",
]
