"""Helper utilities for integrating callbacks into learners.

This module provides utilities to make it easy for learners to support
the callback system without modifying the Protocol definitions.
"""

from collections.abc import Callable
from typing import Any

from .callbacks import CallbackManager, LearnerCallback
from .callbacks_rich import RichCallback
from .model import Model
from .named import Named


def create_callback_manager(
    callbacks: list[LearnerCallback] | LearnerCallback | None,
) -> CallbackManager:
    """Create a CallbackManager from various input formats.

    Args:
        callbacks: Callbacks to manage. Can be:
            - None: Uses default callbacks (RichCallback)
            - Single callback: Wraps in a list
            - List of callbacks: Uses directly

    Returns:
        CallbackManager instance.

    Example:
        >>> manager = create_callback_manager(RichCallback())
        >>> manager.on_learning_start(learner)
    """
    if callbacks is None:
        callbacks = get_default_callbacks()
    if isinstance(callbacks, list):
        return CallbackManager(callbacks)
    return CallbackManager([callbacks])


def get_default_callbacks() -> list[LearnerCallback]:
    """Get the default callbacks for learners.

    Returns a list with RichCallback as the default.

    Returns:
        List containing default callbacks.
    """
    return [RichCallback()]


class CallbackMixin:
    """Mixin to add callback support to learners.

    This mixin provides common functionality for managing callbacks
    in learner implementations.

    Example:
        >>> class MyLearner(CallbackMixin):
        ...     def __init__(self, callbacks=None):
        ...         self.callback_manager = self._setup_callbacks(callbacks)
        ...
        ...     def learn(self, inputs, outputs):
        ...         self.callback_manager.on_learning_start(self)
        ...         # ... learning logic ...
        ...         self.callback_manager.on_learning_end(self, model)
        ...         return model
    """

    callback_manager: CallbackManager

    def _setup_callbacks(
        self,
        callbacks: list[LearnerCallback] | LearnerCallback | None = None,
        use_default: bool = True,  # noqa: FBT001, FBT002
    ) -> CallbackManager:
        """Set up the callback manager.

        Args:
            callbacks: User-provided callbacks. If None and use_default is
                True, uses default callbacks.
            use_default: Whether to use default callbacks when callbacks is
                None.

        Returns:
            Configured CallbackManager.
        """
        if callbacks is None and use_default:
            callbacks = get_default_callbacks()
        return create_callback_manager(callbacks)

    def _wrap_learning_with_callbacks(
        self,
        learner: Named,
        learning_fn: Callable[[], Model],
        context: dict[str, Any] | None = None,
    ) -> Model:
        """Wrap a learning function with callback invocations.

        This helper automatically handles calling callbacks before, during,
        and after learning, including error handling.

        Args:
            learner: The learner instance (usually self).
            learning_fn: The actual learning function to call (returns Model).
            context: Optional context to pass to on_learning_start.

        Returns:
            The trained model.

        Raises:
            Any exception raised by learning_fn.

        Example:
            >>> def learn(self, inputs, outputs):
            ...     def _learn():
            ...         # actual learning logic
            ...         return model
            ...
            ...     return self._wrap_learning_with_callbacks(self, _learn)
        """
        callback_manager = self.callback_manager

        callback_manager.on_learning_start(learner, context)
        try:
            model = learning_fn()
            callback_manager.on_learning_end(learner, model)
        except Exception as e:
            callback_manager.on_learning_error(learner, e)
            raise
        else:
            return model


__all__ = [
    "CallbackMixin",
    "create_callback_manager",
    "get_default_callbacks",
]
