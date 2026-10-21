"""Helpers for integrating callbacks into learners."""

from collections.abc import Callable
from typing import Any

from flowcean.core.callbacks.base import CallbackManager, LearnerCallback
from flowcean.core.named import Named


def create_callback_manager(
    callbacks: list[LearnerCallback] | LearnerCallback | None,
) -> CallbackManager:
    """Create a CallbackManager from supported callback inputs.

    Args:
        callbacks: Callbacks to manage. Can be:
            - None: Uses no callbacks
            - Single callback: Wraps in a list
            - List of callbacks: Uses directly

    Returns:
        CallbackManager instance.
    """
    if callbacks is None:
        return CallbackManager([])
    if isinstance(callbacks, list):
        return CallbackManager(callbacks)
    return CallbackManager([callbacks])


def get_default_callbacks() -> list[LearnerCallback]:
    """Return the default callbacks for learners.

    Returns:
        An empty list so learners stay silent unless callbacks are provided.
    """
    return []


class CallbackMixin:
    """Mixin to add callback support to learners."""

    # Concrete learners initialize this mixin requirement in their constructors.
    callback_manager: CallbackManager  # pyright: ignore[reportUninitializedInstanceVariable]

    def _setup_callbacks(
        self,
        callbacks: list[LearnerCallback] | LearnerCallback | None = None,
        use_default: bool = True,
    ) -> CallbackManager:
        """Set up the callback manager."""
        if callbacks is None and use_default:
            callbacks = get_default_callbacks()
        return create_callback_manager(callbacks)

    def _wrap_learning_with_callbacks[LearnedModel: Named](
        self,
        learner: Named,
        learning_fn: Callable[[], LearnedModel],
        context: dict[str, Any] | None = None,
    ) -> LearnedModel:
        """Wrap a learning function with callback invocations."""
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
