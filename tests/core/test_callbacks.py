"""Tests for the learner callback system."""

from unittest.mock import MagicMock

import polars as pl
import pytest

from flowcean.core import (
    CallbackManager,
    LearnerCallback,
    LoggingCallback,
    Model,
    RichCallback,
    SilentCallback,
    create_callback_manager,
)
from flowcean.core.named import Named


class MockLearner(Named):
    """Mock learner for testing."""

    def __init__(self, name: str = "MockLearner") -> None:
        self._name = name


class MockModel(Model):
    """Mock model for testing."""

    def _predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        return input_features


class TestCallbackManager:
    """Test the CallbackManager class."""

    def test_create_empty_manager(self) -> None:
        """Test creating a manager with no callbacks."""
        manager = CallbackManager([])
        assert len(manager.callbacks) == 0

    def test_create_manager_with_callbacks(self) -> None:
        """Test creating a manager with callbacks."""
        callback1 = SilentCallback()
        callback2 = SilentCallback()
        manager = CallbackManager([callback1, callback2])
        assert len(manager.callbacks) == 2

    def test_on_learning_start(self) -> None:
        """Test that on_learning_start is called on all callbacks."""
        callback1 = MagicMock(spec=LearnerCallback)
        callback2 = MagicMock(spec=LearnerCallback)
        manager = CallbackManager([callback1, callback2])

        learner = MockLearner()
        context = {"test": "value"}
        manager.on_learning_start(learner, context)

        callback1.on_learning_start.assert_called_once_with(learner, context)
        callback2.on_learning_start.assert_called_once_with(learner, context)

    def test_on_learning_progress(self) -> None:
        """Test that on_learning_progress is called on all callbacks."""
        callback1 = MagicMock(spec=LearnerCallback)
        callback2 = MagicMock(spec=LearnerCallback)
        manager = CallbackManager([callback1, callback2])

        learner = MockLearner()
        progress = 0.5
        metrics = {"loss": 0.1}
        manager.on_learning_progress(learner, progress, metrics)

        callback1.on_learning_progress.assert_called_once_with(
            learner,
            progress,
            metrics,
        )
        callback2.on_learning_progress.assert_called_once_with(
            learner,
            progress,
            metrics,
        )

    def test_on_learning_end(self) -> None:
        """Test that on_learning_end is called on all callbacks."""
        callback1 = MagicMock(spec=LearnerCallback)
        callback2 = MagicMock(spec=LearnerCallback)
        manager = CallbackManager([callback1, callback2])

        learner = MockLearner()
        model = MockModel()
        metrics = {"final_loss": 0.05}
        manager.on_learning_end(learner, model, metrics)

        callback1.on_learning_end.assert_called_once_with(
            learner,
            model,
            metrics,
        )
        callback2.on_learning_end.assert_called_once_with(
            learner,
            model,
            metrics,
        )

    def test_on_learning_error(self) -> None:
        """Test that on_learning_error is called on all callbacks."""
        callback1 = MagicMock(spec=LearnerCallback)
        callback2 = MagicMock(spec=LearnerCallback)
        manager = CallbackManager([callback1, callback2])

        learner = MockLearner()
        error = ValueError("Test error")
        manager.on_learning_error(learner, error)

        callback1.on_learning_error.assert_called_once_with(learner, error)
        callback2.on_learning_error.assert_called_once_with(learner, error)


class TestCreateCallbackManager:
    """Test the create_callback_manager helper function."""

    def test_create_from_none(self) -> None:
        """Test creating manager from None returns default callbacks."""
        manager = create_callback_manager(None)
        assert isinstance(manager, CallbackManager)
        # Default behavior: returns RichCallback when None is passed
        assert len(manager.callbacks) == 1
        assert isinstance(manager.callbacks[0], RichCallback)

    def test_create_from_single_callback(self) -> None:
        """Test creating manager from a single callback."""
        callback = SilentCallback()
        manager = create_callback_manager(callback)
        assert isinstance(manager, CallbackManager)
        assert len(manager.callbacks) == 1
        assert manager.callbacks[0] is callback

    def test_create_from_list(self) -> None:
        """Test creating manager from a list of callbacks."""
        callbacks: list[LearnerCallback] = [
            SilentCallback(),
            SilentCallback(),
        ]
        manager = create_callback_manager(callbacks)
        assert isinstance(manager, CallbackManager)
        assert len(manager.callbacks) == 2


class TestSilentCallback:
    """Test the SilentCallback class."""

    def test_silent_callback_does_nothing(self) -> None:
        """Test that SilentCallback methods don't raise errors."""
        callback = SilentCallback()
        learner = MockLearner()
        model = MockModel()

        # Should not raise any errors
        callback.on_learning_start(learner, {"test": "value"})
        callback.on_learning_progress(learner, 0.5, {"loss": 0.1})
        callback.on_learning_end(learner, model, {"final_loss": 0.05})
        callback.on_learning_error(learner, ValueError("test"))


class TestLoggingCallback:
    """Test the LoggingCallback class."""

    def test_logging_callback_initialization(self) -> None:
        """Test LoggingCallback initialization."""
        callback = LoggingCallback()
        assert callback.logger is not None

    def test_logging_callback_with_custom_logger(self) -> None:
        """Test LoggingCallback with custom logger."""
        import logging

        custom_logger = logging.getLogger("test_logger")
        callback = LoggingCallback(logger=custom_logger)
        assert callback.logger is custom_logger


class TestRichCallback:
    """Test the RichCallback class."""

    def test_rich_callback_initialization(self) -> None:
        """Test RichCallback initialization."""
        callback = RichCallback()
        assert callback.console is not None
        assert callback.show_metrics is True

    def test_rich_callback_without_metrics(self) -> None:
        """Test RichCallback with metrics disabled."""
        callback = RichCallback(show_metrics=False)
        assert callback.show_metrics is False


class TestCallbackIntegration:
    """Integration tests for callbacks with learners."""

    def test_callback_with_sklearn_learner(self) -> None:
        """Test using callbacks with a real sklearn learner."""
        import numpy as np

        from flowcean.sklearn import RandomForestRegressorLearner

        # Create sample data
        rng = np.random.default_rng(42)
        n_samples = 100
        x = rng.standard_normal((n_samples, 3))
        y = x[:, 0] + x[:, 1] - x[:, 2] + rng.standard_normal(n_samples) * 0.1

        inputs = pl.LazyFrame(
            {
                "x1": x[:, 0],
                "x2": x[:, 1],
                "x3": x[:, 2],
            },
        )
        outputs = pl.LazyFrame({"y": y})

        # Create learner with silent callback
        learner = RandomForestRegressorLearner(
            n_estimators=10,
            callbacks=SilentCallback(),
        )

        # Should complete without errors
        model = learner.learn(inputs, outputs)
        assert model is not None

    def test_callback_error_handling(self) -> None:
        """Test that callbacks receive error notifications."""
        from flowcean.sklearn import RandomForestRegressorLearner

        # Create a mock callback that tracks errors
        error_callback = MagicMock(spec=LearnerCallback)

        learner = RandomForestRegressorLearner(
            n_estimators=10,
            callbacks=error_callback,
        )

        # Create invalid data that will cause an error
        inputs = pl.LazyFrame({"x": [1, 2, 3]})
        outputs = pl.LazyFrame({"y": ["invalid", "data", "types"]})

        # Should raise an error and call on_learning_error
        with pytest.raises(
            (ValueError, TypeError),
            match=r"could not convert|not supported",
        ):
            learner.learn(inputs, outputs)

        # Verify error callback was called
        assert error_callback.on_learning_error.called
