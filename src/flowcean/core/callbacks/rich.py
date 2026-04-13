"""Rich console callbacks for learner progress feedback.

Example:
    >>> from flowcean.core.callbacks import RichCallback
    >>> from flowcean.sklearn import RandomForestRegressorLearner
    >>> learner = RandomForestRegressorLearner(callbacks=[RichCallback()])
"""

from typing import Any

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text

from flowcean.core.callbacks.base import LearnerCallback
from flowcean.core.model import Model
from flowcean.core.named import Named


class RichCallback(LearnerCallback):
    """Rich console callback with adaptive progress display."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        show_metrics: bool = True,
    ) -> None:
        """Initialize the Rich callback."""
        self.console = console or Console()
        self.show_metrics = show_metrics
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._has_progress_updates = False

    def on_learning_start(
        self,
        learner: Named,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Display learning start message."""
        message = Text()
        message.append("⠿ ", style="bold blue")
        message.append(f"[{learner.name}] Learning", style="bold blue")
        if context and self.show_metrics:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            message.append(f" ({context_str})", style="dim")
        message.append("...", style="dim")

        self.console.print(message)
        self._has_progress_updates = False

    def on_learning_progress(
        self,
        learner: Named,
        progress: float | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Update progress display with current progress and metrics."""
        if not self._has_progress_updates:
            self._has_progress_updates = True
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=self.console,
            )

            description = f"[{learner.name}] Learning"
            self._task_id = self._progress.add_task(
                description,
                total=100 if progress else None,
            )

            self._live = Live(
                self._progress,
                console=self.console,
                refresh_per_second=10,
            )
            self._live.start()

        if self._progress and self._task_id is not None:
            description = f"[{learner.name}] Learning"
            if metrics and self.show_metrics:
                metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
                description += f" ({metrics_str})"

            if progress is not None:
                if self._progress.tasks[self._task_id].total is None:
                    self._progress.update(self._task_id, total=100)
                self._progress.update(
                    self._task_id,
                    completed=progress * 100,
                    description=description,
                )
            else:
                self._progress.update(
                    self._task_id,
                    description=description,
                    advance=0.1,
                )

            if self._live:
                self._live.refresh()

    def on_learning_end(
        self,
        learner: Named,
        model: Model,  # noqa: ARG002
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Display learning completion message."""
        if self._progress and self._task_id is not None:
            if self._progress.tasks[self._task_id].total is None:
                self._progress.update(self._task_id, total=100)
            self._progress.update(self._task_id, completed=100)

        if self._live:
            self._live.stop()

        message = Text(
            f"✓ [{learner.name}] Learning finished",
            style="bold green",
        )
        if metrics and self.show_metrics:
            metrics_str = ", ".join(f"{k}={v}" for k, v in metrics.items())
            message.append(f" ({metrics_str})", style="green")

        self.console.print(message)
        self._live = None
        self._progress = None
        self._task_id = None
        self._has_progress_updates = False

    def on_learning_error(
        self,
        learner: Named,
        error: Exception,
    ) -> None:
        """Display error message."""
        if self._live:
            self._live.stop()

        self.console.print(
            f"✗ [{learner.name}] Learning failed: {error}",
            style="bold red",
        )

        self._live = None
        self._progress = None
        self._task_id = None
        self._has_progress_updates = False


class RichSpinnerCallback(LearnerCallback):
    """Simplified Rich callback with just a spinner."""

    def __init__(
        self,
        console: Console | None = None,
    ) -> None:
        """Initialize the Rich spinner callback."""
        self.console = console or Console()
        self._live: Live | None = None

    def on_learning_start(
        self,
        learner: Named,
        context: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        """Display learning start message with spinner."""
        text = Text()
        text.append("⠋", style="bold blue")
        text.append(f" [{learner.name}] Learning", style="bold")

        self._live = Live(text, console=self.console, refresh_per_second=10)
        self._live.start()

    def on_learning_progress(
        self,
        learner: Named,
        progress: float | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Update spinner state."""

    def on_learning_end(
        self,
        learner: Named,
        model: Model,  # noqa: ARG002
        metrics: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> None:
        """Display learning completion message."""
        if self._live:
            self._live.stop()

        self.console.print(
            f"✓ [{learner.name}] Learning finished",
            style="bold green",
        )
        self._live = None

    def on_learning_error(
        self,
        learner: Named,
        error: Exception,
    ) -> None:
        """Display error message."""
        if self._live:
            self._live.stop()

        self.console.print(
            f"✗ [{learner.name}] Learning failed: {error}",
            style="bold red",
        )
        self._live = None
