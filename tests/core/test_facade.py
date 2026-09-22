"""Tests for the core public facade."""

from flowcean import core
from flowcean.core.metric import ActiveMetric


def test_core_facade_exports_active_metric() -> None:
    assert core.ActiveMetric is ActiveMetric
    assert "ActiveMetric" in core.__all__
