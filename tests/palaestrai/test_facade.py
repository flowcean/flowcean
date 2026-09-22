"""Tests for the optional PalaestrAI facade."""

import pytest

pytest.importorskip("harl")
pytest.importorskip("palaestrai")

from flowcean import palaestrai
from flowcean.palaestrai.sac_learner import SACLearner
from flowcean.palaestrai.sac_model import SACModel


def test_palaestrai_facade_exports_sac_types() -> None:
    assert palaestrai.__all__ == ("SACLearner", "SACModel")
    assert palaestrai.SACLearner is SACLearner
    assert palaestrai.SACModel is SACModel
