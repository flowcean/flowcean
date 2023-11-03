from dataclasses import dataclass

from .learner import Learner
from .metadata import Metadata
from .metric import Metric
from .transform import Transform


@dataclass
class Experiment:
    seed: int
    metadata: Metadata
    inputs: list[str]
    outputs: list[str]
    train_test_split: float
    transforms: list[Transform]
    learner: Learner
    metrics: list[Metric]
