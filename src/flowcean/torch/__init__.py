from flowcean._optional import raise_for_missing_optional_dependency

try:
    from .dataset import TorchDataset
    from .lightning_learner import LightningLearner
    from .linear_regression import LinearRegression
    from .model import PyTorchModel
    from .multi_layer_perceptron import MultilayerPerceptron
except ModuleNotFoundError as error:
    raise_for_missing_optional_dependency(
        error,
        extra="torch",
        module="flowcean.torch",
        missing_dependencies={"lightning", "torch"},
    )

__all__ = [
    "LightningLearner",
    "LinearRegression",
    "MultilayerPerceptron",
    "PyTorchModel",
    "TorchDataset",
]
