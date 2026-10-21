import polars as pl

from flowcean.core.environment.active import ActiveEnvironment
from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.model import Predictor
from flowcean.core.transform import Identity, Transform


def deploy(
    environment: ActiveEnvironment[pl.LazyFrame, pl.DataFrame]
    | IncrementalEnvironment[pl.LazyFrame],
    model: Predictor[pl.LazyFrame, pl.LazyFrame],
    input_transforms: Transform[pl.LazyFrame, pl.LazyFrame] | None = None,
    output_transforms: Transform[pl.LazyFrame, pl.LazyFrame] | None = None,
) -> None:
    """Make one tabular prediction and, for an active environment, act once.

    The action is collected eagerly. This is not a streaming deployment loop.

    Args:
        environment: custom system environment
        model: the trained model
        input_transforms: system specific transforms for model input
        output_transforms: system specific transforms for system input
    """
    if input_transforms is None:
        input_transforms = Identity()
    if output_transforms is None:
        output_transforms = Identity()

    observation = environment.observe()
    output = model.predict(input_transforms.apply(observation))

    if isinstance(environment, ActiveEnvironment):
        environment.act(output_transforms.apply(output).collect())
