import polars as pl
from flowcean.core.environment.active import ActiveEnvironment
from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.transform import Transform, Identity

from flowcean.core.model import ModelWithTransform

def deploy(
    environment: ActiveEnvironment | IncrementalEnvironment,
    model: ModelWithTransform,
    input_transforms: Transform = Identity(),
    output_transforms: Transform  = Identity()
) -> None:
    """
    Deploy a trained model to a custom environment

    Args:
        environment: custom system environment
        model: the trained model
        input_transforms: system specific transforms for model input
        output_transforms: system specific transforms for system input

    Returns:
        -
    """
    observation = environment.observe()
    output = model.predict(input_transforms.apply(observation))

    if isinstance(environment, ActiveEnvironment):
        environment.act(output_transforms.apply(output).collect())