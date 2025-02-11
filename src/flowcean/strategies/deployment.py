from flowcean.core.environment.active import ActiveEnvironment
from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.model import ModelWithTransform
from flowcean.core.transform import Identity, Transform


def deploy(
    environment: ActiveEnvironment | IncrementalEnvironment,
    model: ModelWithTransform,
    input_transforms: Transform | None = None,
    output_transforms: Transform | None = None,
) -> None:
    """Deploy a trained model to a custom environment.

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
