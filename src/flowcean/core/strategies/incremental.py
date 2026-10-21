import polars as pl

from flowcean.core.environment.incremental import IncrementalEnvironment
from flowcean.core.learner import SupervisedIncrementalLearner
from flowcean.core.model import Model
from flowcean.core.transform import (
    Identity,
    Transform,
    fit_incremental,
    inverse,
    validate_incremental_fit,
)


def learn_incremental[LearnedModel: Model[pl.LazyFrame, pl.LazyFrame]](
    environment: IncrementalEnvironment[pl.LazyFrame],
    learner: SupervisedIncrementalLearner[
        pl.LazyFrame, pl.LazyFrame, LearnedModel
    ],
    inputs: list[str],
    outputs: list[str],
    input_transform: Transform[pl.LazyFrame, pl.LazyFrame] | None = None,
    output_transform: Transform[pl.LazyFrame, pl.LazyFrame] | None = None,
) -> LearnedModel:
    """Fit supported transforms and a mutable tabular model on successive batches.

    Both transforms must support incremental fitting. Unsupported transforms
    are rejected before observing data or mutating the learner.
    """
    if input_transform is None:
        input_transform = Identity()
    if output_transform is None:
        output_transform = Identity()
    validate_incremental_fit(input_transform)
    validate_incremental_fit(output_transform)

    model = None
    output_inverse = None
    for data in environment:
        input_features = data.select(inputs)
        output_features = data.select(outputs)
        fit_incremental(input_transform, input_features)
        fit_incremental(output_transform, output_features)
        output_inverse = inverse(output_transform)
        model = learner.learn_incremental(
            input_transform(input_features),
            output_transform(output_features),
        )
    if model is None or output_inverse is None:
        message = "No data found in environment."
        raise ValueError(message)
    model.pre_transform = input_transform | model.pre_transform
    model.post_transform = model.post_transform | output_inverse
    return model
