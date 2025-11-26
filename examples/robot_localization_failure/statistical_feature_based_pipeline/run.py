#!/usr/bin/env python

import logging
from pathlib import Path

import flowcean.cli
from flowcean.core import learn_offline
from flowcean.core.strategies.offline import evaluate_offline
from flowcean.polars.environments.dataframe import DataFrame
from flowcean.sklearn.metrics.classification import (
    Accuracy,
    ClassificationReport,
    FBetaScore,
    PrecisionScore,
    Recall,
)
from flowcean.xgboost.learner import XGBoostClassifierLearner
from statistical_feature_based_pipeline.training import collect_data

logger = logging.getLogger(__name__)


def main() -> None:
    config = flowcean.cli.initialize()

    samples_train, samples_eval = collect_data(config)

    inputs = samples_train.drop(
        ["is_delocalized", "position_error", "heading_error"],
    ).columns
    outputs = samples_train.select(["is_delocalized"]).columns

    xgb = XGBoostClassifierLearner()
    learners = [
        xgb,
    ]

    models = []
    for learner in learners:
        print(f"Training model: {learner.name}")
        model = learn_offline(
            DataFrame(samples_train),
            learner,
            inputs=inputs,
            outputs=outputs,
        )
        models.append(model)

    metrics = [
        Accuracy(),
        ClassificationReport(),
        FBetaScore(beta=1.0),
        PrecisionScore(),
        Recall(),
    ]
    report = evaluate_offline(
        models,
        DataFrame(samples_eval),
        inputs=inputs,
        outputs=outputs,
        metrics=metrics,
    )
    print(report.pretty_print())
    # Save the model
    best_model = models[
        0
    ]  # replace with select_best_model() once it is available
    model_path = Path(config.learning.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model to %s", model_path)
    best_model.save(model_path)


if __name__ == "__main__":
    main()
