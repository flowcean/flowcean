#!/usr/bin/env python

import logging
from functools import partial
from pathlib import Path

import optuna
import polars as pl
from omegaconf import DictConfig, ListConfig
from optuna import Trial
from training import collect_data, evaluate, train

import flowcean.cli

logger = logging.getLogger(__name__)


def objective(
    trial: Trial,
    train_data: pl.DataFrame,
    eval_data: pl.DataFrame,
    config: DictConfig | ListConfig,
) -> float:
    config.learning.learning_rate = trial.suggest_float(
        "learning_rate",
        low=1e-5,
        high=1e-2,
        log=True,
    )
    config.learning.batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128, 256],
    )
    config.architecture.image_size = trial.suggest_categorical(
        "image_size",
        [64, 128, 150, 224],
    )
    config.architecture.width_meters = trial.suggest_float(
        "width_meters",
        5.0,
        50.0,
    )

    model = train(
        train_data=train_data,
        config=config,
    )
    report = evaluate(
        model=model,
        test_data=eval_data,
    )

    f1_score = report["FBetaScore"]
    logger.info("Trial finished with F1=%.4f", f1_score)

    return f1_score  # pyright: ignore[reportReturnType]


def main() -> None:
    config = flowcean.cli.initialize()

    samples_train, samples_eval = collect_data(config)

    logger.info("Starting Optuna optimization")
    study = optuna.create_study(
        direction="maximize",
        storage=config.optuna.storage,
        sampler=optuna.samplers.TPESampler(),
    )
    study.optimize(
        partial(
            objective,
            train_data=samples_train,
            eval_data=samples_eval,
            config=config,
        ),
        n_trials=2,
    )

    logger.info("Best trial: %s", study.best_trial.params)

    # --- Retrain final model with best params ---
    best_params = study.best_trial.params
    config.learning.learning_rate = best_params["learning_rate"]
    config.learning.batch_size = best_params["batch_size"]
    config.architecture.image_size = best_params["image_size"]
    config.architecture.width_meters = best_params["width_meters"]

    logger.info("Retraining final model with best parameters")

    final_model = train(
        train_data=samples_train,
        config=config,
    )

    # Save final model
    model_path = Path(config.learning.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving best model to %s", model_path)
    final_model.save(model_path)


if __name__ == "__main__":
    main()
