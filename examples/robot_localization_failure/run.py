#!/usr/bin/env python

import logging
from pathlib import Path

import polars as pl
from training import collect_data, evaluate, train

import flowcean.cli

logger = logging.getLogger(__name__)


def main() -> None:
    config = flowcean.cli.initialize()
    print("Configuration:", config)
    samples_train, samples_eval = collect_data(config)

    ##### Make the dataset balanced by throwing out the rows ########
    # Count how many rows per class
    counts = samples_train.group_by("is_delocalized").count()

    # Find the smallest class size
    min_count = counts["count"].min()

    # Take `min_count` rows from each class and overwrite samples_train
    samples_train = pl.concat(
        [
            samples_train.filter(pl.col("is_delocalized") == val).head(
                min_count,
            )
            for val in [True, False]
        ],
    )

    ##################################################################

    model = train(
        train_data=samples_train,
        config=config,
    )
    report = evaluate(model=model, test_data=samples_eval, config=config)
    print(report)
    # Save the model
    model_path = Path(config.learning.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model to %s", model_path)
    model.save(model_path)


if __name__ == "__main__":
    main()
