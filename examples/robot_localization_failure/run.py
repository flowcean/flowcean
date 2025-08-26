#!/usr/bin/env python

import logging
from pathlib import Path

from training import collect_data, evaluate, train

import flowcean.cli

logger = logging.getLogger(__name__)


def main() -> None:
    config = flowcean.cli.initialize()

    samples_train, samples_eval = collect_data(config)

    model = train(
        train_data=samples_train,
        config=config,
    )
    report = evaluate(model=model, test_data=samples_eval)
    print(report)
    # Save the model
    model_path = Path(config.learning.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model to %s", model_path)
    model.save(model_path)


if __name__ == "__main__":
    main()
