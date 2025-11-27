#!/usr/bin/env python

import logging

from training import collect_data

import flowcean.cli
from flowcean.core.model import Model

logger = logging.getLogger(__name__)


def main() -> None:
    config = flowcean.cli.initialize()

    model = Model.load(
        "models/model_with_sigmoid_optimized_only_small_maps.fml",
    )
    print("Model type:", type(model))
    model.binary_classification_threshold = 0.3

    # evaluation
    return

    samples_train, samples_eval = collect_data(config)

    print("samples_train:", samples_train)
    print("samples_eval:", samples_eval)

    print("samples_train schema:", samples_train.schema)
    print("samples_eval schema:", samples_eval.schema)

    lazy_samples = samples_eval.lazy()

    print("lazy_samples:", lazy_samples)
    print("lazy_samples schema:", lazy_samples.schema)

    x = model.predict(lazy_samples).collect()

    print("Predictions:", x)

    counts = x.group_by("is_delocalized").len().sort("is_delocalized")

    print("Prediction counts:\n", counts)


if __name__ == "__main__":
    main()
