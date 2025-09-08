import random
import sys
import time
from pathlib import Path

import polars as pl
from matplotlib import pyplot as plt

from flowcean.core.model import Model


def shift_in_time(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        pl.col("/turtle1/pose/x", "/turtle1/pose/y", "/turtle1/pose/theta")
        .shift(-1)
        .name.suffix("_next"),
    ).filter(pl.col("/turtle1/pose/x_next").is_not_null())


def plot_predictions_vs_ground_truth(
    samples_eval: pl.DataFrame,
    models: dict[str, Model],
    input_names: list[str],
    output_names: list[str],
) -> None:
    # check if plots directory exists
    Path("plots").mkdir(exist_ok=True)
    for model_name, model in models.items():
        predictions = model.predict(
            samples_eval.select(input_names).lazy(),
        ).collect()
        # create x-y plot
        plt.figure(figsize=(12, 12))
        plt.scatter(
            samples_eval.select(
                pl.col("/turtle1/pose/x_next"),
            ).to_series(),
            samples_eval.select(
                pl.col("/turtle1/pose/y_next"),
            ).to_series(),
            label="Ground Truth",
            color="red",
        )
        plt.scatter(
            model.predict(samples_eval.select(input_names).lazy())
            .collect()
            .select(pl.col("/turtle1/pose/x_next"))
            .to_series(),
            model.predict(samples_eval.select(input_names).lazy())
            .collect()
            .select(pl.col("/turtle1/pose/y_next"))
            .to_series(),
            label="Predictions",
            color="blue",
        )
        plt.title(f"2D Trajectory - {model_name}")
        plt.xlabel("x position")
        plt.ylabel("y position")
        plt.legend()
        plt.savefig(Path(f"plots/{model_name}_2d_trajectory.png"))
        plt.close()
        # create time series plots
        for output_name in output_names:
            plt.figure(figsize=(12, 6))
            plt.plot(
                samples_eval.select(pl.col(output_name)).to_series(),
                label="Ground Truth",
                color="blue",
            )
            plt.plot(
                predictions.select(pl.col(output_name)).to_series(),
                label="Predictions",
                color="red",
            )
            plt.title(
                f"Predictions vs Ground Truth - {model_name} - {output_name}",
            )
            plt.xlabel("Sample Index")
            plt.ylabel(output_name)
            plt.legend()
            plt.savefig(
                Path(
                    f"plots/{model_name}_{output_name.replace('/', '_')}.png",
                ),
            )
            plt.close()


def surprise() -> None:
    confetti = ["✨", "🎉", "🎊", "🌟", "💫"]
    message = "🎉 Congratulations! You finished the tutorial! 🎉"

    # Print message with typing effect
    for char in message:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.05)
    print("\n")

    # Animation: falling confetti
    for _ in range(15):
        line = "".join(random.choice(confetti) for _ in range(50))
        print(line)
        time.sleep(0.1)

    print("\n🎯 Great job! Keep learning & experimenting! 🚀")
