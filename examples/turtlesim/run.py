import logging
from os import PathLike

import polars as pl
from custom_transforms.zero_order_hold_matching import ZeroOrderHold
from matplotlib import pyplot as plt
from omegaconf import DictConfig, ListConfig

import flowcean
import flowcean.cli
from examples.turtlesim.custom_metrics.euclidean_distance import (
    EuclideanDistance,
)
from flowcean.core.model import Model
from flowcean.core.strategies import evaluate_offline, learn_offline
from flowcean.polars.environments.dataframe import DataFrame
from flowcean.ros.rosbag import load_rosbag
from flowcean.sklearn.metrics.regression import (
    MaxError,
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)
from flowcean.sklearn.random_forest import RandomForestRegressorLearner
from flowcean.sklearn.regression_tree import RegressionTree
from flowcean.torch.lightning_learner import (
    LightningLearner,
    MultilayerPerceptron,
)

logger = logging.getLogger(__name__)


def explode_and_unnest(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.explode("measurements")
        .unnest("measurements")
        .unnest("value")
        .drop("/turtle1/cmd_vel", "/turtle1/pose", "time")
    )


def shift_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col("/turtle1/pose/x").shift(-1).alias("/turtle1/pose/x_next"),
            pl.col("/turtle1/pose/y").shift(-1).alias("/turtle1/pose/y_next"),
            pl.col("/turtle1/pose/theta")
            .shift(-1)
            .alias("/turtle1/pose/theta_next"),
        ],
    ).filter(pl.col("/turtle1/pose/x_next").is_not_null())


def load_and_process_rosbag(
    path: str | PathLike,
    config: DictConfig | ListConfig,
) -> pl.DataFrame:
    logger.info("Processing rosbag: %s", path)
    data = load_rosbag(
        path=path,
        topics={
            "/turtle1/cmd_vel": [
                "linear.x",
                "angular.z",
            ],
            "/turtle1/pose": [
                "x",
                "y",
                "theta",
            ],
        },
        message_paths=config.rosbag.message_paths,
    )
    transform = ZeroOrderHold(
        features=[
            "/turtle1/cmd_vel",
            "/turtle1/pose",
        ],
        name="measurements",
    )
    transformed_data: pl.DataFrame = transform.apply(data).collect(
        engine="streaming",
    )
    transformed_data = explode_and_unnest(transformed_data)
    logger.info("After exploding and unnesting: %s", transformed_data)
    transformed_data = shift_columns(transformed_data)
    logger.info("After shifting columns: %s", transformed_data)
    return transformed_data


def plot_predictions_vs_ground_truth(
    samples_eval: pl.DataFrame,
    models: dict[str, Model],
    input_names: list[str],
    output_names: list[str],
) -> None:
    for model_name, model in models.items():
        predictions = model.predict(
            samples_eval.select(input_names).lazy(),
        ).collect()
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
                f"plots/{model_name}_{output_name.replace('/', '_')}_plot.png",
            )
            plt.close()


def main() -> None:
    config = flowcean.cli.initialize()
    train_path = config.rosbag.training_paths[0]
    logger.info("Loading training rosbag from: %s", train_path)
    samples_train = load_and_process_rosbag(train_path, config)
    eval_path = config.rosbag.evaluation_paths[0]
    logger.info("Loading evaluation rosbag from: %s", eval_path)
    samples_eval = load_and_process_rosbag(eval_path, config)
    logger.info("Training samples: %s", samples_train)
    logger.info("Evaluation samples: %s", samples_eval)
    input_names = samples_train.drop(
        "/turtle1/pose/x_next",
        "/turtle1/pose/y_next",
        "/turtle1/pose/theta_next",
    ).columns
    output_names = samples_train.select(
        "/turtle1/pose/x_next",
        "/turtle1/pose/y_next",
        "/turtle1/pose/theta_next",
    ).columns
    tree_params = {
        "max_leaf_nodes": 1000,
    }
    regression_tree_learner = RegressionTree(**tree_params)
    regression_tree_model = learn_offline(
        DataFrame(samples_train),
        regression_tree_learner,
        inputs=input_names,
        outputs=output_names,
    )
    forest_params = {
        "n_estimators": 100,
        "max_depth": 10,
    }
    random_forest_learner = RandomForestRegressorLearner(**forest_params)
    random_forest_model = learn_offline(
        DataFrame(samples_train),
        random_forest_learner,
        inputs=input_names,
        outputs=output_names,
    )
    mlp_learner = LightningLearner(
        module=MultilayerPerceptron(
            learning_rate=1e-3,
            input_size=len(input_names),
            output_size=len(output_names),
        ),
        batch_size=64,
        max_epochs=5,
    )
    mlp_model = learn_offline(
        DataFrame(samples_train),
        mlp_learner,
        inputs=input_names,
        outputs=output_names,
    )
    models = {
        "regression_tree": regression_tree_model,
        "random_forest": random_forest_model,
        "multilayer_perceptron": mlp_model,
    }
    metrics = [
        MaxError(),
        MeanAbsoluteError(),
        MeanSquaredError(),
        R2Score(),
    ]
    example_rt_output = regression_tree_model.predict(
        samples_eval.select(input_names).limit(10).lazy(),
    )
    logger.info("Example output: %s", example_rt_output.collect())
    example_rf_output = random_forest_model.predict(
        samples_eval.select(input_names).limit(10).lazy(),
    )
    logger.info("Example output: %s", example_rf_output.collect())
    example_mlp_output = mlp_model.predict(
        samples_eval.select(input_names).limit(10).lazy(),
    )
    logger.info("Example output: %s", example_mlp_output.collect())

    reports = {}
    for model_name, model in models.items():
        for output_name in output_names:
            logger.info("\nEvaluating %s with %s:", output_name, model_name)
            report = evaluate_offline(
                model=model,
                environment=DataFrame(samples_eval),
                metrics=metrics,
                inputs=input_names,
                outputs=[output_name],
            )
            reports[(model_name, output_name)] = report
            formatted_report = "\n".join(
                f"  {metric_name}: {value:.2e}"
                for metric_name, value in report[output_name].items()
            )
            logger.info(
                "Evaluation report for %s:\n%s",
                output_name,
                formatted_report,
            )

    # Evaluate EuclideanDistance metric
    euclidean_distances = {}
    for model_name, model in models.items():
        columns = ["/turtle1/pose/x_next", "/turtle1/pose/y_next"]
        print(f"\nEvaluating Euclidean Distance with {model_name}:")
        report = evaluate_offline(
            model=model,
            environment=DataFrame(samples_eval),
            metrics=[EuclideanDistance(columns=columns)],
            inputs=input_names,
            outputs=columns,
        )
        print(report)
        value = report["multi_output"]["EuclideanDistance"]
        formatted_report = f"  EuclideanDistance: {value:.4f}"
        print(formatted_report)
        logger.info(
            "Evaluation report for %s with %s:\n%s",
            "EuclideanDistance",
            model_name,
            formatted_report,
        )
        euclidean_distances[model_name] = value

    print("euclidean_distances:", euclidean_distances)
    best_model = min(
        euclidean_distances,
        key=euclidean_distances.get,
    )
    logger.info("Best model: %s", best_model)
    # save the best model
    models[best_model].save("best_model.fml")


if __name__ == "__main__":
    main()
