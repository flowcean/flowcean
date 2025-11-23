import polars as pl
import flowcean.cli

from ml_pipeline.dataset.bag_processor import process_single_bag

from ml_pipeline.utils.paths import DATASETS
from ml_pipeline.dataset.helpers import get_topics


def main():
    config = flowcean.cli.initialize()
    topics = get_topics()

    pos_th = float(config.localization.position_threshold)
    head_th = float(config.localization.heading_threshold)

    # Training bags
    train_tables = []
    for bag in config.rosbag.training_paths:
        df = process_single_bag(
            bag_path=bag,
            topics=topics,
            message_paths=config.rosbag.message_paths,
            position_threshold=pos_th,
            heading_threshold=head_th,
        )
        train_tables.append(df)

    train_df = pl.concat(train_tables, how="vertical").sort("time")
    train_df.write_parquet(DATASETS / "train.parquet")
    train_df.write_csv(DATASETS / "train.csv")

    print("✔ Saved training dataset")

    # Evaluation bags
    eval_tables = []
    for bag in config.rosbag.evaluation_paths:
        df = process_single_bag(
            bag_path=bag,
            topics=topics,
            message_paths=config.rosbag.message_paths,
            position_threshold=pos_th,
            heading_threshold=head_th,
        )
        eval_tables.append(df)

    eval_df = pl.concat(eval_tables, how="vertical").sort("time")
    eval_df.write_parquet(DATASETS / "eval.parquet")
    eval_df.write_csv(DATASETS / "eval.csv")

    print("✔ Saved evaluation dataset")


if __name__ == "__main__":
    main()
