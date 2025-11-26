import polars as pl
from ml_pipeline.dataset.bag_processor import process_single_bag
from ml_pipeline.dataset.helpers import get_topics
from ml_pipeline.utils.paths import DATASETS

import flowcean.cli


def safe_iter(x):
    """Convert None → empty list.
    Convert single string → [string].
    Return list unchanged.
    """
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def main():
    # Load config
    config = flowcean.cli.initialize()
    topics = get_topics()

    pos_th = float(config.localization.position_threshold)
    head_th = float(config.localization.heading_threshold)

    # Convert configs safely
    training_paths = safe_iter(config.rosbag.training_paths)
    eval_paths = safe_iter(config.rosbag.evaluation_paths)

    # ============================================================
    # TRAINING SET
    # ============================================================
    if len(training_paths) > 0:
        print(f"📦 Processing {len(training_paths)} training bag(s)...")

        train_tables = []
        for bag in training_paths:
            print(f" → Training bag: {bag}")
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
    else:
        print("⚠️ No training bags found — skipping training dataset creation.")

    # ============================================================
    # EVALUATION SET
    # ============================================================
    if len(eval_paths) > 0:
        print(f"📦 Processing {len(eval_paths)} evaluation bag(s)...")

        eval_tables = []
        for bag in eval_paths:
            print(f" → Evaluation bag: {bag}")
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
    else:
        print(
            "⚠️ No evaluation bags found — skipping evaluation dataset creation.",
        )

    # ============================================================
    # FINAL SANITY CHECK
    # ============================================================
    if len(training_paths) == 0 and len(eval_paths) == 0:
        print(
            "\n❌ ERROR: Both training_paths and evaluation_paths are empty.",
        )
        print("Nothing to process. Check your config.yaml.")
        return


if __name__ == "__main__":
    main()
