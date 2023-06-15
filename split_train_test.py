#!/usr/bin/env python

import argparse
import os
from pathlib import Path
from typing import Optional

import polars as pl
from sklearn.model_selection import train_test_split


def split(
    csv_file: Path,
    train_filename,
    test_filename,
    test_size,
    output_dir: Optional[Path] = None,
    random_state: int = 42,
):
    data = pl.read_csv(csv_file)

    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )

    if output_dir is None:
        output_dir = Path(os.path.dirname(csv_file))
    train.write_csv(output_dir / train_filename)
    test.write_csv(output_dir / test_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_file",
        type=Path,
        help="csv file to split",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="output directory",
        default=None,
    )
    parser.add_argument(
        "--train_filename",
        type=str,
        help="name of the test dataset file",
        default="train.csv",
    )
    parser.add_argument(
        "--test_filename",
        type=str,
        help="name of the training dataset file",
        default="test.csv",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        help="relative size of the test dataset",
        default=0.2,
    )
    arguments = parser.parse_args()
    split(
        arguments.csv_file,
        train_filename=arguments.train_filename,
        test_filename=arguments.test_filename,
        test_size=arguments.test_size,
        output_dir=arguments.output_dir,
    )
