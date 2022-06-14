import os
from typing import Tuple

import click
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.8


def split_train_val_data(
        data: pd.DataFrame, target: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разбиение на трейн и тест."""

    train_data, val_data, y_train, y_test = train_test_split(
        data,
        target,
        train_size=TRAIN_SIZE,
        shuffle=True,
        random_state=42
    )
    return train_data, val_data, y_train, y_test


@click.command(name="split")
@click.option("--input-dir")
@click.option("--output-dir")
def split(input_dir: str, output_dir: str):

    data = pd.read_csv(os.path.join(input_dir, "data_processed.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    train_data, val_data, y_train, y_test = split_train_val_data(data, target)

    os.makedirs(output_dir, exist_ok=True)

    train_data.to_csv(os.path.join(output_dir, "train_data.csv"), index=None)
    val_data.to_csv(os.path.join(output_dir, "val_data.csv"), index=None)
    y_train.to_csv(os.path.join(output_dir, "train_target.csv"), index=None)
    y_test.to_csv(os.path.join(output_dir, "val_target.csv"), index=None)


if __name__ == '__main__':
    split()
