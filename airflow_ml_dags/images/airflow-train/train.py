import os
import pickle

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

N_ESTIMATORS = 50


@click.command(name="train")
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir: str, output_dir):
    """Тренируем и сохраняем модель."""

    train_data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
    train_target = pd.read_csv(os.path.join(input_dir, "train_target.csv"))

    model = RandomForestClassifier(N_ESTIMATORS).fit(train_data, train_target["condition"])

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train()
