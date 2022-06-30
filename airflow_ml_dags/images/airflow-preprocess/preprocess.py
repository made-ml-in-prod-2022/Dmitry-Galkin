import os

import click
import pandas as pd


@click.command(name="preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):
    """Препроцессинг данных."""

    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    # просто перенесем данные
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data_processed.csv"), index=None)
    target.to_csv(os.path.join(output_dir, "target.csv"), index=None)


if __name__ == '__main__':
    preprocess()
