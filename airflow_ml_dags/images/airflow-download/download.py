import os
from typing import Tuple

import click
import gdown
import numpy as np
import pandas as pd

TARGET_NAME = "condition"


def download_data_from_google_drive(url: str, output: str):
    """Выкачивание данных из Гугла."""
    gdown.download(url=url, output=output, quiet=False)


def read_data(path: str) -> pd.DataFrame:
    """Чтение данных."""
    data = pd.read_csv(path)
    return data


def transform_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Трансформирование, чтобы всегда были новые."""

    target = data[TARGET_NAME]
    data_transformed = data.drop(columns=TARGET_NAME)

    for col in data_transformed.columns:
        data_transformed[col] = np.random.choice(
            data[col], data_transformed.shape[0], replace=True
        )

    return data_transformed, target


@click.command(name="download")
@click.option("--output-dir")
@click.option(
    "--data-url",
    default="https://drive.google.com/uc?id=1H0IDV40zbiJlBh5O27yfTc3wZ-23eK3Z"
)
def download(output_dir: str, data_url: str):
    """Загрузка данных."""
    os.makedirs(output_dir, exist_ok=True)
    output_raw = os.path.join(output_dir, "data.csv")
    download_data_from_google_drive(url=data_url, output=output_raw)
    data = read_data(path=output_raw)
    data, target = transform_data(data)
    data.to_csv(os.path.join(output_dir, "data.csv"), index=False)  # перезаписываем
    target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == '__main__':
    download()
