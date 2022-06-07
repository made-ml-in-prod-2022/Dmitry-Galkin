# -*- coding: utf-8 -*-
from typing import NoReturn, Tuple

import gdown
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.enities.split_params import SplittingParams


def download_data_from_google_drive(url: str, output: str) -> NoReturn:
    gdown.download(url=url, output=output, quiet=False)


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data
