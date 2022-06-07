import os
import sys
from os.path import exists

import pandas as pd
from py._path.local import LocalPath

from ml_project.data.make_dataset import (download_data_from_google_drive,
                                          read_data, split_train_val_data)
from ml_project.enities.split_params import SplittingParams

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))

DATA_URL = "https://drive.google.com/uc?id=156Wh9zJx-3srAhqB_BbBHCMspfrcxMHG"
VAL_SIZE = 0.2


def test_download_data_from_google(tmpdir: LocalPath):

    filename = tmpdir.join("tmpfile.csv")
    download_data_from_google_drive(
        url=DATA_URL,
        output=filename
    )

    assert exists(filename)


def test_read_data(dataset_path: str):
    data = read_data(dataset_path)
    assert isinstance(data, pd.DataFrame)


def test_split_train_val_data(dataset_path: str):

    data = read_data(dataset_path)
    params = SplittingParams(val_size=VAL_SIZE)
    train, test = split_train_val_data(data=data, params=params)

    assert abs(data.shape[0] * VAL_SIZE - test.shape[0]) <= 1
    assert abs(data.shape[0] * (1 - VAL_SIZE) - train.shape[0]) <= 1
