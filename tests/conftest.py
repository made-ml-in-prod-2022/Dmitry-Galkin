import os
from typing import List

import pytest


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "train_data_sample.csv")


@pytest.fixture()
def target_col():
    return "condition"


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "cp",
        "restecg",
        "slope",
        "ca",
        "thal",
    ]


@pytest.fixture()
def simple_categorical_features() -> List[str]:
    return [
        "sex",
        "fbs",
        "exang",
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    ]
