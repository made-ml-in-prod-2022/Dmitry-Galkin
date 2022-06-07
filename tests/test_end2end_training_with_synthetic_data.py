import os
import sys
from typing import List

import numpy as np
import pandas as pd
import pytest
from py._path.local import LocalPath

from ml_project.data.make_dataset import read_data
from ml_project.enities import (FeatureParams, SplittingParams, TrainingParams,
                                TrainingPipelineParams)
from ml_project.train_pipeline import run_train_pipeline

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))


@pytest.fixture()
def dataset_synthetic_path(dataset_synthetic_generation):
    curdir = os.path.dirname(__file__)
    path = os.path.join(curdir, "train_data_synthetic_sample.csv")
    dataset_synthetic = dataset_synthetic_generation
    dataset_synthetic.to_csv(path)
    return path


@pytest.fixture
def dataset_synthetic_generation(
        dataset_path: str,
        categorical_features: List[str],
        numerical_features: List[str],
        simple_categorical_features: List[str],
) -> pd.DataFrame:

    data = read_data(dataset_path)
    data_synthetic = data.copy()

    all_features = categorical_features + numerical_features + simple_categorical_features
    for col in all_features:
        data_synthetic[col] = np.random.choice(
            data[col], data_synthetic.shape[0], replace=True
        )

    return data_synthetic


def test_train_e2e(
    tmpdir: LocalPath,
    dataset_synthetic_path: str,
    categorical_features: List[str],
    numerical_features: List[str],
    simple_categorical_features: List[str],
    target_col: str,
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        input_data_train_path=dataset_synthetic_path,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=239),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            simple_categorical_features=simple_categorical_features,
            target_col=target_col,
        ),
        train_params=TrainingParams(model_type="LogisticRegression"),
    )
    real_model_path, metrics = run_train_pipeline(params)
    assert metrics["roc_auc_score"] > 0.3
    assert metrics["accuracy_score"] > 0.2
    assert metrics["f1_score"] > 0.2
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)
