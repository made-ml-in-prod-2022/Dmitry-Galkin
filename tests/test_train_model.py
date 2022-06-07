import os
import pickle
import sys
from typing import List, Tuple

import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from ml_project.data.make_dataset import read_data
from ml_project.enities import TrainingParams
from ml_project.enities.feature_params import FeatureParams
from ml_project.features.build_features import (build_transformer,
                                                extract_target, make_features)
from ml_project.models.model_fit_predict import serialize_model, train_model

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))


@pytest.fixture
def features_and_target(
        dataset_path: str,
        categorical_features: List[str],
        numerical_features: List[str],
        simple_categorical_features: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        simple_categorical_features=simple_categorical_features,
        target_col="condition",
    )
    data = read_data(dataset_path)
    transformer = build_transformer(params)
    transformer.fit(data)
    features = make_features(transformer, data)
    target = extract_target(data, params)
    return features, target


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    assert isinstance(model, LogisticRegression)
    check_is_fitted(model)


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    n_estimators = 10
    model = LogisticRegression()
    real_output = serialize_model(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, LogisticRegression)

