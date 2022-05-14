import os
import sys
from typing import List

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_project.data.make_dataset import read_data
from ml_project.enities.feature_params import FeatureParams
from ml_project.features.build_features import (
    build_categorical_pipeline, build_numerical_pipeline, build_transformer,
    build_transformer_log_pipeline, build_transformer_square_pipeline,
    extract_target, make_features)
from ml_project.features.custom_transformer import FeatureTransformer

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))

DATA = np.array([1, 2, 3])[:, None]


@pytest.fixture
def feature_params(
    categorical_features: List[str],
    numerical_features: List[str],
    simple_categorical_features: List[str],
    target_col: str,
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        simple_categorical_features=simple_categorical_features,
        target_col=target_col,
    )
    return params


def test_build_categorical_pipeline():

    pipe = build_categorical_pipeline()
    assert 1 == len(pipe.steps)
    assert isinstance(pipe.steps[0][1], OneHotEncoder)

    pipe.fit(DATA)
    data_transformed = pipe.transform(DATA)
    assert 3 == data_transformed.shape[1]
    assert 3 == data_transformed.sum()


def test_build_numerical_pipeline():

    pipe = build_numerical_pipeline()
    assert 1 == len(pipe.steps)
    assert isinstance(pipe.steps[0][1], StandardScaler)

    pipe.fit(DATA)
    data_transformed = pipe.transform(DATA)
    assert 0 == int(round(data_transformed.mean()))


def test_build_transformer_log_pipeline():

    pipe = build_transformer_log_pipeline()
    assert 1 == len(pipe.steps)
    assert isinstance(pipe.steps[0][1], FeatureTransformer)

    pipe.fit(DATA)
    data_transformed = pipe.transform(DATA)
    assert np.allclose(np.log(DATA), data_transformed, atol=1e-6)


def test_build_transformer_square_pipeline():

    pipe = build_transformer_square_pipeline()
    assert 1 == len(pipe.steps)
    assert isinstance(pipe.steps[0][1], FeatureTransformer)

    pipe.fit(DATA)
    data_transformed = pipe.transform(DATA)
    assert np.allclose(DATA**2, data_transformed)


def test_extract_target(feature_params: FeatureParams, dataset_path: str):
    data = read_data(dataset_path)
    target = extract_target(data, feature_params)

    assert np.allclose(
        data[feature_params.target_col].to_numpy(), target.to_numpy()
    )


def test_make_features(feature_params: FeatureParams, dataset_path: str):
    data = read_data(dataset_path)
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    features = make_features(transformer, data)
    assert not pd.isnull(features).any().any()
