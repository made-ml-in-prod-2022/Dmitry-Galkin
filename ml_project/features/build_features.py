import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_project.enities.feature_params import FeatureParams
from ml_project.features.custom_transformer import FeatureTransformer


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("scaler", StandardScaler()), ]
    )
    return num_pipeline


def build_transformer_log_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("log", FeatureTransformer("log"))]
    )
    return num_pipeline


def build_transformer_square_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [("square", FeatureTransformer("square"))]
    )
    return num_pipeline


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return transformer.transform(df)


def build_transformer(params: FeatureParams) -> ColumnTransformer:

    column_transformer_input = []
    if params.categorical_features:
        column_transformer_input.append(
            ("categorical_pipeline", build_categorical_pipeline(), params.categorical_features)
        )
    if params.numerical_features:
        column_transformer_input.append(
            ("numerical_pipeline", build_numerical_pipeline(), params.numerical_features)
        )
    if params.simple_categorical_features:
        column_transformer_input.append(
            ("simple_pipeline", "passthrough", params.simple_categorical_features)
        )
    if params.transformer_features_log:
        column_transformer_input.append(
            (
                "transformer_log_pipeline",
                build_transformer_log_pipeline(),
                params.transformer_features_log
            )
        )
    if params.transformer_features_square:
        column_transformer_input.append(
            (
                "transformer_square_pipeline",
                build_transformer_square_pipeline(),
                params.transformer_features_square,
            )
        )

    transformer = ColumnTransformer(column_transformer_input)

    return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target
