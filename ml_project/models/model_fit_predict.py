import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from ml_project.enities.train_params import TrainingParams

SklearnClassificationModel = Union[RandomForestClassifier, LogisticRegression]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnClassificationModel:
    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            n_estimators=train_params.n_estimators, random_state=train_params.random_state
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
    model: Pipeline, features: pd.DataFrame
) -> np.ndarray:
    predicts_proba = model.predict_proba(features)[:, 1]
    predicts = model.predict(features)
    return predicts_proba, predicts


def evaluate_model(
    predicts_proba: np.ndarray, predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    return {
        "roc_auc_score": roc_auc_score(target, predicts_proba),
        "accuracy_score": accuracy_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }


def create_inference_pipeline(
    model: SklearnClassificationModel, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def load_model(path_to_model: str) -> SklearnClassificationModel:
    with open(path_to_model, "rb") as f:
        model = pickle.load(f)
    return model


def save_predictions(data: np.ndarray, output: str):
    with open(output, "w") as f:
        f.write(("\n".join(str(x) for x in data)))
