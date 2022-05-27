import logging
import os
import pickle
import sys
from typing import List, Optional

import gdown
import pandas as pd
import uvicorn
from fastapi import FastAPI, Response, status
from sklearn.pipeline import Pipeline

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))

from online_inference.heart_disease_model import (HeartDiseaseModel,
                                                  HeartDiseaseModelResponse)

logger = logging.getLogger(__name__)

app = FastAPI()
model: Optional[Pipeline] = None

MODEL_DEFAULT_PATH = "models/model_logistic_regression.pkl"


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_predict(
    data: List, features: List[str], model: Pipeline,
) -> List[HeartDiseaseModelResponse]:

    data = pd.DataFrame(data, columns=features)
    proba_predicts = model.predict_proba(data)[:, 1]
    predicts = model.predict(data)

    return [
        HeartDiseaseModelResponse(
            condition=int(condition),
            condition_proba=float(condition_proba)
        )
        for condition, condition_proba in zip(predicts, proba_predicts)
    ]


@app.get("/")
def main():
    return "heart-disease-cleveland-uci predictor"


@app.on_event("startup")
def load_model():
    global model

    model_path = os.getenv("PATH_TO_MODEL")

    if model_path is not None:
        gdown.download(url=model_path, output=MODEL_DEFAULT_PATH, quiet=False)

    if os.path.isfile(MODEL_DEFAULT_PATH):
        model = load_object(MODEL_DEFAULT_PATH)
    else:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)


@app.get("/predict/", response_model=List[HeartDiseaseModelResponse])
def predict(request: HeartDiseaseModel):
    return make_predict(request.data, request.features, model)


@app.get("/health")
def health(response: Response) -> bool:
    if model is None:
        response.status_code = status.HTTP_404_NOT_FOUND
    return not (model is None)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=os.getenv("PORT", 8000))
