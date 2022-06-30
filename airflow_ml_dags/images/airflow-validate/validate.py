import json
import os
import pickle
from typing import Dict

import click
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def load_model(path: str):
    """Загрузка модели."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def evaluate(
        predicts: np.ndarray, predicts_proba: np.ndarray, target: pd.Series
) -> Dict[str, int]:
    """Подсчет метрик."""

    metrics_dict = {}

    metrics_dict.update({"accuracy": accuracy_score(target, predicts)})
    metrics_dict.update({"roc_auc": roc_auc_score(target, predicts_proba)})
    metrics_dict.update({"f1": f1_score(target, predicts)})
    return metrics_dict


@click.command(name="validate")
@click.option("--input-dir")
@click.option("--model-dir")
def validate(input_dir: str, model_dir: str):
    """Валидация модели."""

    val_data = pd.read_csv(os.path.join(input_dir, "val_data.csv"))
    val_target = pd.read_csv(os.path.join(input_dir, "val_target.csv"))

    model = load_model(os.path.join(model_dir, "model.pkl"))
    pred = model.predict(val_data)
    pred_prob = model.predict_proba(val_data)[:, 1]
    val_real = val_target["condition"].tolist()
    metrics = evaluate(pred, pred_prob, val_real)

    with open(os.path.join(model_dir, "metrics.json"), "w") as metric_file:
        json.dump(metrics, metric_file)


if __name__ == '__main__':
    validate()
