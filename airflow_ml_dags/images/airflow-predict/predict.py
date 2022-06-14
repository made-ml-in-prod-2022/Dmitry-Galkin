import os
import pickle

import click
import pandas as pd


def load_model(path: str):
    """Загрузка модели."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


@click.command(name="predict")
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    """Валидация модели."""

    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    model = load_model(os.path.join(model_dir, "model.pkl"))
    preds = model.predict_proba(data)[:, 1]
    os.makedirs(output_dir, exist_ok=True)
    preds_df = pd.DataFrame(preds)
    preds_df.to_csv(os.path.join(output_dir, "prediction.csv"), index=None)


if __name__ == '__main__':
    predict()
