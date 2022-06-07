import logging
import os
import sys
from pathlib import Path

import click

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))

from ml_project.data import read_data
from ml_project.data.make_dataset import download_data_from_google_drive
from ml_project.enities.train_pipeline_params import \
    read_training_pipeline_params
from ml_project.models.model_fit_predict import (load_model, predict_model,
                                                 save_predictions)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(config_path: str):
    training_pipeline_params = read_training_pipeline_params(config_path)
    return run_predict_pipeline(training_pipeline_params)


def run_predict_pipeline(training_pipeline_params):
    downloading_params = training_pipeline_params.downloading_params
    if downloading_params:
        os.makedirs(downloading_params.output_folder, exist_ok=True)
        download_data_from_google_drive(
            downloading_params.test_url,
            os.path.join(downloading_params.output_folder, Path(downloading_params.test_path).name),
        )

    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_test_path)
    logger.info(f"data.shape is {data.shape}")

    model = load_model(training_pipeline_params.output_model_path)
    logger.info(f"model {training_pipeline_params.output_model_path} was loaded")

    predicts_proba, predicts = predict_model(model, data)
    logger.info(f"predictions were made")

    save_predictions(predicts_proba, training_pipeline_params.output_prediction_proba_path)
    save_predictions(predicts, training_pipeline_params.output_prediction_path)
    logger.info(f"predictions were saved to "
                f"{training_pipeline_params.output_prediction_proba_path} "
                f"and {training_pipeline_params.output_prediction_path}")


@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict_pipeline_command()
