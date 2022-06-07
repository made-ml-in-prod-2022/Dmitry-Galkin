from dataclasses import dataclass
from typing import Optional

import yaml
from marshmallow_dataclass import class_schema

from .download_params import DownloadParams
from .feature_params import FeatureParams
from .split_params import SplittingParams
from .train_params import TrainingParams


@dataclass()
class TrainingPipelineParams:
    input_data_train_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    input_data_test_path: str = None
    output_prediction_path: str = None
    output_prediction_proba_path: str = None
    downloading_params: Optional[DownloadParams] = None
    use_mlflow: bool = True
    mlflow_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment: str = "made"


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
