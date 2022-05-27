import logging
import os
import sys

import gdown
import pandas as pd
import requests

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

DATA_DEFAULT_PATH = "data/data.csv"


if __name__ == "__main__":

    data_path = os.getenv("PATH_TO_DATA")
    if data_path is not None:
        gdown.download(url=data_path, output=DATA_DEFAULT_PATH, quiet=False)

    if os.path.isfile(DATA_DEFAULT_PATH):
        data = pd.read_csv(DATA_DEFAULT_PATH)
    else:
        err = f"PATH_TO_MODEL {data_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    request_features = list(data.columns)
    for i in data.index:
        request_data = data.iloc[i].tolist()
        logger.info(f"\nfeatures: {request_data}")
        response = requests.get(
            "http://127.0.0.1:8000/predict/",
            json={"data": [request_data], "features": request_features},
        )
        logger.info(f"status_code: {response.status_code}")
        logger.info(f"result: {response.json()}")
