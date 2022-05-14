# ml_project




**Installation (Windows):**

> python -m venv .venv
> source .venv/Scripts/activate
> pip install -r requirements.txt

**Usage:**
train: 
python ml_project/train_pipeline configs/train_config_logistic_regression.yaml
predict: 
python ml_project/predict_pipeline configs/train_config_logistic_regression.yaml

**MLflow:**
> mlflow ui
 
 
 **Tests:**
 > pytest tests/