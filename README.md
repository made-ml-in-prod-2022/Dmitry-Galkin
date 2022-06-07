# ml_project

Доступно 2 модели классификации: 
* логистическая регрессия
* случайный лес

Данные автоматически выкачиваются с гугл диска.

**Установка (Windows):**

> python -m venv .venv
> 
> source .venv/Scripts/activate
> 
> pip install -r requirements.txt

**Использование:**

*Тренировка модели:* 
> * python ml_project/train_pipeline configs/train_config_logistic_regression.yaml
> 
> или
> 
> * python ml_project/train_pipeline configs/train_config_random_forest.yaml

*Предсказание:*
> * python ml_project/predict_pipeline configs/train_config_logistic_regression.yaml
> 
> или
> 
> * python ml_project/predict_pipeline configs/train_config_random_forest.yaml



**MLflow:**

> В командной строке необходимо вбить **mlflow ui**, появится **url**, который необходимо вбить в поиской строке браузера и откроется окно с экспериментами и залоггированными метриками
 
 
 **Запуск тестов:**
 
 > * всех сразу: pytest tests/
 > * или надо выбрать конкретный: pytest tests/test_make_dataset.py
