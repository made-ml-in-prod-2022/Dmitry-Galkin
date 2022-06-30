# homework3 - airflow


**Запуск** *(поднятие локально Airflow через docker-compose):*

> export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
> 
> docker compose up --build


За основу взята задача:  https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

Реализовано 3 дага:
* Ежедневное скачивание данных
* Еженедельное обучение модели и запись метрик
* Ежедневное предсказание результатов на основе модели, которая задана в airflow variables
