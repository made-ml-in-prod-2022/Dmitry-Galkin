# online_inference
---
## Запуск Docker (*Windows*)

**Через dockerhub**

> docker pull dgalkin/online_inference:v1
> 
> docker run -p 8000:8000 dgalkin/online_inference:v1

**Локально**

> docker build -t dgalkin/online_inference:v1 .
> 
> docker run -p 8000:8000 dgalkin/online_inference:v1

**Если без Docker**

> uvicorn online_inference.app:app
> 
> или
> 
> PATH_TO_MODEL="https://drive.google.com/uc?id=1MCtuXgweZ0pTuQs-7oTfnCzU1MmJcAc_" uvicorn online_inference.app:app

*Если не указать путь до модели, то автоматически проверится ее наличие в папке models. Если модели не будет, то вывалится ошибка и вернется код 404.*

**Какие есть опции:**

* http://0.0.0.0:8000/docs - документация
* http://0.0.0.0:8000/health - определяет готовность модели (возвращает код 200 при готовности, иначе 404)
* http://0.0.0.0:8000/predict 0 онлайн-инференс

**Скрипт для запросов к серверу:**

> python online_inference/make_request.py
> 
> или
> 
> PATH_TO_DATA="https://drive.google.com/uc?id=1VMzWpU-LgO_2Q9C-hcbQpN2tJdmXi2pz" python online_inference/make_request.py

*Если не указать путь до данных, то автоматически проверится их наличие в папке data. Если данных не будет, то вывалится ошибка.*

**Тесты**

> pytest tests/
