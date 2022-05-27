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
* http://0.0.0.0:8000/predict - онлайн-инференс

**Скрипт для запросов к серверу:**

> python online_inference/make_request.py
> 
> или
> 
> PATH_TO_DATA="https://drive.google.com/uc?id=1VMzWpU-LgO_2Q9C-hcbQpN2tJdmXi2pz" python online_inference/make_request.py

*Вторая инструкция только для локального запуска, без докера.*

*Если не указать путь до данных, то автоматически проверится их наличие в папке data. Если данных не будет, то вывалится ошибка.*

**Тесты**

> pytest tests/

**Валидация входных данных**
* Проверяется порядок столбцов
* Количество столбцов
* Диапазоны значений для категориальных и числовых признаков (для последних с запасов 10%)
* Типы переменных

*Если валидация не пройдена, посылается код 400 с описанием соответствующей ошибки.*

**Оптимизация размера docker image**

Пользовался этой статьей: https://habr.com/ru/company/ruvds/blog/440658/

* Использование базового образа python:3.8-slim-buster вместо python:3.8
* Использование .dockerignore, что позволило сделать всего одну строчку с командой COPY (для файлов)
* Использование только релевантных библиотек
* Сначала установка библиотек из requirements, затем копирование файлов
