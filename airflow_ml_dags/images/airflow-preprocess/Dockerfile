FROM python:3.6-slim-stretch

COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache -r requirements.txt

COPY preprocess.py /preprocess.py

ENTRYPOINT ["python", "preprocess.py"]
