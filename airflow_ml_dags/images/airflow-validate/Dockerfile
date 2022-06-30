FROM python:3.6-slim-stretch

COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache -r requirements.txt

COPY validate.py /validate.py

ENTRYPOINT ["python", "validate.py"]