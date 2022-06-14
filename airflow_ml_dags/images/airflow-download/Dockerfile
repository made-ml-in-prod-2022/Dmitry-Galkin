FROM python:3.6-slim-stretch

COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache -r requirements.txt

COPY download.py /download.py

ENTRYPOINT ["python", "download.py"]
