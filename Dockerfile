FROM python:3.8-slim-buster
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# COPY online_inference /online_inference && tests /tests && models /models && data /data
COPY . .

ENV PATH_TO_MODEL="https://drive.google.com/uc?id=1MCtuXgweZ0pTuQs-7oTfnCzU1MmJcAc_"
ENV PATH_TO_DATA="https://drive.google.com/uc?id=1VMzWpU-LgO_2Q9C-hcbQpN2tJdmXi2pz"

WORKDIR .

CMD ["uvicorn", "online_inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
