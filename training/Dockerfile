# training/Dockerfile
FROM python:3.10.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY training/train.py .
COPY data /app/data

RUN mkdir -p /app/models

CMD ["python", "train.py"]
