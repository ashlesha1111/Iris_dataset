# inference/Dockerfile
FROM python:3.10.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/inference.py .
COPY training/train.py .
COPY data /app/data
COPY models /app/models

CMD ["python", "inference.py"]
