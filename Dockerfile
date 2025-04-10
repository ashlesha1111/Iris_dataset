FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY scripts/ scripts/
COPY data/ data/
CMD ["python", "scripts/train.py"]
RUN touch scripts/__init__.py
