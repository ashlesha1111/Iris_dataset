# Use an official Python image with PyTorch
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt ./
COPY training/train.py ./train.py
COPY data/ ./data/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Run training
CMD ["python", "train.py"]
