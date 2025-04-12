# Iris_dataset
#  Iris Flower Classification - Deep Learning with Docker

This project demonstrates the process of building, training, and deploying a deep learning model for Iris flower classification using PyTorch inside Docker containers. The goal is to showcase the end-to-end pipeline from data preprocessing to model inference, all encapsulated within Docker for reproducibility and modularity.

Iris Flower Classification – Deep Learning with Docker is an end-to-end machine learning pipeline that:

Downloads and preprocesses the classic Iris dataset.

Trains a deep learning model using PyTorch to classify flower species based on petal and sepal measurements.

Saves the trained model (model.pth) for reuse.

Performs batch inference on unseen data using the saved model.

Runs the entire workflow in Docker containers, ensuring full environment isolation and reproducibility.

Includes unit tests for both training and inference to ensure correctness and robustness.

This project is ideal for demonstrating how to build, containerize, and deploy a machine learning solution with PyTorch + Docker — from data to predictions.

---

Iris_dataset/
│
├── data/                            # Preprocessed dataset files
│   ├── iris_train.csv              # Preprocessed training data
│   └── iris_inference.csv          # Preprocessed inference data
│
├── training/                        # Training pipeline
│   ├── Dockerfile                  # Dockerfile for training
│   ├── train.py                    # PyTorch training script
│   ├── model.pth                   # Saved trained model
│   └── requirements.txt           # Python dependencies for training
│
├── inference/                       # Inference pipeline
│   ├── Dockerfile                  # Dockerfile for inference
│   ├── inference.py                # Batch inference script using the trained model
│   └── requirements.txt           # Python dependencies for inference
│
├── scripts/                         # Utility scripts
│   └── preprocess.py              # Script for downloading and preprocessing the dataset
│
├── tests/                           # Unit tests for training and inference
│   ├── test_training.py           # Unit tests for training pipeline
│   └── test_inference.py          # Unit tests for inference pipeline
│
├── .gitignore                       # Git ignore file (e.g., ignores models, logs)
└── README.md                        # Project documentation

# Step 1: Clone the Repository
git clone https://github.com/yourusername/Iris_dataset.git
cd Iris_dataset

# Step 2: Preprocess the Dataset
python scripts/preprocess.py
Download the Dataset

Loads the Iris dataset using scikit-learn's built-in loader.

Convert to DataFrame

Transforms the dataset into a structured pandas DataFrame for easier manipulation.

Clean and Format

Assigns appropriate column names: sepal length, sepal width, petal length, petal width, and target (species).

Maps target integers to species names (setosa, versicolor, virginica).

Shuffle the Data

Randomly shuffles the dataset to ensure a good mix during training and inference.

Split into Two Sets

Training Set: 80% of the data saved as iris_train.csv.

# Step 3: Build Docker Image for Training
docker build -f training/Dockerfile -t iris-train .
The purpose of this step is to create a portable, isolated environment that can run the training script (train.py) with all necessary dependencies using Docker.
Environment Reproducibility
Allows the model training process to be easily repeated, shared, and deployed — all through a simple Docker command.
Part of a Modular ML Pipeline
Keeps training logic separated from data preprocessing and inference, promoting a clean, maintainable pipeline.

# Step 4: Run Docker Container for Training
docker run --rm `
  -v "C:\Users\ashlesha_saxena\Iris_dataset\data:/app/data" `
  -v "C:\Users\ashlesha_saxena\Iris_dataset\training:/app/training" `
  iris-train

# Step 5: Build Docker Image for Inference
docker build -f inference/Dockerfile -t iris-infer .
creates an isolated image that includes all required packages for running inference (PyTorch, pandas, etc.).
Packages  inference.py script into a self-contained Docker image named iris-infer.


# Step 6: Run Docker Container for Inference
docker run --rm `
  -v "C:\Users\ashlesha_saxena\Iris_dataset\data:/app/data" `
  -v "C:\Users\ashlesha_saxena\Iris_dataset\training:/app/training" `
  iris-infer

# Step 7 (Optional): Run Unit Tests
pytest tests/test_training.py
pytest tests/test_inference.py
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(true_labels, predicted_labels)
ConfusionMatrixDisplay(cm, display_labels=['Setosa', 'Versicolor', 'Virginica']).plot()
# Shared across training/requirements.txt and inference/requirements.txt:
torch
pandas
scikit-learn
numpy
pytest   # Optional, for unit testing
# Features Summary
 Fully Dockerized Training & Inference Pipelines

 Clean Data Preprocessing

 Simple and Effective Neural Network
 Batch Inference with Saved Model
 Unit Testing for Robustness
 Reproducible and Portable Setup


