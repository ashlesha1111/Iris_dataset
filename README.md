# Iris_dataset
#  Iris Flower Classification - Deep Learning with Docker

This project demonstrates the process of building, training, and deploying a deep learning model for Iris flower classification using PyTorch inside Docker containers. The goal is to showcase the end-to-end pipeline from data preprocessing to model inference, all encapsulated within Docker for reproducibility and modularity.

---

## What Has Been Done

### 1.  **Data Downloading & Preprocessing**
- **Data Collection**: The Iris dataset is fetched from the [Iris flower data set - Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set) and stored in CSV format.
- **Preprocessing Steps**:
  - Data is cleaned and formatted into the required structure.
  - The dataset is split into two parts:
    - **Training Set** (`iris_train.csv`): Used to train the model.
    - **Inference Set** (`iris_inference.csv`): Used for making predictions after the model is trained.
- **Script**: `scripts/preprocess.py`
  - It handles:
    - Downloading the dataset.
    - Preprocessing (cleaning and splitting).
    - Saving preprocessed data for training and inference.

### 2. **Model Training with PyTorch (Dockerized)**
- **Training Script**: The model is implemented using PyTorch for classifying the Iris flowers.
- **Dockerized Training**:
  - A `Dockerfile` in the `training/` directory is used to create a Docker image that includes all the necessary dependencies for training the model.
  - The model is trained using the preprocessed training data (`iris_train.csv`).
  - The model is saved as a `.pth` file (`model.pth`) after training.
- **Training Flow**:
  - The model is trained on a simple feedforward neural network architecture.
  - The Docker container ensures the training is isolated and reproducible across different environments.

### 3. **Batch Inference (Dockerized)**
- **Inference Script**: After the model is trained, batch predictions are run using the saved model on the `iris_inference.csv`.
- **Dockerized Inference**:
  - A separate Docker container is used for running inference.
  - The trained model (`model.pth`) is loaded, and predictions are made on the inference data.
- **Output**: The predictions are displayed in the container or saved to a file.
- **Inference Flow**:
  - Docker ensures that inference is done in a clean, isolated environment without interfering with the training process.

### 4.  **Testing, Exception Handling & Comments**
- **Unit Tests**: To ensure correctness, unit tests are included for both the training and inference pipelines. Tests cover:
  - Correct loading and splitting of data.
  - Model training and accuracy.
  - Batch inference functionality.
- **Exception Handling**: Basic error handling is implemented to catch issues during data loading, model training, and inference.
- **Inline Comments**: Throughout the scripts, comments explain the functionality of each section of code to improve readability and maintainability.

---

## Folder Structure
Iris_dataset/ │ ├── data/ │ ├── iris_train.csv # Preprocessed training data │ └── iris_inference.csv # Preprocessed inference data │ ├── training/ │ ├── Dockerfile # Docker container for training │ ├── train.py # PyTorch model training script │ ├── model.pth # Saved model file │ └── requirements.txt # Training dependencies │ ├── inference/ │ ├── Dockerfile # Docker container for inference │ ├── inference.py # Inference script using saved model │ └── requirements.txt # Inference dependencies (optional) │ ├── scripts/ │ └── preprocess.py # Preprocessing and dataset splitting │ ├── tests/ │ ├── test_training.py # Tests for training pipeline │ └── test_inference.py # Tests for inference pipeline │ ├── .gitignore # Ignores model files, logs, caches, etc. └── README.md # Project overview and documentation
