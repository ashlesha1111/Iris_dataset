# Iris_dataset
#  Iris Flower Classification - Deep Learning with Docker

This project demonstrates the process of building, training, and deploying a deep learning model for Iris flower classification using PyTorch inside Docker containers. The goal is to showcase the end-to-end pipeline from data preprocessing to model inference, all encapsulated within Docker for reproducibility and modularity.

---

## What Has Been Done
Iris_dataset/
│
├── data/

│   ├── iris_train.csv             # Preprocessed training data
│   └── iris_inference.csv         # Preprocessed inference data
│

├── training/

│   ├── Dockerfile                 # Dockerfile for training
│   ├── train.py                   # PyTorch training script
│   ├── model.pth                  # Trained model output
│   └── requirements.txt           # Python packages for training

│
├── inference/

│   ├── Dockerfile                 # Dockerfile for inference
│   ├── inference.py               # Batch inference script
│   └── requirements.txt           # Python packages for inference
│

├── scripts/

│   └── preprocess.py              # Data download and split script
│

├── tests/

│   ├── test_training.py           # Unit tests for training
│   └── test_inference.py          # Unit tests for inference
│
├── .gitignore                     # Ignore model files, logs, etc.
└── README.md  

# This documentation
#git clone https://github.com/yourusername/Iris_dataset.git
#cd Iris_dataset
#python scripts/preprocess.py


---

##  What Has Been Done

### 1.  Data Downloading & Preprocessing
- **Dataset**: The Iris dataset from [Wikipedia](https://en.wikipedia.org/wiki/Iris_flower_data_set).
- **Steps**:
  - Download and load dataset.
  - Clean and format into structured CSV.
  - Split into:
    - `iris_train.csv` – for training the model.
    - `iris_inference.csv` – for performing inference.
- Script: `scripts/preprocess.py`

### 2.  Model Training with PyTorch (Dockerized)
- A simple feedforward neural network using PyTorch.
- Trained on `iris_train.csv`.
- Saves the trained model as `model.pth`.
- Fully Dockerized for environment isolation.

### 3.  Batch Inference (Dockerized)
- Loads the trained model (`model.pth`).
- Performs inference on `iris_inference.csv`.
- Output is printed or optionally saved.
- Also containerized for modular, reproducible execution.

### 4. Unit Testing & Error Handling
- Unit tests included:
  - `test_training.py` for verifying training pipeline.
  - `test_inference.py` for validating inference logic.
- Exception handling for robustness.
- Inline comments for clarity and maintainability.

---

##  Dockerized Workflow
#commands :
### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Iris_dataset.git
cd Iris_dataset
# ----------------------------------------------
# Clone the GitHub Repository
# ----------------------------------------------
git clone https://github.com/yourusername/Iris_dataset.git
cd Iris_dataset

# ----------------------------------------------
 #  Step 1: Preprocess the Iris dataset
# This script downloads, cleans, and splits the dataset
# ----------------------------------------------
python scripts/preprocess.py

# ----------------------------------------------
# Step 2: Build Docker Image for Model Training
# This image contains PyTorch and training dependencies
# ----------------------------------------------
docker build -f training/Dockerfile -t iris-train .

# ----------------------------------------------
# Step 3: Run Docker Container for Training
# It will read iris_train.csv, train the model, and save model.pth
# ----------------------------------------------
docker run --rm \
  -v ${PWD}/data:/app/data \                # Mount data directory
  -v ${PWD}/training:/app/training \        # Mount training directory (for saving model)
  iris-train                                # Docker image name

# ----------------------------------------------
# Step 4: Build Docker Image for Inference
# This image loads the trained model and runs batch inference
# ----------------------------------------------
docker build -f inference/Dockerfile -t iris-infer .

# ----------------------------------------------
# Step 5: Run Docker Container for Inference
# It will read iris_inference.csv and output predictions
# ----------------------------------------------
docker run --rm \ using file path in place of PWD
  -v ${PWD}/data:/app/data \                # Mount data directory
  -v ${PWD}/training:/app/training \        # Mount model directory (model.pth)
  iris-infer                                 # Docker image name

# ----------------------------------------------
# Step 6 (Optional): Run Unit Tests
# Tests model training and inference using pytest
# ----------------------------------------------
pytest tests/test_training.py
pytest tests/test_inference.py
# requirements.txt (same for training & inference for simplicity)
torch              # PyTorch
pandas             # Data manipulation
scikit-learn       # For accuracy, train-test split, preprocessing
numpy              # Numerical operations

# Optional: for unit testing
pytest
#pytest tests/test_training.py
#pytest tests/test_inference.py

Step : Model Evaluation – Confusion Matrix
You can add a section like this under your evaluation or results section in the README.md
--------------------------------------------------------------------------------------------------




