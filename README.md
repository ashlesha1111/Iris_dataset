# **Iris Classification with Training and Inference Pipelines**

This repository contains a complete machine learning pipeline for training and inference on the Iris dataset. The project leverages Docker containers for encapsulating the training and inference processes, ensuring reproducibility and portability.

---

## **Project Structure**

```
mle-practice/
│
├── training/
│   ├── train_model.py          # Training script
│   ├── model/                  # Directory to save the trained model
│   ├──DockerFile               # Dockerfile for training
│   └──requirements.txt         # Requirements for training
│
├── inference/
│   ├── inference.py            # Inference script
│   ├──DockerFile               # Dockerfile for inference
│   └──requirements.txt         # Requirements for inference
│
├── data/
│   ├── train.csv               # Training data file
│   ├── inference.csv           # Inference data file
│
├── tests/
│   ├── testing.py              # Unit tests for training and inference
│
├── data_processing.py          # Python script for getting data
├── requirements.txt            # Python dependencies
├── .gitignore                  # Ignored files and directories
└── README.md                   # Project documentation
```

---

## **Setup Instructions**

### **Prerequisites**
1. Install [Docker](https://docs.docker.com/get-docker/).
2. Clone this repository:
   ```bash
   git clone https://github.com/ashlesha1111/Iris_dataset.git
   
   ```

---

## **Training**

### **Build the Docker Image for Training**
```bash
docker build -f Dockerfile-training -t Iris_dataset:training .
```

### **Run the Training Container**
```bash
docker run --rm -v $(pwd)/training/model:/app/model -v $(pwd)/data:/app/data Iris_dataset:training
```

The trained model will be saved in the `training/model/` directory as `model.pth`.
![train model outout](assets/train model outout.png)


---

## **Inference**

### **Build the Docker Image for Inference**
```bash
docker build -f Dockerfile-inference -t Iris_dataset:inference .
```

### **Run the Inference Container**
```bash
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/training/model:/app/model mlepractice:inference
```

The inference results will be saved as `inference_results.csv` in the `data/` directory.

---

## **Running Tests**

Unit tests ensure the integrity of the training and inference pipelines.

### **Run Tests**
```bash
python tests/testing.py
```


---

1. **Training Image**  
   ```bash
   docker pull aryankandari500/mlepractice:iris-training
   ```

2. **Inference Image**  
   ```bash
   docker pull aryankandari500/mlepractice:inference
   ```

---

## **Project Highlights**
1. **Training Pipeline**
   - Trains a simple neural network for classifying Iris flower species.
   - Outputs a saved model (`model.pth`) for inference.

2. **Inference Pipeline**
   - Loads the saved model and performs predictions on unseen data.
   - Outputs predictions in a CSV file (`inference_results.csv`).

3. **Modular Design**
   - Separate Dockerfiles for training and inference.
   - Organized directory structure for better maintainability.

4. **Unit Tests**
   - Validates the forward pass, loss computation, and file generation.


---

Feel free to reach out for any issues or improvements. Enjoy experimenting with Iris classification! 
