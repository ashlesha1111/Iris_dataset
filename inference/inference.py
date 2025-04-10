import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from training.train import IrisNet
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def batch_inference(input_csv="data/test.csv", model_path="models/model.pth", output_csv="data/inference_output.csv", cm_path="data/confusion_matrix.png"):
    try:
        logger.info("Loading input data...")
        df = pd.read_csv(input_csv)
        X = df.iloc[:, :-1].values
        y_true = df["species"].values

        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")

        logger.info("Loading model...")
        model = IrisNet()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            preds = model(X_tensor)
            y_pred = torch.argmax(preds, dim=1).numpy()

        # Encode and decode labels
        encoder = LabelEncoder()
        encoder.fit(y_true)
        decoded_preds = encoder.inverse_transform(y_pred)

        # Save inference output
        result = pd.DataFrame({"Actual": y_true, "Predicted": decoded_preds})
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        result.to_csv(output_csv, index=False)
        logger.info(f"Inference complete. Output saved to {output_csv}")

        # Generate and save confusion matrix
        cm = confusion_matrix(y_true, decoded_preds, labels=encoder.classes_)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        plt.savefig(cm_path)
        logger.info(f"Confusion matrix saved to {cm_path}")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise  # Re-raise the exception so it can be caught in unittests

if __name__ == "__main__":
    batch_inference()
