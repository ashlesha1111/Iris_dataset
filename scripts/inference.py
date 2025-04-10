# import torch
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from train import IrisNet
# import torch.utils.data
# from torch.utils.data import DataLoader, Dataset

# def batch_inference():
#     try:
#         df = pd.read_csv("data/test.csv")
#         X = df.iloc[:, :-1].values
#         y_true = df["species"].values

#         model = IrisNet()
#         model.load_state_dict(torch.load("models/model.pth"))
#         model.eval()

#         X_tensor = torch.tensor(X, dtype=torch.float32)

#         with torch.no_grad():
#             preds = model(X_tensor)
#             y_pred = torch.argmax(preds, dim=1).numpy()

#         encoder = LabelEncoder()
#         encoder.fit(y_true)
#         decoded_preds = encoder.inverse_transform(y_pred)

#         result = pd.DataFrame({"Actual": y_true, "Predicted": decoded_preds})
#         result.to_csv("data/inference_output.csv", index=False)
#         print("[INFO] Inference complete.")
#     except Exception as e:
#         print(f"[ERROR] Inference failed: {e}")

# if __name__ == "__main__":
#     batch_inference()

import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from train import IrisNet
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def batch_inference():
    try:
        df = pd.read_csv("data/test.csv")
        X = df.iloc[:, :-1].values
        y_true = df["species"].values

        # Load model
        model = IrisNet()
        model.load_state_dict(torch.load("models/model.pth"))
        model.eval()

        # Convert input to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            preds = model(X_tensor)
            y_pred = torch.argmax(preds, dim=1).numpy()

        # Encode labels
        encoder = LabelEncoder()
        encoder.fit(y_true)
        decoded_preds = encoder.inverse_transform(y_pred)

        # Save CSV
        result = pd.DataFrame({"Actual": y_true, "Predicted": decoded_preds})
        result.to_csv("data/inference_output.csv", index=False)
        print("[INFO] Inference complete.")

        # Generate and save confusion matrix
        cm = confusion_matrix(y_true, decoded_preds, labels=encoder.classes_)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=encoder.classes_, yticklabels=encoder.classes_)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        os.makedirs("data", exist_ok=True)
        plt.savefig("data/confusion_matrix.png")
        print("[INFO] Confusion matrix saved to data/confusion_matrix.png")

    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")

if __name__ == "__main__":
    batch_inference()
