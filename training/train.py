import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
import time



class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.model(x)
    

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
def train_model():
    logging.info("Loading training data...")
    data_path = "data/train.csv"
    model_dir = "models"
    model_path = os.path.join(model_dir, "model.pth")
    
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        print(f"[INFO] Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        X = df.iloc[:, :-1].values
        y = LabelEncoder().fit_transform(df["species"])

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = IrisNet()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        print("[INFO] Training started...")
        for epoch in range(50):
            for X_batch, y_batch in loader:
                preds = model(X_batch)
                loss = loss_fn(preds, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), model_path)
        print(f"[INFO] Model trained and saved at {model_path}")
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")

if __name__ == "__main__":
    train_model()
