import pandas as pd
from sklearn.model_selection import train_test_split
import os

def download_iris_data():
    """Download and split the Iris dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

    os.makedirs("data", exist_ok=True)

    try:
        df = pd.read_csv(url, names=column_names)
        df.dropna(inplace=True)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['species'])
        train_df.to_csv("data/train.csv", index=False)
        test_df.to_csv("data/test.csv", index=False)
        print("[INFO] Data downloaded and split successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to process data: {e}")

if __name__ == "__main__":
    download_iris_data()
