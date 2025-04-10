import os
import pandas as pd

def test_data_created():
    assert os.path.exists("data/train.csv")
    assert os.path.exists("data/test.csv")
    df = pd.read_csv("data/train.csv")
    assert not df.empty

def test_model_saved():
    assert os.path.exists("models/model.pth")
