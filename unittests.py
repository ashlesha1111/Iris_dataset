import unittest
import os
import pandas as pd
import torch

from training.train import train_model
from training.train import IrisNet
from inference.inference import batch_inference


class TestIrisPipeline(unittest.TestCase):

    def setUp(self):
        self.train_data_path = "data/train.csv"
        self.test_data_path = "data/test.csv"
        self.model_path = "models/model.pth"
        self.inference_output = "data/inference_output.csv"
        self.conf_matrix_path = "data/confusion_matrix.png"

    def test_training_creates_model_file(self):
        """Test if training process creates model file"""
        train_model()
        self.assertTrue(os.path.exists(self.model_path), "Trained model file not found!")

    def test_model_structure(self):
        """Test model structure and output shape"""
        model = IrisNet()
        sample_input = torch.randn(10, 4)
        output = model(sample_input)
        self.assertEqual(output.shape, (10, 3), "Model output shape is incorrect!")

    def test_batch_inference_runs(self):
        """Test if batch inference runs and creates output file"""
        batch_inference()
        self.assertTrue(os.path.exists(self.inference_output), "Inference output file not found!")
        self.assertTrue(os.path.exists(self.conf_matrix_path), "Confusion matrix image not found!")

    def test_inference_output_format(self):
        """Test the format of the inference output file"""
        if os.path.exists(self.inference_output):
            df = pd.read_csv(self.inference_output)
            self.assertIn("Actual", df.columns)
            self.assertIn("Predicted", df.columns)
            self.assertGreater(len(df), 0, "Inference output file is empty!")

    def test_model_file_integrity(self):
        """Test loading of the saved model file"""
        model = IrisNet()
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        sample_input = torch.randn(5, 4)
        with torch.no_grad():
            preds = model(sample_input)
        self.assertEqual(preds.shape, (5, 3), "Loaded model's predictions are not correct!")

    def test_missing_model_raises_error(self):
        """Test if inference raises error when model is missing"""
        backup_path = self.model_path + ".bak"
        if os.path.exists(self.model_path):
            os.rename(self.model_path, backup_path)
        try:
            with self.assertRaises(Exception):
                batch_inference()
        finally:
            if os.path.exists(backup_path):
                os.rename(backup_path, self.model_path)


if __name__ == "__main__":
    unittest.main()
