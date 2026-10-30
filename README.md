# AI-Workflow-Capstone
Coursera AI Workflow Capstone: Final Deliverable Implementation Guide
. Unit Testing Suite (unittests/)To fulfill the unit testing criteria, implement modular test files under unittests/ that can be executed via a single test runner script.unittests/ApiTests.pyPythonimport unittest
import json
from app import app

class TestApi(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_single_country(self):
        payload = {"country": "United Kingdom", "target_date": "2019-08-01", "test": True}
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("y_pred", data)

    def test_predict_all_countries(self):
        payload = {"country": "all", "target_date": "2019-08-01", "test": True}
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("y_pred", data)

if __name__ == '__main__':
    unittest.main()
unittests/ModelTests.pyPythonimport unittest
import os
import numpy as np
from model import model_train, model_predict

class TestModel(unittest.TestCase):
    def test_01_train(self):
        """Test training workflow in test mode to isolate outputs."""
        model_train(data_dir="data/cs-train", test=True)
        self.assertTrue(os.path.exists("models/test-all-0_1.joblib"))

    def test_02_predict(self):
        """Test predictions with generated test models."""
        result = model_predict("all", "2019-08-01", test=True)
        self.assertIn("y_pred", result)
        self.assertIsInstance(result["y_pred"], (list, np.ndarray))

if __name__ == '__main__':
    unittest.main()
unittests/LoggerTests.pyPythonimport unittest
import os
import csv
from logger import log_prediction, log_training

class TestLogger(unittest.TestCase):
    def test_01_log_prediction(self):
        log_file = "logs/test-predict.log"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        log_prediction(y_pred=[100.0], input_data={"country": "all"}, runtime="0.05", test=True)
        self.assertTrue(os.path.exists(log_file))

    def test_02_log_training(self):
        log_file = "logs/test-train.log"
        if os.path.exists(log_file):
            os.remove(log_file)
            
        log_training(dates=("2017-11-28", "2019-07-19"), performance={"rmse": 10.5}, runtime="1.2", test=True)
        self.assertTrue(os.path.exists(log_file))

if __name__ == '__main__':
    unittest.main()
Single Test Execution Script (run-tests.py)Pythonimport unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover('unittests', pattern='*Tests.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    exit(0 if result.wasSuccessful() else 1)
Isolation Strategy: Notice the test=True parameter passed to model training, predictions, and logging functions. This routes reading and writing operations to isolated test subdirectories (models/test-*, logs/test-*), preventing corruption of production artifacts.2. Containerization (Dockerfile)DockerfileFROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose API port
EXPOSE 8080

# Run all unit tests during build to ensure integrity
RUN python run-tests.py

CMD ["python", "app.py"]
3. Post-Production Performance & Drift Monitoring Script (monitor.py)Pythonimport os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def evaluate_drift(log_file_path, actual_data_path):
    """Calculates performance degradation over time and visualizes actuals vs baseline."""
    if not os.path.exists(log_file_path):
        print(f"Log file not found: {log_file_path}")
        return

    # Load prediction logs and actual target data
    logs_df = pd.read_csv(log_file_path)
    actuals_df = pd.read_csv(actual_data_path)
    
    # Merge on timestamp/date index
    merged = pd.merge(logs_df, actuals_df, on="date")
    
    # Calculate RMSE metric over rolling windows
    merged['rmse'] = np.sqrt((merged['y_pred'] - merged['y_actual'])**2)
    
    # Plot baseline vs model predictions
    plt.figure(figsize=(10, 5))
    plt.plot(merged['date'], merged['y_actual'], label='Actual Revenue', color='black')
    plt.plot(merged['date'], merged['y_pred'], label='Model Forecast', color='blue', linestyle='--')
    plt.plot(merged['date'], merged['baseline_pred'], label='Baseline (Naive)', color='red', linestyle=':')
    plt.title('Post-Production Revenue Forecast vs Baseline')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.legend()
    plt.tight_layout()
    plt.savefig('logs/performance_monitoring.png')
    plt.close()

if __name__ == '__main__':
    evaluate_drift("logs/predict-production.log", "data/cs-production/actuals.csv")
4. Submission Checklist MatrixEvaluation Checklist ItemImplementation DetailsUnit tests for API?Defined in unittests/ApiTests.py using Flask.test_client().Unit tests for Model?Defined in unittests/ModelTests.py covering train/predict.Unit tests for Logging?Defined in unittests/LoggerTests.py for training and inference logs.Single test runner script?Yes, python run-tests.py runs all discoverable tests and returns exit code 0.Isolate test read/write?Yes, test=True flag writes artifacts to /test- designated paths.API handles specific & all countries?Endpoint accepts {"country": "United Kingdom"} and {"country": "all"}.Automated data ingestion?Function cslib.fetch_data() aggregates raw JSONs into structured time-series DataFrames.Multiple models compared?Comparison executed across Random Forest, Gradient Boosting, and Baseline Naive models during model_train().EDA and baseline comparison visual?Visualizations saved during EDA and model evaluation phases (performance_monitoring.png).Dockerized workflow?Complete repository containerized and built via standard Docker build commands.
