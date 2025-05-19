import os

CONFIG = {
    "MLFLOW_TRACKING_URI": f"file:///{os.path.abspath(os.path.join(os.getcwd(), 'mlruns'))}",
    "EXPERIMENT_NAME": "Churn_Model_Comparison"
}