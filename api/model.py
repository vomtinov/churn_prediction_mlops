import mlflow
import pandas as pd
from config.config import CONFIG

# Load model once during startup
model = mlflow.sklearn.load_model(CONFIG['best_model_path'])

def predict_churn(input_data: dict):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return {"prediction": int(prediction), "probability": round(probability, 4)}
