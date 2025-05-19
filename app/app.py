import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import sys
import os

# 🔧 Add root path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import CONFIG

# 🔍 Load Best Model from MLflow
def get_best_model_uri():
    client = MlflowClient(tracking_uri=CONFIG["MLFLOW_TRACKING_URI"])
    experiment = client.get_experiment_by_name(CONFIG["EXPERIMENT_NAME"])
    
    if experiment is None:
        st.error("⚠️ Experiment not found in MLflow.")
        st.stop()
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    
    if not runs:
        st.error("⚠️ No runs found in MLflow for this experiment.")
        st.stop()
    
    return f"{runs[0].info.artifact_uri}/model"

# 🧠 Load Best Model
model_uri = get_best_model_uri()
model = mlflow.sklearn.load_model(model_uri)

# 🖥️ Streamlit UI
st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("🔮 Customer Churn Prediction (Best Model)")

# 📥 Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.slider("Monthly Charges", 0, 150, 70)
TotalCharges = st.slider("Total Charges", 0, 9000, 2000)

# 🔮 Predict Button
if st.button("Predict"):
    input_df = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("## 🧾 Prediction Result")
    st.success(f"Churn: {'Yes' if prediction == 1 else 'No'}")
    st.info(f"Churn Probability: {prob * 100:.2f}%")



# 🛰️ API Access Info Section
st.markdown("---")
st.markdown("### 🔗 Want to use this model in your own app?")

api_url = "http://127.0.0.1:8000/predict/"  # 🔁 Update this if deploying on Render/Cloud

st.code(api_url, language="markdown")

# Copy to clipboard using a text_input
copied_url = st.text_input("📋 Copy this API endpoint 👇", value=api_url, key="api_copy")

st.caption("Paste this in your app or code to make POST requests for predictions.")

# Optional code sample
with st.expander("📎 Example: How to call this API from Python"):
    st.code(f"""
import requests

url = "{api_url}"
data = {{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70,
    "TotalCharges": 2000
}}

response = requests.post(url, json=data)
print(response.json())
    """, language="python")


import json

with open("config/best_run.json", "r") as f:
    run_info = json.load(f)

run_id = run_info["run_id"]

# For CLI-based serving:
# mlflow models serve -m runs:/<run_id>/model --port 1234