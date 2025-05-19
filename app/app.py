import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# 🔍 Find the best model from MLflow
def get_best_model_uri():
    client = MlflowClient(tracking_uri="file:///D:/projects/churn_prediction_mlops/mlruns")
    experiment = client.get_experiment_by_name("Churn_Model_Comparison")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    best_run = runs[0]
    return f"{best_run.info.artifact_uri}/model"

# ✅ Load best model
model_uri = get_best_model_uri()
model = mlflow.sklearn.load_model(model_uri)

# 🖥️ Streamlit UI
st.title("🔮 Customer Churn Prediction (Best Model)")

# 👤 Inputs
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

# 🔮 Predict
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

    st.markdown("## 🧾 Prediction Result:")
    st.success(f"Churn: {'Yes' if prediction == 1 else 'No'}")
    st.info(f"Churn Probability: {prob*100:.2f}%")
