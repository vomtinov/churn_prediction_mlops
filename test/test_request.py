import requests

url = "http://127.0.0.1:1234/invocations"
headers = {"Content-Type": "application/json"}

# ✅ Use plain JSON records (not NumPy)
payload = {
    "dataframe_records": [{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 1397.47
    }]
}

response = requests.post(url, json=payload, headers=headers)
print("✅ Response:", response.json())
