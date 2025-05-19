# 📦 Imports
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import shutil
import sys

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ✅ Safe config import for both .py and .ipynb
if "__file__" in globals():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
else:
    base_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

sys.path.append(base_dir)
from config.config import CONFIG

# 🔁 Clear previous MLruns
mlruns_dir = CONFIG["MLFLOW_TRACKING_URI"].replace("file:///", "").replace("file://", "")
if os.path.exists(mlruns_dir):
    shutil.rmtree(mlruns_dir)
    print("🧹 Cleared previous MLflow runs")

# 🔗 Set tracking
mlflow.set_tracking_uri(CONFIG["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(CONFIG["EXPERIMENT_NAME"])

# 📊 Load Data
df = pd.read_csv(os.path.join(base_dir, "data", "telco_churn.csv"))
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df.drop(columns=['customerID'], errors='ignore', inplace=True)

# 📚 Feature Engineering
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

X = df[numerical_cols + categorical_cols]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧱 Preprocessing
numeric_transformer = Pipeline([('scaler', StandardScaler())])
categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# 🚀 Logging Function
def evaluate_and_log(model_pipeline, run_name, params=None):
    with mlflow.start_run(run_name=run_name):
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model_type", run_name)
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model_pipeline, "model")
        print(f"✅ Logged {run_name} → acc: {acc:.3f}, f1: {f1:.3f}")

# 🧠 Run 1: XGBoost Default
xgb_default = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(eval_metric='logloss'))
])
evaluate_and_log(xgb_default, "XGBoost_Default")

# 🧠 Run 2: XGBoost Balanced
xgb_bal = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(scale_pos_weight=2, eval_metric='logloss'))
])
evaluate_and_log(xgb_bal, "XGBoost_Balanced", {"scale_pos_weight": 2})

# 🧠 Run 3: Logistic Regression Balanced
logreg = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])
evaluate_and_log(logreg, "Logistic_Regression_Balanced", {"class_weight": "balanced"})
