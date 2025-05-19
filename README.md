# 🔮 Churn Prediction MLOps Pipeline

An end-to-end machine learning pipeline for predicting customer churn, integrating **MLflow** for experiment tracking and **Streamlit** for an interactive user interface.

---

## 🚀 Features

- ✅ **Comprehensive ML Pipeline**: From data preprocessing and model training to evaluation and deployment.
- ✅ **Automated Model Selection**: Automatically selects the best-performing model based on accuracy metrics.
- ✅ **Interactive Streamlit App**: User-friendly interface for real-time churn prediction.
- ✅ **MLflow Integration**: Tracks experiments, parameters, and metrics for reproducibility.
- ✅ **Modular Codebase**: Well-structured and easy to navigate for scalability and maintenance.

---

## 🧰 Tech Stack

| Layer                | Tools / Frameworks           |
|---------------------|------------------------------|
| Programming         | Python                       |
| Data Processing     | Pandas, NumPy                |
| Modeling            | Scikit-learn, XGBoost        |
| Experiment Tracking | MLflow                       |
| Interface           | Streamlit                    |
| Dev Environment     | VSCode, Jupyter Notebooks    |

---

## 📁 Project Structure

churn_prediction_mlops/
├── data/ # Raw and processed data
├── notebooks/ # Jupyter notebooks for EDA and experimentation
├── src/ # Source code for data processing and modeling
├── models/ # Trained and serialized models
├── utils/ # Utility scripts
├── app/ # Streamlit application
├── tests/ # Test scripts
├── MLproject # MLflow project file
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Git ignore file


---

## ⚙️ Setup Instructions

### 🔧 Clone the Repository

```bash
git clone https://github.com/vomtinov/churn_prediction_mlops.git
cd churn_prediction_mlops


## ⚙️ Setup Instructions

### 🔧 Clone the Repository

```bash
git clone https://github.com/vomtinov/churn_prediction_mlops.git
cd churn_prediction_mlops

## 🧪 Create and Activate Virtual Environment

python -m venv venv

# On Windows
venv\Scripts\activate

# On Unix or MacOS
source venv/bin/activate

## 📦 Install Dependencies

pip install -r requirements.txt


##📊 MLflow Experiment Tracking
#Launch MLflow UI

mlflow ui

Then go to:
http://localhost:5000

##🧠 Train the Model
This command will preprocess the data, train models, and log runs to MLflow.
python notebooks/2_model_pipeline.py

##🖥️ Launch Streamlit App

streamlit run app/app.py
http://localhost:8501

##🧪 Run Tests

pytest tests/

🌐 Deployment
Local: Run Streamlit app locally (as shown above).

##Cloud Platforms:

Streamlit Cloud

Heroku

Render

AWS / GCP / Azure


##🤝 Contributing
Contributions are welcome!
Fork the repo → Create a new branch → Commit changes → Open a Pull Request ✅

##📌 Author: vomtinov
⭐ If you like this project, consider starring it!