🔮 Churn Prediction MLOps Pipeline
An end-to-end machine learning pipeline for predicting customer churn, integrating MLflow for experiment tracking and Streamlit for an interactive user interface.

🚀 Features
✅ Comprehensive ML Pipeline: From data preprocessing and model training to evaluation and deployment.

✅ Automated Model Selection: Automatically selects the best-performing model based on accuracy metrics.

✅ Interactive Streamlit App: User-friendly interface for real-time churn prediction.

✅ MLflow Integration: Tracks experiments, parameters, and metrics for reproducibility.

✅ Modular Codebase: Well-structured and easy to navigate for scalability and maintenance.

🧰 Tech Stack
Layer	Tools / Frameworks
Programming	Python
Data Processing	Pandas, NumPy
Modeling	Scikit-learn, XGBoost
Experiment Tracking	MLflow
Interface	Streamlit
Dev Environment	VSCode, Jupyter Notebooks

📁 Project Structure
bash
Copy
Edit
churn_prediction_mlops/
├── data/                   # Raw and processed data
├── notebooks/              # Jupyter notebooks for EDA and experimentation
├── src/                    # Source code for data processing and modeling
├── models/                 # Trained and serialized models
├── utils/                  # Utility scripts
├── app/                    # Streamlit application
├── tests/                  # Test scripts
├── MLproject               # MLflow project file
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
⚙️ Setup Instructions
🔧 Clone the Repository
bash
Copy
Edit
git clone https://github.com/vomtinov/churn_prediction_mlops.git
cd churn_prediction_mlops
🧪 Create and Activate Virtual Environment
bash
Copy
Edit
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
📦 Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
📊 MLflow Experiment Tracking
Run MLflow tracking server locally:

bash
Copy
Edit
mlflow ui
Then open your browser to:

arduino
Copy
Edit
http://localhost:5000
🧠 Train the Model
This will clear previous logs, train 3 models, and log them to MLflow.

bash
Copy
Edit
python notebooks/2_model_pipeline.py
🖥️ Launch Streamlit App
bash
Copy
Edit
streamlit run app/app.py
Open http://localhost:8501 in your browser to use the prediction app.

🧪 Run Tests
bash
Copy
Edit
pytest tests/
🌐 Deployment
Local Deployment: Done via streamlit run app/app.py

Cloud Deployment Options:

Streamlit Cloud

Heroku

Render

AWS / GCP / Azure

