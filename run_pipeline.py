import sys
import subprocess

print("✅ Running model pipeline script...")
subprocess.run([sys.executable, "notebooks/2_model_pipeline.py"], check=True)

print("🚀 Launching Streamlit app...")
subprocess.run(["streamlit", "run", "app/app.py"])

from utils.get_best_run import save_best_run_id
save_best_run_id("Churn_Model_Comparison")