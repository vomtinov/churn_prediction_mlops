import json
from mlflow.tracking import MlflowClient

def save_best_run_id(experiment_name="Churn_Model_Comparison", save_path="config/best_run.json"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )[0]

    print("✅ Best Run ID:", best_run.info.run_id)

    with open(save_path, "w") as f:
        json.dump({"run_id": best_run.info.run_id}, f)

if __name__ == "__main__":
    save_best_run_id()
