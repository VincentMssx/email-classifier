
import mlflow
from mlflow.tracking import MlflowClient

# Initialize MLflow client
client = MlflowClient()

# Name of the experiment
experiment_name = "NewsGroup Classification"

# Get the experiment
experiment = client.get_experiment_by_name(experiment_name)

# Search for the best run in the experiment
runs = client.search_runs(
    experiment_ids=experiment.experiment_id,
    order_by=["metrics.accuracy DESC"],
    max_results=1
)

# Get the best run
best_run = runs[0]

# Print the best run's parameters and metrics
print("Best run ID:", best_run.info.run_id)
print("Best run parameters:", best_run.data.params)
print("Best run metrics:", best_run.data.metrics)
