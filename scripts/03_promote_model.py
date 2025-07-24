import argparse
import mlflow
from mlflow.exceptions import MlflowException

def promote_model(model_name, alias):
    """
    Promote a model using the new MLflow Model Registry API with aliases
    instead of the deprecated stages system.
    """
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Get all versions of the model (new approach)
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if not all_versions:
            print(f"No models with name '{model_name}' found.")
            return
        
        # Find the latest version (highest version number)
        latest_version = max(all_versions, key=lambda v: int(v.version))
        latest_version_number = latest_version.version
        
        print(f"Found latest version: {latest_version_number}. Setting alias '{alias}'.")
        
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=latest_version_number
        )
        
        print(f"Model '{model_name}' version {latest_version_number} has been assigned alias '{alias}'.")

    except MlflowException as e:
        print(f"MLflow error promoting model: {e}")
    except Exception as e:
        print(f"Unexpected error promoting model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model in the registry")
    parser.add_argument("--stage", type=str, default="Staging", help="Stage to promote the model to (e.g., Staging, Production)")
    args = parser.parse_args()
    promote_model(args.model_name, args.stage)