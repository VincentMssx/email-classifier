import argparse
import mlflow
import joblib
import torch
import os
from mlflow.tracking import MlflowClient

def predict(model_name, version):
    # --- 1. Get Run ID from Model Registry ---
    client = MlflowClient()
    try:
        model_version = client.get_model_version(name=model_name, version=version)
        run_id = model_version.run_id
    except Exception as e:
        print(f"Error getting model version. Could not find version '{version}' for model '{model_name}'.")
        print(f"Please ensure the model name and version/stage are correct.")
        print(f"MLFlow error: {e}")
        return

    print(f"Found run ID: {run_id} for model {model_name} version {version}")

    # --- 2. Load Model and Artifacts ---
    artifact_dir = f"artifacts_for_prediction"
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    # Construct paths for downloaded artifacts
    vectorizer_path = os.path.join(artifact_dir, "tfidf_vectorizer.pkl")
    target_names_path = os.path.join(artifact_dir, "target_names.pkl")

    # Download artifacts if they don't exist
    if not os.path.exists(vectorizer_path):
        client.download_artifacts(run_id, "tfidf_vectorizer.pkl", artifact_dir)
    if not os.path.exists(target_names_path):
        client.download_artifacts(run_id, "target_names.pkl", artifact_dir)

    # Load artifacts
    vectorizer = joblib.load(vectorizer_path)
    target_names = joblib.load(target_names_path)

    # Load the PyTorch model from the run's artifacts
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from: {model_uri}")
    loaded_model = mlflow.pytorch.load_model(model_uri)
    loaded_model.eval()

    # --- 3. Predict ---
    texts_to_predict = [
        "The recent launch of the new GPU has everyone talking about computer graphics performance.",
        "A debate in congress today focused on international space exploration policy.",
        "The best way to fix a car engine is to first understand how combustion works."
    ]
    
    # Preprocess the text
    X_pred = vectorizer.transform(texts_to_predict).toarray()
    X_pred_tensor = torch.FloatTensor(X_pred)

    # Make predictions
    with torch.no_grad():
        outputs = loaded_model(X_pred_tensor)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.numpy()

    print("\n--- Predictions ---")
    for text, pred_index in zip(texts_to_predict, predictions):
        print(f"Text: '{text[:60]}...'\n")
        print(f"  -> Predicted Category: {target_names[pred_index]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the registered model")
    parser.add_argument("--version", type=str, default="Staging", help="Version of the model to use for prediction (e.g., 'Staging', '1')")
    args = parser.parse_args()
    predict(args.model_name, args.version)
