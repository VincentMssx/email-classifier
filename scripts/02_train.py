import argparse
import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.types.schema import Schema, TensorSpec
from mlflow.models import ModelSignature

# Import our custom classes
from src.model import NewsClassifier

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    return fig

# --- Main Training Function ---
def train(args):
    # --- 1. MLFlow Setup ---
    mlflow.set_experiment("NewsGroup Classification")
    with mlflow.start_run() as run:
        print("Starting run:", run.info.run_id)
        
        # --- 2. Load Data ---
        X_train, y_train = joblib.load('data/processed/train.pkl')
        X_test, y_test = joblib.load('data/processed/test.pkl')
        target_names = joblib.load('data/processed/target_names.pkl')
        
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # --- 3. Log Parameters and Tags ---
        mlflow.log_params(vars(args))
        mlflow.set_tag("model_type", "Deep Neural Network")
        mlflow.set_tag("framework", "PyTorch")

        # --- 4. Model, Loss, and Optimizer ---
        model = NewsClassifier(
            input_dim=X_train.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=len(target_names),
            activation_fn_name=args.activation
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # --- 5. Training Loop ---
        for epoch in range(args.epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # --- Log metrics per epoch ---
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                val_loss = criterion(test_outputs, y_test_tensor).item()
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = accuracy_score(y_test, predicted.numpy())
            
            mlflow.log_metric("validation_loss", val_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{args.epochs}, Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

        # --- 6. Final Evaluation & Logging ---
        model.eval()
        with torch.no_grad():
            final_outputs = model(X_test_tensor)
            probs = torch.nn.functional.softmax(final_outputs, dim=1).numpy()
            _, final_preds = torch.max(final_outputs, 1)
            final_preds = final_preds.numpy()

        metrics = {
            "f1_score_weighted": f1_score(y_test, final_preds, average='weighted'),
            "precision_weighted": precision_score(y_test, final_preds, average='weighted'),
            "recall_weighted": recall_score(y_test, final_preds, average='weighted'),
            "roc_auc_ovr_weighted": roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')
        }
        mlflow.log_metrics(metrics)
        print("Final Metrics:", metrics)

        # Log confusion matrix
        cm_fig = plot_confusion_matrix(y_test, final_preds, target_names)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")

        # --- 7. Save artifacts that will be packaged with the model ---
        mlflow.log_artifact('data/processed/tfidf_vectorizer.pkl')
        mlflow.log_artifact('data/processed/target_names.pkl')

        # --- 8. Log the PyTorch Model ---
        print("Logging PyTorch model...")
        
        # Define input signature for the model
        input_schema = Schema([
            TensorSpec(np.dtype(np.float32), (-1, X_train.shape[1]))
        ])
        output_schema = Schema([
            TensorSpec(np.dtype(np.float32), (-1, len(target_names)))
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name=args.model_name,
            signature=signature,
            registered_model_name=args.model_name,
            pip_requirements=["torch", "scikit-learn", "pandas", "numpy", "joblib"]
        )
        
        print(f"\nModel '{args.model_name}' has been trained and registered in the MLFlow Model Registry.")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function (relu, leaky_relu, tanh)")
    parser.add_argument("--model_name", type=str, default="NewsFlow-Classifier", help="Name for the registered model")
    
    args = parser.parse_args()
    train(args)
