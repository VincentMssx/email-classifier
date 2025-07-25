import argparse
import itertools
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
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

# Import our custom classes
from src.model import NewsClassifier

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Generates and returns a matplotlib figure containing the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    return fig

def train_single_model(params):
    """
    Trains a single model with the given parameters and logs the results to MLflow.
    """
    mlflow.set_experiment("NewsGroup Classification")
    with mlflow.start_run(run_name=f"grid_search_{params['lr']}_{params['hidden_dim']}_{params['activation']}") as run:
        run_id = run.info.run_id
        print(f"Starting run: {run_id} with params: {params}")

        # Log parameters
        mlflow.log_params(params)
        mlflow.set_tag("model_type", "Deep Neural Network")
        mlflow.set_tag("framework", "PyTorch")

        # Load data
        try:
            X_train, y_train = joblib.load('data/processed/train.pkl')
            X_test, y_test = joblib.load('data/processed/test.pkl')
            target_names = joblib.load('data/processed/target_names.pkl')
        except FileNotFoundError:
            print("Error: Data files not found. Please run the data preparation script first.")
            return

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

        # Initialize model, criterion, and optimizer
        model = NewsClassifier(
            input_dim=X_train.shape[1],
            hidden_dim=params['hidden_dim'],
            output_dim=len(target_names),
            activation_fn_name=params['activation']
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])

        # Training loop
        for epoch in range(params['epochs']):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Log metrics per epoch
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                val_loss = criterion(test_outputs, y_test_tensor).item()
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
            
            mlflow.log_metric("validation_loss", val_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)

        # Final evaluation and logging
        model.eval()
        with torch.no_grad():
            final_outputs = model(X_test_tensor)
            probs = torch.nn.functional.softmax(final_outputs, dim=1).numpy()
            _, final_preds = torch.max(final_outputs, 1)
            final_preds = final_preds.numpy()

        metrics = {
            "f1_score_weighted": f1_score(y_test_tensor.numpy(), final_preds, average='weighted'),
            "precision_weighted": precision_score(y_test_tensor.numpy(), final_preds, average='weighted'),
            "recall_weighted": recall_score(y_test_tensor.numpy(), final_preds, average='weighted'),
            "roc_auc_ovr_weighted": roc_auc_score(y_test_tensor.numpy(), probs, multi_class='ovr', average='weighted')
        }
        mlflow.log_metrics(metrics)
        print(f"Run {run_id} final metrics: {metrics}")

        # Log confusion matrix
        cm_fig = plot_confusion_matrix(y_test_tensor.numpy(), final_preds, target_names)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")

        # Log model artifacts
        mlflow.log_artifact('data/processed/tfidf_vectorizer.pkl')
        mlflow.log_artifact('data/processed/target_names.pkl')

        # Log the PyTorch model
        input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, X_train.shape[1]))])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, len(target_names)))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name=params['model_name'],
            signature=signature,
            registered_model_name=params['model_name']
        )
        print(f"Model '{params['model_name']}' has been trained and registered in the MLFlow Model Registry.")

def run_grid_search(args):
    """
    Runs a grid search over the specified hyperparameter space.
    """
    param_grid = {
        'lr': [0.01, 0.001],
        'hidden_dim': [128, 256],
        'batch_size': [64, 128],
        'epochs': [10, 20],
        'activation': ['relu', 'tanh']
    }

    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Starting grid search with {len(param_combinations)} combinations...")
    for params in param_combinations:
        # Add model_name to the params dictionary
        params['model_name'] = args.model_name
        train_single_model(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a News Classifier model.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--activation", type=str, default="relu", choices=['relu', 'tanh', 'leaky_relu'], help="Activation function")
    parser.add_argument("--model_name", type=str, default="NewsFlow-Classifier", help="Name for the registered model")
    parser.add_argument("--grid_search", action="store_true", help="Enable grid search for hyperparameter tuning")

    args = parser.parse_args()

    if args.grid_search:
        run_grid_search(args)
    else:
        params = {
            'lr': args.lr,
            'hidden_dim': args.hidden_dim,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'activation': args.activation,
            'model_name': args.model_name
        }
        train_single_.model(params)
