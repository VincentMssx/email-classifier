# NewsFlow Classifier

This project is a machine learning application for classifying newsgroup posts (similar to emails) into different categories. It uses a PyTorch-based neural network for classification and is fully integrated with MLflow to manage the entire machine learning lifecycle, from experiment tracking to model deployment.

## Features

- **Text Classification**: Trains a deep neural network to classify text into one of 20 newsgroup categories.
- **End-to-End MLOps**: Uses **MLflow** for:
    - **Experiment Tracking**: Logging parameters, metrics, and artifacts for every training run.
    - **Model Registry**: Versioning and managing models for production.
    - **Reproducibility**: Packaging the model with all its dependencies.
- **Modular Structure**: The code is organized into clear, distinct scripts for data preparation, training, and prediction.

## Tech Stack

- **Python 3.12**
- **PyTorch**: For building and training the neural network.
- **Scikit-learn**: For text vectorization (TF-IDF).
- **MLflow**: For MLOps and experiment management.
- **Pandas & NumPy**: For data manipulation.
- **Joblib**: For saving and loading Python objects.

## Project Structure

```
.
├── data/
│   └── processed/      # Stores processed data, vectorizers, and target names
├── mlruns/             # Default directory for MLflow tracking data
├── scripts/
│   ├── 01_prepare_data.py  # Script to process raw text data
│   ├── 02_train.py         # Script to train the model and log to MLflow
│   ├── 03_promote_model.py # Script to promote a model version in the registry
│   └── 04_predict.py       # Script to make predictions with a registered model
├── src/
│   ├── model.py            # Defines the PyTorch neural network architecture
│   └── custom_model.py     # (Legacy) Wrapper for custom pyfunc model
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Setup and Installation

Follow these steps to set up the project environment.

**1. Prerequisites**
- Python 3.12 or later
- `pip` package manager

**2. Create a Virtual Environment**
It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate the environment
venv\Scripts\activate
```

**3. Install Dependencies**
Install all the required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Usage

Follow this sequence to run the full pipeline.

### Step 1: Prepare the Data

First, run the data preparation script. This will process the raw text, create the TF-IDF vectorization, and save the training and testing datasets.

```bash
python scripts/01_prepare_data.py
```
This will create `train.pkl`, `test.pkl`, `tfidf_vectorizer.pkl`, and `target_names.pkl` in the `data/processed/` directory.

### Step 2: Train the Model

Next, run the training script. This will train the neural network, log all results to MLflow, and register the trained model in the MLflow Model Registry.

```bash
python scripts/02_train.py --epochs 20 --lr 0.001
```
You can customize the training by providing different arguments:
- `--epochs`: Number of training epochs (e.g., `20`).
- `--lr`: Learning rate (e.g., `0.001`).
- `--hidden_dim`: Dimension of the hidden layers (e.g., `256`).
- `--model_name`: Name for the registered model in MLflow (e.g., `"NewsFlow-Classifier"`).

### Step 3: Explore Results in the MLflow UI

To visualize your experiments, compare runs, and see the registered models, launch the MLflow UI.

```bash
# Run this from the project's root directory
mlflow ui
```
Open your browser and navigate to `http://127.0.0.1:5000`.

### Step 4: Make Predictions

Once a model is trained and registered, you can use the prediction script to classify new text. This script fetches the model and its artifacts directly from the MLflow Model Registry.

```bash
# Predict using the model version currently in the "Staging" stage
python scripts/04_predict.py --model_name "NewsFlow-Classifier" --version "Staging"

# Or predict using a specific version number
python scripts/04_predict.py --model_name "NewsFlow-Classifier" --version "3"
```

## Model Lifecycle with MLflow

This project uses the MLflow Model Registry to manage the model lifecycle.

1.  **Registration**: The `02_train.py` script automatically registers a new version of the `NewsFlow-Classifier` model after each successful training run.
2.  **Promotion**: The `03_promote_model.py` script can be used to transition a model version to a different stage (e.g., from `None` to `Staging` or from `Staging` to `Production`). This provides a structured workflow for deploying models.
3.  **Inference**: The `04_predict.py` script demonstrates how to load a model by its stage or version for inference, ensuring that your application always uses the correct, approved model.
