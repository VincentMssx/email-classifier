import joblib
import pandas as pd
import torch
import numpy as np
import json
import os
import mlflow.pyfunc
from src.model import NewsClassifier

class TfidfPytorchWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow pyfunc model that combines TF-IDF vectorization with PyTorch classification
    """
    
    def load_context(self, context):
        """Load model artifacts during model loading"""
        try:
            # Load vectorizer
            vectorizer_path = context.artifacts["vectorizer_path"]
            self.vectorizer = joblib.load(vectorizer_path)
            print(f"Loaded vectorizer from: {vectorizer_path}")
            
            # Load target names
            target_names_path = context.artifacts["target_names_path"]
            self.target_names = joblib.load(target_names_path)
            print(f"Loaded target names from: {target_names_path}")
            
            # Load model configuration
            if "model_config_path" in context.artifacts:
                config_path = context.artifacts["model_config_path"]
                with open(config_path, 'r') as f:
                    self.model_config = json.load(f)
            else:
                # Fallback to direct config if available
                self.model_config = context.artifacts.get("model_config", {})
                
            print(f"Model config: {self.model_config}")
            
            # Initialize and load PyTorch model
            self.pytorch_model = NewsClassifier(
                input_dim=self.model_config["input_dim"],
                hidden_dim=self.model_config["hidden_dim"],
                output_dim=self.model_config["output_dim"],
                activation_fn_name=self.model_config["activation"]
            )
            
            # Load model state dict
            pytorch_model_path = context.artifacts["pytorch_model_path"]
            self.pytorch_model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
            self.pytorch_model.eval()
            print(f"Loaded PyTorch model from: {pytorch_model_path}")
            
        except Exception as e:
            print(f"Error in load_context: {e}")
            raise
    
    def predict(self, context, model_input):
        """
        Make predictions on input data
        
        Args:
            context: MLflow context (passed automatically)
            model_input: pandas DataFrame with text data
            
        Returns:
            numpy array of predicted class indices
        """
        try:
            # Handle different input formats
            if isinstance(model_input, pd.DataFrame):
                if len(model_input.columns) == 1:
                    # Single column DataFrame
                    texts = model_input.iloc[:, 0].tolist()
                else:
                    # Multiple columns, take the first one
                    texts = model_input.iloc[:, 0].tolist()
            elif isinstance(model_input, (list, np.ndarray)):
                texts = list(model_input)
            else:
                texts = [str(model_input)]
            
            print(f"Processing {len(texts)} text(s) for prediction")
            
            # Vectorize the text
            X_vectorized = self.vectorizer.transform(texts)
            
            # Convert to PyTorch tensor
            X_tensor = torch.FloatTensor(X_vectorized.toarray())
            
            # Make predictions
            with torch.no_grad():
                outputs = self.pytorch_model(X_tensor)
                _, predicted = torch.max(outputs.data, 1)
                predictions = predicted.numpy()
            
            print(f"Predictions: {predictions}")
            return predictions
            
        except Exception as e:
            print(f"Error in predict: {e}")
            raise
    
    def predict_proba(self, context, model_input):
        """
        Get prediction probabilities
        
        Args:
            context: MLflow context (passed automatically)
            model_input: pandas DataFrame with text data
            
        Returns:
            numpy array of prediction probabilities
        """
        try:
            # Handle different input formats
            if isinstance(model_input, pd.DataFrame):
                if len(model_input.columns) == 1:
                    texts = model_input.iloc[:, 0].tolist()
                else:
                    texts = model_input.iloc[:, 0].tolist()
            elif isinstance(model_input, (list, np.ndarray)):
                texts = list(model_input)
            else:
                texts = [str(model_input)]
            
            # Vectorize the text
            X_vectorized = self.vectorizer.transform(texts)
            
            # Convert to PyTorch tensor
            X_tensor = torch.FloatTensor(X_vectorized.toarray())
            
            # Make predictions
            with torch.no_grad():
                outputs = self.pytorch_model(X_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy()
            
            return probabilities
            
        except Exception as e:
            print(f"Error in predict_proba: {e}")
            raise