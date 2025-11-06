"""
Sales prediction module.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error


class SalesPredictor:
    """Handles sales predictions using trained model."""
    
    def __init__(self, model_path: str, features_path: str):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model
            features_path: Path to saved feature list
        """
        self.model = joblib.load(model_path)
        self.features = joblib.load(features_path)
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on preprocessed data.
        
        Args:
            df: Preprocessed dataframe
        
        Returns:
            Dataframe with predictions
        """
        has_actual = "weekly_sales" in df.columns
        
        metadata = df[["date", "store_original"]].copy() if "store_original" in df.columns else df[["date", "store"]].copy()
        if has_actual:
            metadata["actual_sales"] = df["weekly_sales"]
        
        X = df[self.features]
        
        predictions = self.model.predict(X)
        result = metadata.copy()
        result["predicted_sales"] = predictions
        
        return result
    
    def calculate_rmse(self, df: pd.DataFrame) -> float:
        """
        Calculate RMSE if actual sales are available.
        
        Args:
            df: Dataframe with actual_sales and predicted_sales columns
        
        Returns:
            RMSE value
        """
        if "actual_sales" not in df.columns:
            return None
        
        return np.sqrt(mean_squared_error(df["actual_sales"], df["predicted_sales"]))