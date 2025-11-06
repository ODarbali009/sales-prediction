"""
Data preprocessing for sales prediction.
Handles feature engineering for both training and prediction.
"""
import pandas as pd
import numpy as np


class SalesDataPreprocessor:
    """Preprocesses sales data with feature engineering."""
    
    def __init__(self):
        self.lags_sales = [1, 2, 3, 4, 8, 12, 26, 52]
        self.lags_exog = [1, 2, 4]
        self.exog_cols = ["temperature", "fuel_Price", "cpi", "unemployment"]
    
    def preprocess(self, df: pd.DataFrame, fill_nan: bool = False) -> pd.DataFrame:

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors='coerce')
        df = df.sort_values(["store", "date"]).reset_index(drop=True)
        
        df["store_original"] = df["store"]
        
        df = self._add_date_features(df)
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)

        if fill_nan:
            df = self._fill_missing_values(df)
        else:
            df = df.dropna().reset_index(drop=True)
        
        
        return df
    
    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add date-based features."""
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
        df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for sales and exogenous variables."""
        for lag in self.lags_sales:
            df[f"weekly_sales_lag_{lag}"] = df.groupby("store")["weekly_sales"].shift(lag)

        df["weekly_sales_lag_26"] = df["weekly_sales_lag_26"].fillna(
            df["weekly_sales_lag_12"]
        )
        df["weekly_sales_lag_52"] = df["weekly_sales_lag_52"].fillna(
            df["weekly_sales_lag_26"]
        )
        for lag in self.lags_exog:
            for col in self.exog_cols:
                df[f"{col}_lag_{lag}"] = df.groupby("store")[col].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics."""
        df['rolling_mean_4'] = df.groupby('store')['weekly_sales'].transform(lambda x: x.rolling(4).mean().shift(1))
        df['rolling_std_4'] = df.groupby('store')['weekly_sales'].transform(lambda x: x.rolling(4).std().shift(1))
        df['rolling_mean_12'] = df.groupby('store')['weekly_sales'].transform(lambda x: x.rolling(12).mean().shift(1))
        
        return df
    
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with store-grouped means."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ["store", "store_original"]:
                df[col] = df.groupby("store")[col].transform(
                    lambda x: x.fillna(x.mean())
                )
        df = df.dropna().reset_index(drop=True)

        return df