"""
Model training script.
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
from preprocessor import SalesDataPreprocessor


def train_model(data_path: str, output_dir: str = "model"):
    """
    Train sales prediction model.
    
    Args:
        data_path: Path to training data CSV
        output_dir: Directory to save model artifacts
    """
    df = pd.read_csv(data_path)
    
    preprocessor = SalesDataPreprocessor()
    df = preprocessor.preprocess(df, fill_nan=False)
    
    split_date = "2012-09-01"
    train = df[df["date"] < split_date]
    valid = df[df["date"] >= split_date]
    
    target = "weekly_sales"
    features = [col for col in df.columns if col not in ["date", target, "store_original"]]
    
    X_train, y_train = train[features], train[target]
    X_valid, y_valid = valid[features], valid[target]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_valid)}")
    
    model = LGBMRegressor(
        n_estimators=3000,          
        learning_rate=0.05,         
        max_depth=8,                
        num_leaves=100,             
        min_child_samples=20,       
        subsample=0.8,              
        colsample_bytree=0.8,       
        reg_lambda=1.0,             
        reg_alpha=0.1,              
        random_state=42,
        verbose=-1,                 
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[early_stopping(100, verbose=100)]
    )
    
    # Evaluate
    y_pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    print(f"\nValidation RMSE: {rmse:.2f}")
    
    # Save model and features
    joblib.dump(model, f"{output_dir}/lgbm_sales_model.pkl")
    joblib.dump(features, f"{output_dir}/lgbm_sales_model_features.pkl")
    
    print(f"\nModel saved to {output_dir}/")
    
    # Feature importance
    plot_feature_importance(model, features, output_dir)
    
    return model, features


def plot_feature_importance(model, features, output_dir: str):
    """Plot and save feature importance."""
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    print("\nTop 30 Feature Importances:")
    print(importance_df.head(30))
    
    plt.figure(figsize=(10, 8))
    plt.barh(
        importance_df['feature'][:30][::-1],
        importance_df['importance'][:30][::-1]
    )
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Top 30 Feature Importances (LightGBM)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    plt.show()


if __name__ == "__main__":
    train_model("stores-sales.csv", output_dir="model")