# Store Sales Prediction App

A machine learning application that predicts weekly store sales using LightGBM and Streamlit.

![Demo](assets/demo.gif)

## Project Structure

```
sales-prediction/
├── model/                          # Model artifacts directory
│   ├── lgbm_sales_model.pkl       # Trained model
│   └── lgbm_sales_model_features.pkl  # Feature list
├── preprocessor.py                 # Data preprocessing module
├── predictor.py                    # Prediction module
├── train.py                        # Model training script
├── app.py                          # Streamlit application
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Place your `stores-sales.csv` in the project directory and run:

```bash
python train.py
```

This will:
- Preprocess the data with feature engineering
- Train a LightGBM model
- Save the model to `model/` directory
- Display validation RMSE and feature importance

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

## Usage

1. **Upload CSV**: Upload a CSV file with the same structure as the training data
2. **Generate Predictions**: Click the button to process and predict
3. **View Results**: 
   - See predictions in a table
   - Download predictions as CSV
   - View actual vs predicted visualization
   - Check RMSE metrics

## Model Approach

### Model Selection & Preprocessing

We use **LightGBM** for its speed and predictive power. 

#### Feature Engineering
- **Date features:** Year, month, week of year, plus cyclical encoding (`week_sin`, `week_cos`) to capture yearly seasonality.  
- **Lag features:** Past weekly sales (`lag_1, lag_2, ..., lag_52`) and past exogenous variables (`temperature`, `fuel_price`, `cpi`, `unemployment`) to model temporal dependencies.  
- **Rolling statistics:** Rolling mean and std of past 4 and 12 weeks to capture trends and momentum.  
- **Missing values:** Long-term lags (26, 52 weeks) filled using shorter lags to keep rows usable.

#### Data Leakage Prevention
- All `weekly_sales` lags and rolling features are **shifted**, so the model never sees the current week’s sales.  
- Exogenous variables and date features are either lagged or known in advance.  
- This ensures **no leakage**, enabling reliable forecasting.


### Model Selection

We used **LightGBM** for its speed and accuracy with complex temporal features. Key parameters include `n_estimators=3000`, `learning_rate=0.05`, `max_depth=8`, `num_leaves=100`, and regularization (`reg_lambda=1.0`, `reg_alpha=0.1`). On validation, the model achieved **RMSE ≈ 45,981**.



## CSV Format

Required columns:
- `store`: Store identifier
- `date`: Week of sales (DD-MM-YYYY)
- `weekly_sales`: Total sales (optional for prediction-only)
- `holiday_flag`: 1 if holiday week, 0 otherwise
- `temperature`: Temperature in °F
- `fuel_Price`: Cost of fuel
- `cpi`: Consumer Price Index
- `unemployment`: Unemployment rate

