"""
Streamlit app for sales prediction.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from preprocessor import SalesDataPreprocessor
from predictor import SalesPredictor


st.set_page_config(page_title="Store Sales Predictor", layout="wide")

st.title("Weekly Store Sales Predictor")
st.markdown("Upload a CSV file to predict weekly sales per store.")

# st.sidebar.header("Options")
# fill_nan = st.sidebar.checkbox(
#     "Fill missing values with store means", 
#     value=False,
#     help="If checked, missing values will be filled with store-grouped means. Otherwise, rows with missing values will be dropped."
# )
fill_nan = False # For simplicity, we set fill_nan to False in this app

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"File uploaded successfully!")
        
        with st.expander("View uploaded data (first 10 rows)"):
            st.dataframe(df.head(10))
        
        if st.button("Generate Predictions", type="primary"):
            with st.spinner("Processing data and generating predictions..."):
                preprocessor = SalesDataPreprocessor()
                df_processed = preprocessor.preprocess(df, fill_nan=fill_nan)
                
                predictor = SalesPredictor(
                    model_path="model/lgbm_sales_model.pkl",
                    features_path="model/lgbm_sales_model_features.pkl"
                )
                results = predictor.predict(df_processed)
                
                st.session_state['results'] = results
                st.session_state['has_actual'] = 'actual_sales' in results.columns
                
                if st.session_state['has_actual']:
                    rmse = predictor.calculate_rmse(results)
                    st.session_state['rmse'] = rmse
        
        if 'results' in st.session_state:
            results = st.session_state['results']
            has_actual = st.session_state['has_actual']
            
            st.success(f"Predictions generated!")
            
            if has_actual and 'rmse' in st.session_state:
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("RMSE", f"{st.session_state['rmse']:,.2f}")
            
            st.subheader("Predictions")
            display_cols = ["date", "store_original", "predicted_sales"]
            if has_actual:
                display_cols.append("actual_sales")
            
            st.dataframe(
                results[display_cols].style.format({
                    "predicted_sales": "{:,.2f}",
                    "actual_sales": "{:,.2f}" if has_actual else None
                }),
                height=400
            )
            
            download_df = results[display_cols].copy()
            download_df["predicted_sales"] = download_df["predicted_sales"].round(2)
            if has_actual:
                download_df["actual_sales"] = download_df["actual_sales"].round(2)
            
            csv = download_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
            
            if has_actual:
                st.subheader("📈 Actual vs Predicted Sales")
                
                store_col = "store_original" if "store_original" in results.columns else "store"
                stores = sorted(results[store_col].unique())
                selected_store = st.selectbox("Select Store", stores)
                
                store_data = results[results[store_col] == selected_store].copy()
                store_data = store_data.sort_values("date")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=store_data["date"],
                    y=store_data["actual_sales"],
                    mode='lines+markers',
                    name='Actual Sales',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=6)
                ))
                
                fig.add_trace(go.Scatter(
                    x=store_data["date"],
                    y=store_data["predicted_sales"],
                    mode='lines+markers',
                    name='Predicted Sales',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title=f"Store {selected_store} - Actual vs Predicted Sales",
                    xaxis_title="Date",
                    yaxis_title="Weekly Sales ($)",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Store-specific RMSE
                store_rmse = ((store_data["actual_sales"] - store_data["predicted_sales"]) ** 2).mean() ** 0.5
                st.info(f"RMSE for Store {selected_store}: **${store_rmse:,.2f}**")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please ensure your CSV has the required columns: store, date, weekly_sales, holiday_flag, temperature, fuel_Price, cpi, unemployment")

else:
    st.info("Please upload a CSV file to get started")
    
    # expected format
    with st.expander("Expected CSV Format"):
        st.markdown("""
        Your CSV should contain the following columns:
        - **store**: Store identifier
        - **date**: Week of sales (format: DD-MM-YYYY)
        - **weekly_sales**: Total sales for the given store (optional for prediction-only)
        - **holiday_flag**: 1 if the week includes a holiday, 0 otherwise
        - **temperature**: Temperature in °F
        - **fuel_Price**: Cost of fuel in the region
        - **cpi**: Consumer Price Index
        - **unemployment**: Unemployment rate
        """)