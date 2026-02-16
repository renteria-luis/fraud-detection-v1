import streamlit as st
import pandas as pd
import joblib
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from src.features.custom_transformers import FeatureEngineering  # Your FE logic

# --- Page Configuration ---
st.set_page_config(
    page_title="Fraud Sentinel V1",
    page_icon="🛡️",
    layout="wide"
)

# --- Singleton to Load the Model ---
@st.cache_resource
def load_artifacts():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load the pipeline that already includes preprocessing and XGBoost
    model = joblib.load("models/fraud_detection_v1_xgb.pkl")
    threshold = config['v1_xgboost']['deployment']['threshold']
    return model, threshold

model, threshold = load_artifacts()

# --- User Interface ---
st.title("🛡️ Fraud Sentinel: Real-Time Detection")
st.markdown(f"**Model Version:** `1.0.0` | **Operating Threshold:** `{threshold}`")

st.sidebar.header("Transaction Details")
st.sidebar.info("Input the features derived from PCA (V1-V28) and transaction metadata.")

# --- Input Form ---
with st.sidebar:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
    time = st.number_input("Time (Seconds since first transaction)", min_value=0.0, value=0.0)
    
    # Create sliders or inputs for the most important variables based on your EDA
    v14 = st.slider("V14 (Highest Correlation)", -20.0, 20.0, 0.0)
    v12 = st.slider("V12", -20.0, 20.0, 0.0)
    v10 = st.slider("V10", -20.0, 20.0, 0.0)
    
    # For the rest of the Vs, use a neutral value (0.0) to avoid overwhelming the user
    other_vs = {f"V{i}": 0.0 for i in range(1, 29) if i not in [10, 12, 14]}

# --- Build the DataFrame for Prediction ---
input_data = {
    "Time": time,
    "Amount": amount,
    "V10": v10, "V12": v12, "V14": v14,
    **other_vs
}
input_df = pd.DataFrame([input_data])

# Reorder columns to match training
cols_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
input_df = input_df[cols_order]

# --- Prediction ---
if st.button("Analyze Transaction"):
    # The pipeline already includes FeatureEngineering (Time -> IsNight, etc)
    proba = model.predict_proba(input_df)[0, 1]
    is_fraud = proba >= threshold

    # --- Visual Results ---
    col1, col2 = st.columns(2)
    
    with col1:
        if is_fraud:
            st.error(f"🚨 FRAUD DETECTED")
        else:
            st.success(f"✅ TRANSACTION LEGITIMATE")
        
        st.metric("Fraud Probability", f"{proba:.4%}")
    
    with col2:
        # Mini "risk thermometer" chart
        fig, ax = plt.subplots(figsize=(6, 1))
        color = 'red' if is_fraud else 'green'
        ax.barh(["Risk"], [proba], color=color)
        ax.axvline(threshold, color='black', linestyle='--', label='Threshold')
        ax.set_xlim(0, 1)
        ax.legend()
        st.pyplot(fig)

    st.divider()
    st.subheader("Technical Raw Output")
    st.json({
        "prediction": "Fraud" if is_fraud else "Legitimate",
        "probability": float(proba),
        "threshold": threshold,
        "input_summary": {"Amount": amount, "V14": v14}
    })