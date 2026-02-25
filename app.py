import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Sentinel V1 — PaySim",
    page_icon="🛡️",
    layout="wide"
)

# ── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    model     = joblib.load("models/fraud_detection_v1_xgb.pkl")
    threshold = config['v1_xgboost']['deployment']['threshold']
    pr_auc    = config['v1_xgboost']['deployment']['pr_auc']
    return model, threshold, pr_auc

model, threshold, pr_auc = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ Fraud Sentinel — PaySim V1")
st.caption(
    f"XGBoost · PR-AUC `{pr_auc}` · Operating threshold `{threshold:.4f}` · "
    f"Recall `85%` · Precision `84%`"
)
st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Transaction Details")

    tx_type = st.selectbox(
        "Transaction Type",
        options=["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"],
        help="Only TRANSFER and CASH_OUT have fraud cases in PaySim."
    )

    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        format="%.2f"
    )

    col_time1, col_time2 = st.columns(2)
    with col_time1:
        hour = st.number_input("Hour (0–23)", min_value=0, max_value=23, value=12)
    with col_time2:
        minute = st.number_input("Minute (0–59)", min_value=0, max_value=59, value=0)

    # Convert hour to step — step is just the hour of the simulation cycle
    step = int(hour)

    st.subheader("Balance Information")

    col_a, col_b = st.columns(2)
    with col_a:
        old_balance_org = st.number_input(
            "Origin Balance (before tx)",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            format="%.2f"
        )
    with col_b:
        old_balance_dest = st.number_input(
            "Destination Balance (before tx)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            format="%.2f",
            help="A destination balance of $0 is a strong fraud signal."
        )

    analyze = st.button("🔍 Analyze Transaction", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
with right:
    st.subheader("Risk Assessment")

    if analyze:
        input_df = pd.DataFrame([{
            "step":           int(step),
            "type":           tx_type,
            "amount":         float(amount),
            "nameOrig":       "C000000000",
            "oldbalanceOrg":  float(old_balance_org),
            "newbalanceOrig": 0.0,
            "nameDest":       "C999999999",
            "oldbalanceDest": float(old_balance_dest),
            "newbalanceDest": 0.0,
            "isFlaggedFraud": 0,
        }])

        proba    = model.predict_proba(input_df)[0, 1]
        is_fraud = proba >= threshold

        # ── Verdict ──
        if is_fraud:
            st.error("🚨 FRAUD DETECTED", icon="🚨")
        else:
            st.success("✅ TRANSACTION LEGITIMATE", icon="✅")

        st.metric(
            label="Fraud Probability",
            value=f"{proba:.2%}",
            delta=f"{proba - threshold:+.2%} vs threshold",
            delta_color="inverse"
        )

        # ── Risk Gauge ──
        fig, ax = plt.subplots(figsize=(6, 1.2))
        fig.patch.set_alpha(0)
        ax.set_facecolor("#0e1117")

        bar_color = "#e74c3c" if is_fraud else "#2ecc71"
        ax.barh(["Risk"], [proba], color=bar_color, height=0.5)
        ax.barh(["Risk"], [1 - proba], left=[proba], color="#2a2a2a", height=0.5)
        ax.axvline(threshold, color="white", linestyle="--", linewidth=1.2, label=f"Threshold ({threshold:.3f})")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(colors="white")
        ax.legend(loc="upper right", fontsize=8, labelcolor="white", framealpha=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig)
        plt.close()

        st.divider()

        # ── Feature context ──
        st.subheader("Key Risk Signals")

        signals = {
            "Destination account was empty": old_balance_dest == 0,
            "Amount > origin balance":       amount > old_balance_org,
            "Night transaction (00–06h)":    (step % 24) in range(0, 7),
            "High amount (> $200k)":         amount > 200_000,
        }

        for signal, triggered in signals.items():
            icon = "🔴" if triggered else "🟢"
            st.markdown(f"{icon} {signal}")

        st.divider()

        # ── Raw output ──
        with st.expander("Technical Output"):
            st.json({
                "prediction":  "Fraud" if is_fraud else "Legitimate",
                "probability": round(float(proba), 6),
                "threshold":   threshold,
                "input": {
                    "type":            tx_type,
                    "amount":          amount,
                    "step":            int(step),
                    "old_balance_org":  old_balance_org,
                    "old_balance_dest": old_balance_dest,
                }
            })
    else:
        st.info("Fill in the transaction details on the left and click **Analyze Transaction**.")

        st.markdown("""
        **How it works**

        This model was trained on the PaySim synthetic financial dataset.
        It detects two fraud patterns:

        - **TRANSFER** → fraudulent agent empties origin account and transfers funds
        - **CASH_OUT** → funds are withdrawn after the transfer

        The pipeline applies feature engineering internally — you only need
        to provide raw transaction data.

        **Key signals the model uses:**
        - Origin account balance relative to transfer amount
        - Whether the destination account was empty before receiving funds
        - Transaction type and hour of day
        - Destination account historical behavior
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Fraud Sentinel V1 · PaySim Dataset · XGBoost Default · "
    "PR-AUC 0.9079 · Threshold optimized for Recall ≥ 85%"
)