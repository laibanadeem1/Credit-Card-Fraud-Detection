import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# ---------- Load model, scaler, and data ----------
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    df["Amount_log"] = np.log1p(df["Amount"])
    return df

model, scaler = load_model_and_scaler()
df = load_data()

feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount_scaled", "Time_scaled"]

# Apply the same scaling used during training
df[["Amount_scaled", "Time_scaled"]] = scaler.transform(df[["Amount_log", "Time"]])

# ---------- UI ----------
st.title("Credit Card Fraud Detection")
st.write(
    "This demo uses a Random Forest model trained on the "
    "[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). "
    "Since the dataset's features are anonymized (PCA-transformed), this demo lets you sample a real "
    "transaction from the dataset and see the model's prediction, rather than entering values by hand."
)

st.subheader("Sample a Transaction")

col1, col2 = st.columns(2)
with col1:
    sample_type = st.selectbox("Transaction type to sample", ["Random", "Known Fraud", "Known Not Fraud"])
with col2:
    threshold = st.slider("Fraud probability threshold", 0.0, 1.0, 0.2, 0.01)

if st.button("Sample a new transaction"):
    if sample_type == "Known Fraud":
        row = df[df["Class"] == 1].sample(1)
    elif sample_type == "Known Not Fraud":
        row = df[df["Class"] == 0].sample(1)
    else:
        row = df.sample(1)
    st.session_state["row"] = row

if "row" not in st.session_state:
    st.session_state["row"] = df.sample(1)

row = st.session_state["row"]
X_row = row[feature_cols]

prob_fraud = model.predict_proba(X_row)[0][1]
prediction = "Fraud" if prob_fraud >= threshold else "Not Fraud"
actual = "Fraud" if row["Class"].values[0] == 1 else "Not Fraud"

st.subheader("Transaction Details")
detail_col1, detail_col2 = st.columns(2)
with detail_col1:
    st.metric("Amount", f"${row['Amount'].values[0]:.2f}")
with detail_col2:
    hour = (row["Time"].values[0] % 86400) // 3600
    st.metric("Hour of Day", f"{int(hour)}:00")

st.subheader("Model Prediction")
pred_col1, pred_col2, pred_col3 = st.columns(3)
with pred_col1:
    st.metric("Fraud Probability", f"{prob_fraud:.4f}")
with pred_col2:
    st.metric("Prediction", prediction)
with pred_col3:
    st.metric("Actual Label", actual)

if prediction == actual:
    st.success(f"Correct prediction at threshold {threshold:.2f}.")
else:
    st.error(f"Incorrect prediction at threshold {threshold:.2f}.")

st.caption(
    "Threshold default is set to 0.2, the value selected during threshold tuning to prioritize "
    "recall (catching more fraud) over precision, since missing fraud carries a higher cost than "
    "a false alarm. Adjust the slider to see how the prediction changes at different thresholds."
)

# ---------- Model info ----------
with st.expander("About the model"):
    st.write(
        """
        - **Model:** Random Forest (max_depth=10, min_samples_leaf=5)
        - **Test set F1-score:** 0.84
        - **Test set Precision (Fraud):** 0.95
        - **Test set Recall (Fraud):** 0.76
        - Trained on 284,807 transactions, with only about 0.17% labeled as fraud.
        - Features V1 to V28 are anonymized PCA components; Amount and Time were log-transformed
          and scaled before training.
        """
    )