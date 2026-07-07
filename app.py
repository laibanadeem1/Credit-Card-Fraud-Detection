import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Card Fraud Screening", layout="wide")

# ---------- Load model and scaler ----------
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

REQUIRED_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

# ---------- UI ----------
st.title("Credit Card Fraud Detection")
st.write(
    "This demo uses a Random Forest model trained on the "
    "[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) "
    "to screen credit card transactions for fraud."
)

tab_single, tab_batch = st.tabs(["Evaluate Transaction", "Upload a CSV File"])

# =========================================================
# TAB 1 — Single sample transaction, quick demo
# =========================================================
with tab_single:

    @st.cache_data
    def load_full_dataset():
        df = pd.read_csv("creditcard.csv")
        df["Amount_log"] = np.log1p(df["Amount"])
        return df

    try:
        full_df = load_full_dataset()
        dataset_available = True
    except FileNotFoundError:
        dataset_available = False

    if not dataset_available:
        st.warning("creditcard.csv not found in the app directory. This tab needs the full dataset to sample from.")
    else:
        st.write("Click below to pull a random real transaction from the dataset, then check the model's prediction.")

        if st.button("Evaluate Transaction"):
            row = full_df.sample(1)
            st.session_state["single_row"] = row
            st.session_state["single_row_id"] = int(row.index[0])
            st.session_state.pop("single_prediction", None)  #clear old prediction on new sample

        if "single_row" in st.session_state:
            row = st.session_state["single_row"]
            row_id = st.session_state["single_row_id"]

            st.markdown(f"### Transaction #{row_id}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Amount", f"${row['Amount'].values[0]:,.2f}")
            with col2:
                st.metric("Time", f"{int(row['Time'].values[0]):,}")
            with col3:
                st.metric("Hidden Features (V1-V28)", "Loaded")

            single_threshold = st.slider(
                "Fraud probability threshold", 0.0, 1.0, 0.2, 0.01,
                key="single_threshold",
                help="Default is 0.2, the threshold selected during this project's threshold tuning to prioritize recall."
            )

            if st.button("Predict"):
                row_processed = row.copy()
                row_processed["Hour"] = (row_processed["Time"] % 86400) // 3600
                row_processed[["Amount_scaled", "Time_scaled"]] = scaler.transform(
                    row_processed[["Amount_log", "Time"]]
                )
                feature_cols = [f"V{i}" for i in range(1, 29)] + ["Hour", "Amount_scaled", "Time_scaled"]
                X_row = row_processed[feature_cols]

                prob_fraud = model.predict_proba(X_row)[0][1]
                prediction = "Fraud" if prob_fraud >= single_threshold else "Not Fraud"
                confidence = prob_fraud if prediction == "Fraud" else (1 - prob_fraud)
                actual = "Fraud" if row["Class"].values[0] == 1 else "Not Fraud"

                st.session_state["single_prediction"] = (prediction, confidence, actual)

            if "single_prediction" in st.session_state:
                prediction, confidence, actual = st.session_state["single_prediction"]
                st.markdown("### Result")
                pred_col1, pred_col2, pred_col3 = st.columns(3)
                with pred_col1:
                    if prediction == "Fraud":
                        st.error(f"Prediction: {prediction}")
                    else:
                        st.success(f"Prediction: {prediction}")
                with pred_col2:
                    st.metric("Confidence", f"{confidence * 100:.1f}%")
                with pred_col3:
                    st.metric("Actual Label", actual)

                if prediction == actual:
                    st.success("Correct prediction.")
                else:
                    st.error("Incorrect prediction.")
        else:
            st.info("Click \"Evaluate Transaction\" to get started.")

# =========================================================
# TAB 2 — Batch CSV upload
# =========================================================
with tab_batch:
    st.write(
        "Upload a batch of transactions to screen them for fraud. "
        "The file must contain columns V1 through V28, Time, and Amount. "
        "A Class column is optional; if included, the app will also show how well the model performed on your file."
    )

    with st.expander("Don't have a file? Download a sample to try"):
        st.write(
            "This sample contains a mix of typical and fraud-like transactions, "
            "formatted to match what the model expects."
        )
        try:
            with open("sample_transactions.csv", "rb") as f:
                st.download_button(
                    "Download sample_transactions.csv",
                    data=f,
                    file_name="sample_transactions.csv",
                    mime="text/csv",
                )
        except FileNotFoundError:
            st.warning("sample_transactions.csv not found in the app directory.")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    threshold = st.slider(
        "Fraud probability threshold", 0.0, 1.0, 0.2, 0.01,
        key="batch_threshold",
        help="Default is 0.2, the threshold selected during this project's threshold tuning to prioritize recall."
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in data.columns]
        if missing_cols:
            st.error(f"The file is missing required columns: {', '.join(missing_cols)}")
            st.stop()

        has_labels = "Class" in data.columns

        # ---------- Preprocess ----------
        processed = data.copy()
        processed["Amount_log"] = np.log1p(processed["Amount"])
        processed["Hour"] = (processed["Time"] % 86400) // 3600
        processed[["Amount_scaled", "Time_scaled"]] = scaler.transform(processed[["Amount_log", "Time"]])

        feature_cols = [f"V{i}" for i in range(1, 29)] + ["Hour", "Amount_scaled", "Time_scaled"]
        X_input = processed[feature_cols]

        # ---------- Predict ----------
        probs = model.predict_proba(X_input)[:, 1]
        predictions = (probs >= threshold).astype(int)

        results = data.copy()
        results["Fraud_Probability"] = probs
        results["Prediction"] = np.where(predictions == 1, "Fraud", "Not Fraud")

        # ---------- Summary ----------
        st.subheader("Screening Summary")
        total = len(results)
        flagged = int(predictions.sum())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Transactions Screened", f"{total:,}")
        with col2:
            st.metric("Flagged as Fraud", f"{flagged:,}")
        with col3:
            st.metric("Flagged Rate", f"{(flagged / total * 100):.2f}%")

        # ---------- Optional evaluation, if Class column provided ----------
        if has_labels:
            from sklearn.metrics import classification_report, confusion_matrix

            st.subheader("Evaluation Against Provided Labels")
            report = classification_report(
                data["Class"], predictions, target_names=["Not Fraud", "Fraud"],
                output_dict=True, zero_division=0
            )
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df)

            cm = confusion_matrix(data["Class"], predictions)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: Not Fraud", "Actual: Fraud"],
                columns=["Predicted: Not Fraud", "Predicted: Fraud"],
            )
            st.write("Confusion Matrix")
            st.dataframe(cm_df)

        # ---------- Results table ----------
        st.subheader("Transaction Results")

        col_a, col_b = st.columns(2)
        with col_a:
            show_only_flagged = st.checkbox("Show only transactions flagged as Fraud")
        with col_b:
            show_hidden_features = st.checkbox("Show hidden features (V1-V28)")

        display_df = results[results["Prediction"] == "Fraud"] if show_only_flagged else results
        display_df = display_df.sort_values("Fraud_Probability", ascending=False)

        # Put the meaningful columns first; V1-V28 are anonymized and not useful to read directly
        v_cols = [c for c in display_df.columns if c.startswith("V") and c[1:].isdigit()]
        front_cols = [c for c in ["Time", "Amount", "Fraud_Probability", "Prediction"] if c in display_df.columns]
        other_cols = [c for c in display_df.columns if c not in front_cols and c not in v_cols]

        if show_hidden_features:
            ordered_cols = front_cols + other_cols + v_cols
        else:
            ordered_cols = front_cols + other_cols

        display_df = display_df[ordered_cols]
        display_df["Fraud_Probability"] = display_df["Fraud_Probability"].round(4)

        def highlight_fraud(row):
            if row["Prediction"] == "Fraud":
                return ["background-color: #7a1f2b; color: #ffffff"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display_df.style.apply(highlight_fraud, axis=1),
            height=400,
            use_container_width=True,
        )

        # ---------- Download results ----------
        csv_out = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_out,
            file_name="fraud_screening_results.csv",
            mime="text/csv",
        )

    else:
        st.info("Upload a CSV file to begin, or download the sample file above to try the app.")

# ---------- Model info ----------
with st.expander("About the model"):
    st.write(
        """
        - **Model:** Random Forest (max_depth=10, min_samples_leaf=5)
        - **Decision threshold:** 0.2 (tuned; default in scikit-learn is 0.5)
        - **Test set F2-score:** 0.81
        - **Test set Precision (Fraud):** 0.77
        - **Test set Recall (Fraud):** 0.82
        - Trained on 284,807 transactions, with only about 0.17% labeled as fraud.
        - Features V1 to V28 are anonymized PCA components; Amount and Time are log-transformed
          and scaled before being passed to the model.
        - The threshold was tuned to prioritize recall over precision, since missing a fraudulent
          transaction is generally costlier than a false alarm.
        """
    )
