import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

THRESHOLD_DEFAULT = 0.12  # selected during validation-based threshold tuning

# =========================================================
# THEME
# =========================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root {
    --bg: #0B0E14;
    --card: #141922;
    --border: #232B38;
    --text: #E8ECF1;
    --muted: #7C8798;
    --alert: #FF3B5C;
    --alert-dim: rgba(255, 59, 92, 0.12);
    --safe: #00D4A0;
    --safe-dim: rgba(0, 212, 160, 0.12);
    --amber: #FFB020;
}

html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; }
h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}

.stApp { background: var(--bg); }
header[data-testid="stHeader"] { background: var(--bg); }
.block-container { padding-top: 2rem; max-width: 1100px; }

/* Header banner */
.fraud-subbar {
    display: table;
    margin: 0 auto 24px auto;
    padding: 14px 20px;
    background: linear-gradient(135deg, #141922 0%, #0B0E14 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    text-align: center;
}
.fraud-subbar p {
    margin: 0;
    font-size: 14px;
    color: var(--muted);
}   
.status-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    color: var(--safe);
    background: var(--safe-dim);
    border: 1px solid rgba(0, 212, 160, 0.35);
    padding: 6px 14px;
    border-radius: 20px;
    text-transform: uppercase;
    margin-top: 16px;
}

/* Metric cards */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: left;
}
.metric-label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
}

/* Verdict banner */
.verdict-box {
    border-radius: 14px;
    padding: 24px 28px;
    margin: 18px 0;
    border: 1px solid;
}
.verdict-fraud {
    background: var(--alert-dim);
    border-color: rgba(255, 59, 92, 0.4);
}
.verdict-safe {
    background: var(--safe-dim);
    border-color: rgba(0, 212, 160, 0.4);
}
.verdict-label {
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}
.verdict-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px;
    font-weight: 700;
}
.verdict-fraud .verdict-title { color: var(--alert); }
.verdict-safe .verdict-title { color: var(--safe); }

/* Risk meter */
.risk-meter-wrap { margin: 20px 0 8px 0; }
.risk-meter-label {
    display: flex; justify-content: space-between;
    font-size: 11px; color: var(--muted); margin-bottom: 8px;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.risk-meter-track {
    position: relative;
    height: 10px;
    border-radius: 6px;
    background: linear-gradient(90deg, #00D4A0 0%, #FFB020 55%, #FF3B5C 100%);
}
.risk-meter-marker {
    position: absolute;
    top: -6px;
    width: 3px;
    height: 22px;
    background: var(--text);
    border-radius: 2px;
}
.risk-meter-fill-marker {
    position: absolute;
    top: -14px;
    width: 12px;
    text-align: center;
    color: var(--text);
    margin-left: -6px;
    font-size: 12px;
}

/* Section header */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--text);
    margin: 36px 0 4px 0;
    padding-top: 20px;
    border-top: 1px solid var(--border);
}
.section-desc {
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 16px;
}

/* Buttons */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    border-radius: 8px;
    border: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)


def risk_meter_html(prob, threshold):
    pct = max(0, min(100, prob * 100))
    thresh_pct = max(0, min(100, threshold * 100))
    return f"""
    <div class="risk-meter-wrap">
        <div class="risk-meter-label"><span>LOW RISK</span><span>HIGH RISK</span></div>
        <div class="risk-meter-track">
            <div class="risk-meter-marker" style="left: {thresh_pct}%;" title="Decision threshold"></div>
            <div class="risk-meter-fill-marker" style="left: {pct}%;">|</div>
        </div>
    </div>
    """

def verdict_box(prediction):
    is_fraud = prediction == "Fraud"
    css_class = "verdict-fraud" if is_fraud else "verdict-safe"
    label = "FRAUD DETECTED" if is_fraud else "TRANSACTION CLEAR"
    st.markdown(f"""
    <div class="verdict-box {css_class}">
        <div class="verdict-label">Model Verdict</div>
        <div class="verdict-title">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def section(title, desc):
    st.markdown(f"""
    <div class="section-title">{title}</div>
    <div class="section-desc">{desc}</div>
    """, unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()
REQUIRED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# =========================================================
# HEADER
# =========================================================
st.header('Credit Card Fraud Detection', anchor=None, help=None, divider=False, width="stretch", text_alignment="center")

st.markdown("""
<div class="fraud-subbar">
    <p>Random Forest &middot; validation-tuned threshold &middot; trained on 284,807 transactions</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SECTION 1 — Single sample transaction
# =========================================================
section("Single Transaction", "Draw a random transaction from the dataset and run it through the model live.")

@st.cache_data
def load_full_dataset():
    return pd.read_csv("creditcard.csv")

try:
    full_df = load_full_dataset()
    dataset_available = True
except FileNotFoundError:
    dataset_available = False

if not dataset_available:
    st.warning("creditcard.csv not found in the app directory. This section needs the full dataset to sample from.")
else:
    col_btn, col_slider = st.columns([1, 2])
    with col_btn:
        if st.button("Draw Transaction", use_container_width=True):
            row = full_df.sample(1)
            st.session_state["single_row"] = row
            st.session_state["single_row_id"] = int(row.index[0])
            st.session_state.pop("single_prediction", None)
    with col_slider:
        single_threshold = st.slider(
            "Decision threshold", 0.0, 1.0, THRESHOLD_DEFAULT, 0.01,
            key="single_threshold",
            help="Default is 0.12 — the threshold selected during validation-based tuning."
        )

    if "single_row" in st.session_state:
        row = st.session_state["single_row"]
        row_id = st.session_state["single_row_id"]

        st.write("")
        st.caption(f"Transaction #{row_id}")
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Amount", f"${row['Amount'].values[0]:,.2f}")
        with c2:
            metric_card("Time (sec)", f"{int(row['Time'].values[0]):,}")
        with c3:
            metric_card("Hidden Features", "V1-V28 loaded")

        st.write("")
        if st.button("Run Prediction", use_container_width=True):
            X_row = row[REQUIRED_COLUMNS]
            prob_fraud = model.predict_proba(X_row)[0][1]
            prediction = "Fraud" if prob_fraud >= single_threshold else "Not Fraud"
            actual = "Fraud" if row["Class"].values[0] == 1 else "Not Fraud"
            st.session_state["single_prediction"] = (prediction, actual)
             
        if "single_prediction" in st.session_state:
            prediction, actual = st.session_state["single_prediction"]
            verdict_box(prediction)

            c1, c2 = st.columns(2)
            with c1:
                metric_card("Prediction", prediction)
            with c2:
                metric_card("Actual", actual)


# =========================================================
# SECTION 2 — Batch CSV upload
# =========================================================
section("Batch Screening", "Upload a CSV with columns Time, V1-V28, and Amount. Include an optional Class column to see model accuracy on your file.")

with st.expander("No file? Download a sample to try"):
    st.write("A synthetic mix of typical and fraud-like transactions, formatted to match what the model expects.")
    try:
        with open("sample_transactions.csv", "rb") as f:
            st.download_button(
                "Download sample_transactions.csv",
                data=f, file_name="sample_transactions.csv", mime="text/csv",
            )
    except FileNotFoundError:
        st.warning("sample_transactions.csv not found in the app directory.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")

threshold = st.slider(
    "Decision threshold", 0.0, 1.0, THRESHOLD_DEFAULT, 0.01,
    key="batch_threshold",
    help="Default is 0.12 — the threshold selected during validation-based tuning."
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

    X_input = data[REQUIRED_COLUMNS]
    probs = model.predict_proba(X_input)[:, 1]
    predictions = (probs >= threshold).astype(int)

    results = data.copy()
    results["Fraud_Probability"] = probs
    results["Prediction"] = np.where(predictions == 1, "Fraud", "Not Fraud")

    st.write("")
    st.caption("Screening summary")
    total = len(results)
    flagged = int(predictions.sum())

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Screened", f"{total:,}")
    with c2:
        metric_card("Flagged", f"{flagged:,}")
    with c3:
        metric_card("Flag Rate", f"{(flagged / total * 100):.2f}%")

    if has_labels:
        from sklearn.metrics import classification_report, confusion_matrix

        st.write("")
        st.caption("Evaluation against provided labels")
        report = classification_report(
            data["Class"], predictions, target_names=["Not Fraud", "Fraud"],
            output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

        cm = confusion_matrix(data["Class"], predictions)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Not Fraud", "Actual: Fraud"],
            columns=["Predicted: Not Fraud", "Predicted: Fraud"],
        )
        st.write("Confusion Matrix")
        st.dataframe(cm_df, use_container_width=True)

    st.write("")
    st.caption("Transaction results")

    col_a, col_b = st.columns(2)
    with col_a:
        show_only_flagged = st.checkbox("Show only flagged as Fraud")
    with col_b:
        show_hidden_features = st.checkbox("Show hidden features (V1-V28)")

    display_df = results[results["Prediction"] == "Fraud"] if show_only_flagged else results
    display_df = display_df.sort_values("Fraud_Probability", ascending=False)

    v_cols = [c for c in display_df.columns if c.startswith("V") and c[1:].isdigit()]
    front_cols = [c for c in ["Time", "Amount", "Fraud_Probability", "Prediction"] if c in display_df.columns]
    other_cols = [c for c in display_df.columns if c not in front_cols and c not in v_cols]
    ordered_cols = front_cols + other_cols + (v_cols if show_hidden_features else [])

    display_df = display_df[ordered_cols]
    display_df["Fraud_Probability"] = display_df["Fraud_Probability"].round(4)

    def highlight_fraud(row):
        if row["Prediction"] == "Fraud":
            return ["background-color: #2A1520; color: #FF6B85; font-weight: 600"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_df.style.apply(highlight_fraud, axis=1),
        height=400, use_container_width=True,
    )

    csv_out = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV",
        data=csv_out, file_name="fraud_screening_results.csv", mime="text/csv",
    )

# =========================================================
# MODEL INFO
# =========================================================
section("About This Model", "")

st.markdown(f"""
| | |
|---|---|
| **Model** | Random Forest (`max_depth=10`, `min_samples_leaf=5`) in a single Pipeline with RobustScaler |
| **Decision threshold** | `{THRESHOLD_DEFAULT}` (validation-tuned; scikit-learn default is `0.5`) |
| **Test F2-score** | `0.83` |
| **Test Precision (Fraud)** | `0.69` |
| **Test Recall (Fraud)** | `0.88` |
| **Training data** | 284,807 transactions, ~0.17% fraud |
| **Features** | `Time`, `Amount`, `V1`-`V28` (anonymized PCA components) |

The threshold was tuned to favor recall over precision — missing a fraudulent transaction
is treated as costlier than raising a false alarm.
""")