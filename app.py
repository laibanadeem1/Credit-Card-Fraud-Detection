import random
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Ranges the preset buttons randomize within, rather than a single fixed number
SMALL_AMOUNT_RANGE = (2.0, 30.0)      # a coffee to a dinner
LARGE_AMOUNT_RANGE = (300.0, 2000.0)  # a big-ticket purchase
SMALL_HOUR_OF_DAY_RANGE = (7.0, 22.0)  # normal waking hours
LARGE_HOUR_OF_DAY_RANGE = (0.0, 5.0)   # small hours of the night

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
            <div class="risk-meter-marker" style="left: {thresh_pct}%;" title="Cautious level"></div>
            <div class="risk-meter-fill-marker" style="left: {pct}%;">|</div>
        </div>
    </div>
    """

def verdict_box(prediction):
    is_fraud = prediction == "Fraud"
    css_class = "verdict-fraud" if is_fraud else "verdict-safe"
    label = "🚩 FRAUD DETECTED" if is_fraud else "✅ TRANSACTION CLEAR"
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


def fraud_label(value):
    return f"🚩 {value}" if value == "Fraud" else f"✅ {value}"


def section(title, desc):
    st.markdown(f"""
    <div class="section-title">{title}</div>
    <div class="section-desc">{desc}</div>
    """, unsafe_allow_html=True)


V_TOOLTIP = ("V1-V28 are the transaction's underlying signals (things like merchant type or "
             "spending pattern) that banks scramble using PCA before releasing the data, so no "
             "personal or business info is exposed.")


def format_elapsed(seconds):
    """Turn raw dataset seconds into a plain-language 'Day X, HH:MM' label."""
    hours_total = seconds / 3600
    day = int(hours_total // 24) + 1
    hour_of_day = hours_total % 24
    h = int(hour_of_day)
    m = int(round((hour_of_day - h) * 60))
    if m == 60:
        m = 0
        h += 1
    return f"Day {day}, {h:02d}:{m:02d}"


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
    <p>Checks a transaction's details and tells you how likely it is to be fraud &middot; trained on nearly 285,000 real transactions</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    """<p style="font-size: 15px; color: var(--muted); text-align: center; margin: -8px 0 28px 0;">
    Test a single transaction, or upload a batch — this tool tells you how suspicious it looks.
    </p>""",
    unsafe_allow_html=True,
)

# =========================================================
# SECTION 1 — Single sample transaction
# =========================================================
section("Try a Transaction", "Pull a real transaction from the dataset, then adjust its amount and timing to see how the model reacts.")

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
    time_min = float(full_df["Time"].min())
    time_max = float(full_df["Time"].max())
    amount_max = float(np.ceil(full_df["Amount"].max()))

    single_threshold = THRESHOLD_DEFAULT

    if st.button("Draw Transaction", use_container_width=True):
        row = full_df.sample(1)
        st.session_state["single_row"] = row
        st.session_state["single_row_id"] = int(row.index[0])
        # seed the sliders with this transaction's real values
        st.session_state["time_val"] = float(row["Time"].values[0])
        st.session_state["amount_val"] = float(row["Amount"].values[0])
        st.session_state.pop("single_prediction", None)

    if "single_row" in st.session_state:
        row = st.session_state["single_row"]
        row_id = st.session_state["single_row_id"]

        st.write("")
        st.markdown(
            f"""<p style="font-size: 14px; color: var(--muted); margin: 0 0 8px 0;">
            Transaction #{row_id} — the two fields below are the only ones a human can read.
            Everything else is <span title="{V_TOOLTIP}" style="border-bottom: 1px dotted var(--muted); cursor: help;">V1-V28, anonymized signals</span>
            the model uses but that can't be edited here.
            </p>""",
            unsafe_allow_html=True,
        )

        st.caption("Or try a quick scenario (numbers shuffle a bit each click):")
        max_days = max(int(time_max // 86400), 0)  # how many extra days the dataset spans

        def random_time_hours(hour_range):
            day = random.randint(0, max_days)
            hour_of_day = random.uniform(*hour_range)
            return min(day * 24 + hour_of_day, time_max / 3600)

        p1, p2 = st.columns(2)
        with p1:
            if st.button("Small everyday purchase", use_container_width=True):
                st.session_state[f"amount_slider_{row_id}"] = min(random.uniform(*SMALL_AMOUNT_RANGE), amount_max)
                st.session_state[f"time_slider_{row_id}"] = random_time_hours(SMALL_HOUR_OF_DAY_RANGE)
                st.session_state.pop("single_prediction", None)
        with p2:
            if st.button("Large late-night purchase", use_container_width=True):
                st.session_state[f"amount_slider_{row_id}"] = min(random.uniform(*LARGE_AMOUNT_RANGE), amount_max)
                st.session_state[f"time_slider_{row_id}"] = random_time_hours(LARGE_HOUR_OF_DAY_RANGE)
                st.session_state.pop("single_prediction", None)

        c1, c2 = st.columns(2)
        with c1:
            amount_val = st.slider(
                "Transaction Amount ($)",
                min_value=0.0, max_value=max(amount_max, 1.0),
                value=float(st.session_state.get("amount_val", row["Amount"].values[0])),
                step=1.0, format="$%.2f",
                key=f"amount_slider_{row_id}",
            )
        with c2:
            time_val = st.slider(
                "When did it happen? (hours since tracking began)",
                min_value=0.0, max_value=max(time_max / 3600, 1.0),
                value=float(st.session_state.get("time_val", row["Time"].values[0])) / 3600,
                step=0.5,
                key=f"time_slider_{row_id}",
            )
            st.caption(f"≈ {format_elapsed(time_val * 3600)}")

        st.write("")
        if st.button("Run Prediction", use_container_width=True):
            X_row = row[REQUIRED_COLUMNS].copy()
            X_row["Amount"] = amount_val
            X_row["Time"] = time_val * 3600
            prob_fraud = model.predict_proba(X_row)[0][1]
            prediction = "Fraud" if prob_fraud >= single_threshold else "Not Fraud"
            actual = "Fraud" if row["Class"].values[0] == 1 else "Not Fraud"
            st.session_state["single_prediction"] = (prediction, actual, prob_fraud)

        if "single_prediction" in st.session_state:
            prediction, actual, prob_fraud = st.session_state["single_prediction"]
            verdict_box(prediction)

            confidence = prob_fraud if prediction == "Fraud" else (1 - prob_fraud)
            verdict_word = "fraud" if prediction == "Fraud" else "safe"
            st.markdown(
                f"""<p style="font-size: 15px; color: var(--text); margin: 4px 0 12px 0;">
                The model thinks this transaction is <strong>{verdict_word}</strong> with
                <strong>{confidence * 100:.0f}% confidence</strong>.
                </p>""",
                unsafe_allow_html=True,
            )
            st.markdown(risk_meter_html(prob_fraud, single_threshold), unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                metric_card("Prediction", fraud_label(prediction))
            with c2:
                metric_card("What Really Happened", fraud_label(actual))


# =========================================================
# SECTION 2 — Batch CSV upload
# =========================================================
section("Batch Screening", "")
st.markdown(
    f"""<div class="section-desc" style="margin-top: -12px;">
    Upload a CSV with columns Time,
    <span title="{V_TOOLTIP}" style="border-bottom: 1px dotted var(--muted); cursor: help;">V1-V28</span>,
    and Amount. Include an optional Class column (1 = Fraud, 0 = Not Fraud) to see how the model did on your file.
    </div>""",
    unsafe_allow_html=True,
)

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

threshold = THRESHOLD_DEFAULT

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

    label_col = "Actual" if "Actual" in data.columns else ("Class" if "Class" in data.columns else None)
    has_labels = label_col is not None

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

    st.markdown(
        f"""<p style="font-size: 16px; color: var(--text); margin: 0 0 14px 0;">
        <strong>{flagged} of {total}</strong> transactions looked suspicious
        (<strong>{(flagged / total * 100):.1f}%</strong>).
        </p>""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Screened", f"{total:,}")
    with c2:
        metric_card("Flagged", f"{flagged:,}")
    with c3:
        metric_card("Flag Rate", f"{(flagged / total * 100):.2f}%")

    st.write("")
    st.caption("Transaction results")
    st.markdown(
        f"""<p style="font-size: 13px; color: var(--muted); margin: 0 0 12px 0;">
        <strong>Risk Score</strong> is how confident the model is that a transaction is fraud — the higher the
        number, the more suspicious it looks. Anything at or above {THRESHOLD_DEFAULT * 100:.0f}% gets flagged as Fraud.
        </p>""",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        show_only_flagged = st.checkbox("Show only flagged as Fraud")
    with col_b:
        show_hidden_features = st.checkbox("Show hidden features (V1-V28)", help=V_TOOLTIP)

    display_df = results[results["Prediction"] == "Fraud"] if show_only_flagged else results
    display_df = display_df.sort_values("Fraud_Probability", ascending=False).copy()

    if has_labels and label_col == "Class":
        display_df["Actual"] = np.where(display_df["Class"] == 1, "Fraud", "Not Fraud")
        display_df = display_df.drop(columns=["Class"])

    if "Time" in display_df.columns:
        display_df["Time"] = display_df["Time"].apply(format_elapsed)

    v_cols = [c for c in display_df.columns if c.startswith("V") and c[1:].isdigit()]
    front_cols = [c for c in ["Time", "Amount", "Fraud_Probability", "Prediction", "Actual"] if c in display_df.columns]
    other_cols = [c for c in display_df.columns if c not in front_cols and c not in v_cols]
    ordered_cols = front_cols + other_cols + (v_cols if show_hidden_features else [])

    display_df = display_df[ordered_cols]
    display_df["Fraud_Probability"] = (display_df["Fraud_Probability"] * 100).round(1)
    display_df = display_df.rename(columns={"Fraud_Probability": "Risk Score (%)"})

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

with st.expander("Show technical details"):
    st.markdown(f"""
| | |
|---|---|
| **Model** | Random Forest (`max_depth=10`, `min_samples_leaf=5`) in a single Pipeline with RobustScaler |
| **How cautious it is by default** | Flags a transaction once it's `{THRESHOLD_DEFAULT * 100:.0f}%` confident (scikit-learn's own default is `50%`) |
| **Catches actual fraud** | About **88 out of 100** real frauds (misses roughly 12) |
| **Accuracy when it flags something** | About **69 out of 100** flagged transactions are real fraud (roughly 31 are false alarms) |
| **Overall reliability score** | `0.83` out of `1.0`, weighted to reward catching fraud over avoiding false alarms |
| **Training data** | 284,807 transactions, ~0.17% fraud |
| **Features** | `Time`, `Amount`, `V1`-`V28` (anonymized PCA components) |

The threshold was tuned to catch more real fraud, even if that means a few extra false alarms —
missing a fraudulent transaction is treated as costlier than flagging one that turns out to be fine.
""")

    st.markdown(
        f"""<p style="font-size: 13px; color: var(--muted); margin-top: 8px;">
        What's <span title="{V_TOOLTIP}" style="border-bottom: 1px dotted var(--muted); cursor: help;">V1-V28</span>?
        </p>""",
        unsafe_allow_html=True,
    )