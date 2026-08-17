"""
DepositIQ — Term Deposit Subscription Predictor
BITS Pilani WILP M.Tech (AIML/DSE) - ML Assignment 2
Author: Prarthana Naik (2025AC05312)
"""

import io
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.train_models import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SEED,
    build_preprocessor,
    get_model_grid,
)

# ----------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------
st.set_page_config(page_title="DepositIQ", page_icon="\U0001F3E6", layout="wide")

TEAL = "#0B3D3C"
GOLD = "#C9971A"
CREAM = "#FAF7F0"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {CREAM}; color: #1A1A1A; }}
    h1, h2, h3 {{ color: {TEAL}; }}
    section[data-testid="stSidebar"] {{ background-color: {TEAL}; }}
    section[data-testid="stSidebar"] * {{ color: {CREAM} !important; }}
    div.stButton > button, .stDownloadButton > button {{
        background-color: {GOLD}; color: {TEAL}; border: none; font-weight: 600;
    }}
    .stTabs [data-baseweb="tab"] {{ color: {TEAL}; font-weight: 600; }}
    .metric-card {{
        background-color: white; border-left: 6px solid {GOLD};
        padding: 0.8rem 1rem; border-radius: 6px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

EXPECTED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def show_df(df, **kwargs):
    """Render a dataframe, tolerating both old and new Streamlit width APIs."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


@st.cache_resource
def load_metrics():
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        return json.load(f)


@st.cache_resource(show_spinner=True)
def load_model(name: str):
    """Load a saved pipeline. If the pickle can't be read (e.g. a scikit-learn
    version mismatch between training and serving environments), refit the
    pipeline from the saved hyperparameters + the same random seed so the
    metrics stay identical."""
    path = os.path.join(MODEL_DIR, MODEL_FILES[name])
    try:
        return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"Saved model artifact could not be unpickled ({exc}). "
            "Refitting from saved hyperparameters — this can take a minute."
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline

        from model.train_models import DROP_COLS, load_data

        metrics = load_metrics()
        best_params = metrics[name]["best_params"]
        estimator, _ = get_model_grid()[name]
        clf_params = {k.split("__", 1)[1]: v for k, v in best_params.items()}
        estimator.set_params(**clf_params)

        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        df = load_data().drop(columns=DROP_COLS)
        X = df.drop(columns=["y"])
        y = (df["y"] == "yes").astype(int)
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        pipe.fit(X_train, y_train)
        return pipe


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("\U0001F3E6 DepositIQ")
st.sidebar.caption("Term deposit subscription predictor")
st.sidebar.markdown("---")

model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("Test data")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
use_bundled = st.sidebar.checkbox(
    "Use bundled sample test_data.csv instead", value=uploaded_file is None
)

st.sidebar.markdown("---")
st.sidebar.caption("BITS Pilani WILP · M.Tech AIML/DSE · ML Assignment 2")
st.sidebar.caption("Prarthana Naik · 2025AC05312")

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("DepositIQ — Term Deposit Subscription Predictor")
st.write(
    "Predicts whether a bank client will subscribe to a term deposit, "
    "using 5 classifiers trained on the UCI **Bank Marketing** dataset "
    "(45,211 clients, 15 features after dropping the leaky `duration` column)."
)

tab_leaderboard, tab_predict, tab_about = st.tabs(
    ["\U0001F4CA Leaderboard", "\U0001F9EA Predict & Evaluate", "\u2139\uFE0F Dataset & Notes"]
)

metrics_json = load_metrics()
model_names = list(MODEL_FILES.keys())

# ----------------------------------------------------------------------
# Tab 1 — Leaderboard
# ----------------------------------------------------------------------
with tab_leaderboard:
    st.subheader("All 6 metrics, per model")
    rows = []
    for n in model_names:
        m = metrics_json[n]["metrics"]
        rows.append({"Model": n, **m})
    lb_df = pd.DataFrame(rows).set_index("Model")
    lb_df = lb_df.sort_values("AUC", ascending=False)

    try:
        styled = lb_df.style.background_gradient(cmap="YlGn", axis=0).format("{:.4f}")
        st.dataframe(styled, width="stretch")
    except Exception:
        show_df(lb_df.round(4))

    winner = lb_df["AUC"].idxmax()
    baseline = metrics_json.get("_baseline_majority_class_accuracy")
    st.markdown(
        f"""<div class="metric-card">
        <b>Overall winner (highest AUC):</b> {winner}<br>
        <b>Majority-class baseline accuracy:</b> {baseline}
        </div>""",
        unsafe_allow_html=True,
    )
    st.caption(
        "The dataset is imbalanced (~88% 'no' / 12% 'yes'), so accuracy alone is "
        "misleading — AUC and MCC are the more trustworthy metrics here."
    )

# ----------------------------------------------------------------------
# Tab 2 — Predict & Evaluate
# ----------------------------------------------------------------------
with tab_predict:
    st.subheader(f"Selected model: {model_name}")

    df_input = None
    if uploaded_file is not None and not use_bundled:
        df_input = pd.read_csv(uploaded_file)
        st.success(f"Loaded uploaded file with shape {df_input.shape}")
    elif use_bundled and os.path.exists(TEST_DATA_PATH):
        df_input = pd.read_csv(TEST_DATA_PATH)
        st.info(f"Using bundled test_data.csv with shape {df_input.shape}")
    else:
        st.warning("Upload a CSV in the sidebar, or tick 'Use bundled sample test_data.csv'.")

    if df_input is not None:
        missing = [c for c in EXPECTED_FEATURES if c not in df_input.columns]
        if missing:
            st.error(f"Uploaded CSV is missing required columns: {missing}")
        else:
            has_labels = "y" in df_input.columns
            X_new = df_input[EXPECTED_FEATURES]

            model = load_model(model_name)
            y_pred = model.predict(X_new)
            y_proba = model.predict_proba(X_new)[:, 1]

            result_df = df_input.copy()
            result_df["prediction"] = np.where(y_pred == 1, "yes", "no")
            result_df["subscribe_probability"] = y_proba.round(4)

            st.write("**Predictions (first 20 rows):**")
            show_df(result_df.head(20))

            csv_bytes = result_df.to_csv(index=False).encode()
            st.download_button(
                "\U0001F4E5 Download predictions as CSV",
                data=csv_bytes,
                file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
            )

            if has_labels:
                y_true = (df_input["y"] == "yes").astype(int)
                m = compute_metrics(y_true, y_pred, y_proba)

                st.write("**Evaluation metrics on this file:**")
                cols = st.columns(6)
                for c, (k, v) in zip(cols, m.items()):
                    c.metric(k, f"{v:.4f}")

                st.write("**Confusion matrix:**")
                cm = confusion_matrix(y_true, y_pred)
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual: no", "Actual: yes"],
                    columns=["Predicted: no", "Predicted: yes"],
                )
                show_df(cm_df)

                st.write("**Classification report:**")
                report = classification_report(
                    y_true, y_pred, target_names=["no", "yes"], output_dict=True, zero_division=0
                )
                show_df(pd.DataFrame(report).transpose().round(4))
            else:
                st.info(
                    "No 'y' column found in the uploaded file, so only predictions are "
                    "shown (no metrics/confusion matrix — those need true labels)."
                )

# ----------------------------------------------------------------------
# Tab 3 — About
# ----------------------------------------------------------------------
with tab_about:
    st.subheader("Problem statement")
    st.write(
        "Predict whether a client of a Portuguese bank will subscribe to a term "
        "deposit (`y`), based on direct-marketing phone-call campaign data, so the "
        "bank can prioritise which clients to call."
    )

    st.subheader("Dataset")
    st.write(
        "**UCI Bank Marketing** (id 222, `bank-full.csv`) — 45,211 clients, "
        "16 raw input features + binary target. Source: "
        "https://archive.ics.uci.edu/dataset/222/bank+marketing"
    )
    st.write(
        "`duration` (last call duration) was **dropped** before training: the UCI "
        "documentation notes it is only known *after* a call ends and is highly "
        "correlated with the outcome by construction (duration=0 → y='no'), so "
        "keeping it would leak the label and produce an unrealistically strong but "
        "practically useless pre-call predictor."
    )
    st.write(f"Features used: `{', '.join(EXPECTED_FEATURES)}` (15 features).")

    st.subheader("Models")
    st.write(", ".join(model_names))

    st.subheader("Reproducibility")
    st.write(
        f"Stratified 80/20 train/test split, random seed **{SEED}** "
        "(derived from the student ID). Each model tuned with 5-fold "
        "`GridSearchCV` scoring on ROC-AUC. Preprocessing (scaling + one-hot "
        "encoding) is inside a single `sklearn.Pipeline` per model, fit only "
        "on the training fold, so nothing leaks from test to train."
    )
