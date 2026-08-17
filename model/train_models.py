"""
train_models.py
================
BITS Pilani WILP M.Tech (AIML/DSE) - ML Assignment 2
Author: Prarthana Naik (2025AC05312)

Trains 5 classifiers on the UCI Bank Marketing dataset (id=222, bank-full.csv,
45,211 rows / 16 input features / binary target 'y' = term deposit subscription)
and saves the tuned pipelines + a metrics leaderboard + the held-out test split.

Run from the repo root:
    python3 model/train_models.py
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
# Seed derived from the student ID (2025AC05312) instead of the default 42,
# so results differ from classmates using the same dataset/approach.
SEED = 5312

RAW_URL = (
    "https://raw.githubusercontent.com/mikeizbicki/datasets/"
    "master/csv/uci/bank-full.csv.modified"
)
RAW_LOCAL_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "bank_marketing_raw.csv")

COLUMNS = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
    "previous", "poutcome", "y",
]

# 'duration' is dropped: per the UCI documentation, call duration is only known
# AFTER the call happens, and duration=0 implies y='no' by construction. Keeping
# it would leak the outcome and make the "model" trivially strong but useless as
# a pre-call predictor. This is discussed in the README observations.
DROP_COLS = ["duration"]

NUMERIC_FEATURES = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome",
]

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.dirname(__file__)


def load_data() -> pd.DataFrame:
    """Load the raw dataset from local cache, downloading it once if missing."""
    if not os.path.exists(RAW_LOCAL_CACHE):
        print(f"Local cache not found, downloading from {RAW_URL} ...")
        df_raw = pd.read_csv(RAW_URL, sep=";", names=COLUMNS, quotechar='"')
        os.makedirs(os.path.dirname(RAW_LOCAL_CACHE), exist_ok=True)
        df_raw.to_csv(RAW_LOCAL_CACHE, index=False, header=False, sep=";")
    else:
        df_raw = pd.read_csv(RAW_LOCAL_CACHE, sep=";", names=COLUMNS, quotechar='"')
    return df_raw


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def get_model_grid():
    """Returns {name: (estimator, param_grid)} for the 5 required models."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {"clf__max_depth": [8, 15, None], "clf__min_samples_leaf": [1, 5]},
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [11, 25], "clf__weights": ["uniform", "distance"]},
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 15]},
        ),
    }


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_proba), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def main():
    t0 = time.time()
    df = load_data()
    df = df.drop(columns=DROP_COLS)

    print("Shape after loading (duration dropped):", df.shape)
    print("Dtypes:\n", df.dtypes)
    print("Missing values:", df.isnull().sum().sum())
    print("Class balance:\n", df["y"].value_counts(normalize=True))

    X = df.drop(columns=["y"])
    y = (df["y"] == "yes").astype(int)

    baseline_acc = round(max(y.mean(), 1 - y.mean()), 4)
    print(f"Majority-class baseline accuracy: {baseline_acc}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Save the held-out test split verbatim (features + true label) for the
    # Streamlit CSV-upload feature and for grading reproducibility.
    test_df = X_test.copy()
    test_df["y"] = y_test.map({1: "yes", 0: "no"})
    test_df.to_csv(os.path.join(REPO_ROOT, "test_data.csv"), index=False)
    print(f"Saved test_data.csv with shape {test_df.shape}")

    results = {}
    for name, (estimator, grid) in get_model_grid().items():
        print(f"\n=== Tuning {name} ===")
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        search = GridSearchCV(pipe, grid, cv=5, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)

        best = search.best_estimator_
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_proba)
        results[name] = {
            "best_params": search.best_params_,
            "metrics": metrics,
        }
        print(f"Best params: {search.best_params_}")
        print(f"Test metrics: {metrics}")

        fname = name.lower().replace(" ", "_").replace("-", "_") + ".joblib"
        joblib.dump(best, os.path.join(MODEL_DIR, fname), compress=3)
        print(f"Saved {fname}")

    results["_baseline_majority_class_accuracy"] = baseline_acc
    results["_seed"] = SEED
    results["_test_size"] = len(y_test)
    results["_train_size"] = len(y_train)

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone in {time.time() - t0:.1f}s. Wrote model/metrics.json")

    # Leaderboard sorted by AUC
    print("\n=== LEADERBOARD (sorted by AUC) ===")
    lb = sorted(
        ((n, r["metrics"]) for n, r in results.items() if not n.startswith("_")),
        key=lambda kv: kv[1]["AUC"],
        reverse=True,
    )
    for name, m in lb:
        print(f"{name:22s} AUC={m['AUC']:.4f}  MCC={m['MCC']:.4f}  Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}")


if __name__ == "__main__":
    sys.exit(main())
