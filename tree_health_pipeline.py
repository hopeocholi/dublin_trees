"""
Dublin Urban Tree Health Classification
=======================================

End-to-end, reproducible pipeline that predicts the health condition of
public trees managed by Dublin City Council from structural and
environmental attributes.

Two models are trained on an identical preprocessing pipeline and compared
on accuracy and F1-macro (the headline metric for imbalanced multi-class
data): a Logistic Regression baseline and a Random Forest classifier.

Usage
-----
    python src/tree_health_pipeline.py --data data/total-tree-population.csv

Data
----
Download `total-tree-population.csv` from:
    https://data.smartdublin.ie/dataset/tree-dcc
and place it in the data/ directory.
"""

import argparse
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

FEATURE_COLS = [
    "Age",
    "Proximity",
    "Area",
    "Species",
    "StemDiameter_cm_mid",
    "Height_m_mid",
    "Spread_m_mid",
]
NUM_COLS = ["StemDiameter_cm_mid", "Height_m_mid", "Spread_m_mid"]
CAT_COLS = [c for c in FEATURE_COLS if c not in NUM_COLS]


def parse_range_mid(value):
    """Convert a text range like '5 to 10' into its numeric midpoint."""
    if pd.isna(value):
        return np.nan
    nums = re.findall(r"(\d+(?:\.\d+)?)", str(value))
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def load_and_engineer(path):
    """Load the raw CSV and build model-ready features."""
    df = pd.read_csv(path)

    # Text size ranges -> numeric midpoints
    df["StemDiameter_cm_mid"] = df["StemDiameter"].apply(parse_range_mid)
    df["Height_m_mid"] = df["Height"].apply(parse_range_mid)
    df["Spread_m_mid"] = df["Spread"].apply(parse_range_mid)

    # Target: fill missing, then collapse rare labels into "Other"
    df["Condition"] = df["Condition"].fillna("Unknown")
    counts = df["Condition"].value_counts()
    rare = counts[counts < 50].index
    df.loc[df["Condition"].isin(rare), "Condition"] = "Other"

    return df


def build_preprocessor():
    """Shared preprocessing so both models are compared fairly."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUM_COLS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CAT_COLS,
            ),
        ]
    )


def evaluate(name, model, X_test, y_test):
    """Print accuracy, F1-macro and a full per-class report."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average="macro")
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-macro : {f1m:.4f}")
    print(classification_report(y_test, preds))
    return {"model": name, "accuracy": acc, "f1_macro": f1m}


def main(data_path):
    df = load_and_engineer(data_path)

    X = df[FEATURE_COLS].copy()
    y = df["Condition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    preprocess = build_preprocessor()

    log_reg = Pipeline(
        [("prep", preprocess), ("clf", LogisticRegression(max_iter=1000))]
    )
    rf = Pipeline(
        [
            ("prep", preprocess),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    log_reg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    results = [
        evaluate("Logistic Regression (baseline)", log_reg, X_test, y_test),
        evaluate("Random Forest (final)", rf, X_test, y_test),
    ]

    print(f"\n{'=' * 60}\nSummary\n{'=' * 60}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/total-tree-population.csv",
        help="Path to total-tree-population.csv",
    )
    args = parser.parse_args()
    main(args.data)
