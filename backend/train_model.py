
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# ==========================================================
# SLA GUARD - RANDOM FOREST TRAINING
# Dataset: 50,000 rows, 13 columns
# Split: 60% training, 20% validation, 20% testing
# ==========================================================

print("=" * 60)
print("SLA GUARD - RANDOM FOREST TRAINING")
print("=" * 60)

# ----------------------------------------------------------
# 1. FILE PATHS
# ----------------------------------------------------------

DATASET_PATH = Path(
    r"C:\Users\ABHISHEK\Downloads\sla_guard_50000_unique_logs.csv"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "sla_random_forest_model.pkl"
THRESHOLD_PATH = BASE_DIR / "sla_threshold.json"

TARGET = "SLA_Violation"

FEATURES = [
    "cpu_requested",
    "memory_requested",
    "cpu_utilization",
    "memory_utilization",
    "machine_cpu_utilization",
    "machine_memory_utilization",
    "queue_length",
    "priority",
    "scheduling_class",
    "network_latency_ms",
    "estimated_runtime_sec",
    "resource_contention",
]

EXPECTED_ROWS = 50000
TARGET_RECALL = 0.85
RANDOM_STATE = 42

# ----------------------------------------------------------
# 2. LOAD AND VALIDATE DATASET
# ----------------------------------------------------------

print("\nLoading dataset...")

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found at:\n{DATASET_PATH}\n"
        "Change DATASET_PATH to the actual location of your CSV."
    )

df = pd.read_csv(DATASET_PATH)

print(f"Dataset path: {DATASET_PATH}")
print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

if len(df) != EXPECTED_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_ROWS} rows, but loaded {len(df)} rows. "
        "Check that DATASET_PATH points to your 50,000-row CSV."
    )

if len(df.columns) != 13:
    raise ValueError(
        f"Expected 13 columns, but found {len(df.columns)}."
    )

required_columns = FEATURES + [TARGET]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values found in required columns. "
        "Clean the dataset before training."
    )

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError(
        "SLA_Violation must contain only 0 and 1."
    )

df[TARGET] = df[TARGET].astype(int)

print("\nClass distribution:")
print(df[TARGET].value_counts().sort_index())

print("\nClass percentages:")
print(
    (df[TARGET].value_counts(normalize=True).sort_index() * 100)
    .round(2)
)

X = df[FEATURES]
y = df[TARGET]

# ----------------------------------------------------------
# 3. SPLIT DATA: 60% TRAIN, 20% VALIDATION, 20% TEST
# ----------------------------------------------------------

print("\nSplitting dataset...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

print(f"Training records:   {len(X_train)}")
print(f"Validation records: {len(X_val)}")
print(f"Test records:       {len(X_test)}")
print(f"Total records:      {len(X_train) + len(X_val) + len(X_test)}")

# ----------------------------------------------------------
# 4. TRAIN RANDOM FOREST
# ----------------------------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

start_time = time.perf_counter()
model.fit(X_train, y_train)
training_time = time.perf_counter() - start_time

print(f"Training completed in {training_time:.2f} seconds.")

# ----------------------------------------------------------
# 5. SELECT THRESHOLD USING VALIDATION DATA ONLY
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("VALIDATION THRESHOLD SELECTION")
print("=" * 60)

val_probabilities = model.predict_proba(X_val)[:, 1]

threshold_candidates = np.arange(0.05, 0.951, 0.01)

valid_candidates = []

for threshold in threshold_candidates:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val, val_predictions, zero_division=0
    )
    recall = recall_score(
        y_val, val_predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, val_predictions, zero_division=0
    )
    f2 = fbeta_score(
        y_val, val_predictions, beta=2, zero_division=0
    )

    if recall >= TARGET_RECALL:
        valid_candidates.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "f2": float(f2),
            }
        )

if valid_candidates:
    # Highest precision among thresholds satisfying target recall.
    # F2 and threshold provide deterministic tie-breaks.
    selected = max(
        valid_candidates,
        key=lambda item: (
            item["precision"],
            item["f2"],
            item["threshold"],
        ),
    )
    selection_method = (
        f"Highest precision among thresholds achieving "
        f"validation recall >= {TARGET_RECALL:.0%}"
    )
else:
    # If the recall target cannot be achieved, select the
    # threshold with the highest validation F2 score.
    fallback_candidates = []

    for threshold in threshold_candidates:
        predictions = (
            val_probabilities >= threshold
        ).astype(int)

        fallback_candidates.append(
            {
                "threshold": float(threshold),
                "precision": float(
                    precision_score(
                        y_val, predictions, zero_division=0
                    )
                ),
                "recall": float(
                    recall_score(
                        y_val, predictions, zero_division=0
                    )
                ),
                "f1": float(
                    f1_score(
                        y_val, predictions, zero_division=0
                    )
                ),
                "f2": float(
                    fbeta_score(
                        y_val, predictions, beta=2,
                        zero_division=0
                    )
                ),
            }
        )

    selected = max(
        fallback_candidates,
        key=lambda item: (
            item["f2"],
            item["precision"],
        ),
    )
    selection_method = (
        "Recall target not reached; selected highest validation F2"
    )

THRESHOLD = selected["threshold"]

print(f"Selection method: {selection_method}")
print(f"Selected threshold: {THRESHOLD:.2f}")
print(f"Validation precision: {selected['precision']:.4f}")
print(f"Validation recall: {selected['recall']:.4f}")
print(f"Validation F1: {selected['f1']:.4f}")
print(f"Validation F2: {selected['f2']:.4f}")

# ----------------------------------------------------------
# 6. FINAL EVALUATION ON UNTOUCHED TEST DATA
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

start_time = time.perf_counter()

test_probabilities = model.predict_proba(X_test)[:, 1]

inference_time = time.perf_counter() - start_time

test_predictions = (
    test_probabilities >= THRESHOLD
).astype(int)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(
    y_test, test_predictions, zero_division=0
)
recall = recall_score(
    y_test, test_predictions, zero_division=0
)
f1 = f1_score(
    y_test, test_predictions, zero_division=0
)
f2 = fbeta_score(
    y_test, test_predictions, beta=2, zero_division=0
)
roc_auc = roc_auc_score(y_test, test_probabilities)
pr_auc = average_precision_score(y_test, test_probabilities)

cm = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print(f"Threshold: {THRESHOLD:.2f}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-score:  {f1:.4f}")
print(f"F2-score:  {f2:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion matrix:")
print(cm)
print(f"True negatives:  {tn}")
print(f"False alarms:    {fp}")
print(f"Missed violations: {fn}")
print(f"Detected violations: {tp}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=[
            "No SLA Violation",
            "SLA Violation",
        ],
        digits=4,
        zero_division=0,
    )
)

print(f"Test inference time: {inference_time:.4f} seconds")
print(
    f"Inference per row: "
    f"{(inference_time / len(X_test)) * 1000:.4f} ms"
)

# ----------------------------------------------------------
# 7. SAVE MODEL AND THRESHOLD FOR FASTAPI
# ----------------------------------------------------------

print("\nSaving model and threshold...")

joblib.dump(model, MODEL_PATH)

threshold_data = {
    "threshold": float(THRESHOLD),
    "target_recall": TARGET_RECALL,
    "dataset_type": "synthetic",
    "training_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows": int(len(X_test)),
    "feature_names": FEATURES,
    "selection_method": selection_method,
}

with open(THRESHOLD_PATH, "w", encoding="utf-8") as file:
    json.dump(threshold_data, file, indent=4)

print(f"Model saved to: {MODEL_PATH}")
print(f"Threshold saved to: {THRESHOLD_PATH}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("Dataset type: Synthetic")
print("Dataset rows:", len(df))
print("Model: Random Forest")
print(f"Final decision threshold: {THRESHOLD:.2f}")
print("Model and threshold files are ready for FastAPI.")
print(
    "Important: these metrics evaluate held-out synthetic data, "
    "not real cloud workloads."
)
