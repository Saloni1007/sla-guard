
from pathlib import Path
import json

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------------------------------------------------
# APPLICATION SETUP
# --------------------------------------------------

app = FastAPI(title="SLA Guard Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# LOAD TRAINED MODEL AND THRESHOLD
# --------------------------------------------------

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "sla_random_forest_model.pkl"
THRESHOLD_PATH = BASE_DIR / "sla_threshold.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}. "
        "Run train_model.py first."
    )

if not THRESHOLD_PATH.exists():
    raise FileNotFoundError(
        f"Threshold file not found: {THRESHOLD_PATH}. "
        "Run train_model.py first."
    )

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
    threshold_config = json.load(file)

THRESHOLD = float(threshold_config["threshold"])

if not 0 < THRESHOLD < 1:
    raise ValueError(
        "The saved threshold must be between 0 and 1."
    )

# --------------------------------------------------
# FEATURES - MUST MATCH TRAINING ORDER
# --------------------------------------------------

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

# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------

class WorkloadInput(BaseModel):
    cpu_requested: float = Field(ge=0)
    memory_requested: float = Field(ge=0)
    cpu_utilization: float = Field(ge=0)
    memory_utilization: float = Field(ge=0)
    machine_cpu_utilization: float = Field(ge=0)
    machine_memory_utilization: float = Field(ge=0)
    queue_length: int = Field(ge=0)
    priority: int = Field(ge=0)
    scheduling_class: int = Field(ge=0)
    network_latency_ms: float = Field(ge=0)
    estimated_runtime_sec: float = Field(ge=0)
    resource_contention: float = Field(ge=0)

# --------------------------------------------------
# HEALTH AND HOME ENDPOINTS
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "SLA Guard API is running",
        "model_loaded": True,
        "model": "Random Forest",
        "threshold": THRESHOLD,
        "dataset_type": "synthetic",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "threshold_loaded": True,
    }


# --------------------------------------------------
# SLA VIOLATION PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict(data: WorkloadInput):
    try:
        # Keep feature order identical to model training
        values = [[
            getattr(data, feature)
            for feature in FEATURES
        ]]

        # Obtain class probabilities
        probabilities = model.predict_proba(
            np.asarray(values, dtype=float)
        )[0]

        # Locate probability for SLA_Violation = 1
        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError(
                "The trained model does not contain "
                "the SLA violation class (1)."
            )

        violation_index = classes.index(1)
        violation_probability = float(
            probabilities[violation_index]
        )

        # Apply threshold saved by train_model.py
        prediction = int(
            violation_probability >= THRESHOLD
        )

        return {
            "prediction": prediction,
            "result": (
                "SLA Violation Predicted"
                if prediction == 1
                else "No SLA Violation Predicted"
            ),
            "violation_probability": round(
                violation_probability * 100, 2
            ),
            "threshold": THRESHOLD,
            "threshold_percent": round(
                THRESHOLD * 100, 2
            ),
            "model": "Random Forest",
            "dataset_type": "synthetic",
            "message": (
                "Prediction generated successfully. "
                "Results are based on a model trained "
                "on synthetic data."
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}",
        ) from error
