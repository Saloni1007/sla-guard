# SLA Guard
### Machine Learning-Based SLA Violation Prediction System for Cloud Computing

## 1. Project Overview

SLA Guard is a machine learning-based web application designed to predict potential Service Level Agreement (SLA) violations in cloud computing workloads.

It analyzes workload-related parameters such as CPU utilization, memory utilization, queue length, and network latency to estimate the probability of an SLA violation.

The application uses a Random Forest classification model integrated with a FastAPI backend and a React frontend.

### Key Features
- Machine learning-based SLA violation prediction.
- Analysis of 12 workload-related features.
- Probability-based prediction.
- Classification threshold optimization.
- Interactive React frontend.
- FastAPI prediction backend.
- Saved model integration using Joblib.

## 2. Problem Statement

Cloud computing environments execute multiple workloads that share resources such as CPU, memory, and network bandwidth.

High resource utilization, increased queue length, network latency, and resource contention may affect workload performance.

SLA Guard aims to predict whether a workload is likely to be associated with an SLA violation using machine learning.

The problem is formulated as a binary classification task:

- `0` – No SLA violation.
- `1` – SLA violation.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Frontend | React.js, Vite |
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn |
| Algorithm | Random Forest |
| Data Processing | Pandas, NumPy |
| Model Storage | Joblib |
| Version Control | Git, GitHub, Git LFS |

## 4. Dataset Description

The project uses a synthetic cloud workload dataset containing 50,000 records and 13 columns.

It consists of 12 input features and one target variable.

**Dataset:** `sla_guard_50000_unique_logs.csv`

### Input Features

1. `cpu_requested`
2. `memory_requested`
3. `cpu_utilization`
4. `memory_utilization`
5. `machine_cpu_utilization`
6. `machine_memory_utilization`
7. `queue_length`
8. `priority`
9. `scheduling_class`
10. `network_latency_ms`
11. `estimated_runtime_sec`
12. `resource_contention`

**Target Variable:** `SLA_Violation`

- 0 – No violation
- 1 – Violation

The dataset is synthetic and is used for model development and experimentation. It does not represent actual production cloud workload observations.

## 5. Data Preparation

The dataset is divided into three subsets:

| Dataset | Percentage | Records |
|---|---:|---:|
| Training | 60% | 30,000 |
| Validation | 20% | 10,000 |
| Testing | 20% | 10,000 |

- Training data is used to train the classifier.
- Validation data is used for threshold selection.
- Testing data is reserved for final evaluation.

## 6. Machine Learning Model

SLA Guard uses a **Random Forest Classifier**, an ensemble learning algorithm that combines multiple decision trees to generate predictions.

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
```

### Handling Class Imbalance

The model uses `class_weight="balanced_subsample"` to account for differences in target class frequencies during training.

Techniques such as SMOTE and SMOTE-ENN have not been implemented and may be explored in future work.

## 7. Classification Threshold Optimization

The model generates a probability estimate for the SLA violation class.

Instead of using the default threshold of 0.50, SLA Guard uses a selected threshold of approximately **0.30**.

The threshold was selected using validation data with a target recall of approximately 85%.

The classification rule is:

```text
Probability >= 0.30  → SLA Violation
Probability < 0.30   → No SLA Violation
```

A lower threshold prioritizes identifying more positive cases, although it may increase false positives.

## 8. Model Validation Results

The reported validation results at the selected threshold are:

| Metric | Result |
|---|---:|
| Precision | 57.72% |
| Recall | 85.31% |
| Threshold | Approximately 0.30 |

**Precision:** Measures how many predicted violation cases were actually positive.

**Recall:** Measures how many actual violation cases were correctly identified.

These are validation results, not final test-set results.

## 9. System Architecture

SLA Guard follows a frontend-backend-machine-learning architecture.

```text
       USER
         |
         v
   REACT FRONTEND
         |
         v
   FASTAPI BACKEND
         |
         v
  RANDOM FOREST MODEL
         |
         v
 PROBABILITY ESTIMATION
         |
         v
 THRESHOLD APPLICATION
         |
         v
  BINARY PREDICTION
         |
         v
   RESULT DISPLAY
```

### System Workflow

1. User enters workload parameters.
2. React sends the input to FastAPI.
3. Backend prepares the input features.
4. Random Forest generates a probability estimate.
5. The selected threshold is applied.
6. Backend returns the prediction.
7. React displays the result.

## 10. Backend Implementation

The backend is developed using Python and FastAPI.

Its responsibilities include:

- Receiving prediction requests.
- Processing workload features.
- Loading the trained model.
- Reading the classification threshold.
- Generating probability estimates.
- Returning prediction results.

**Prediction Endpoint:** `POST /predict`

**Local API Documentation:** `http://127.0.0.1:8001/docs`

## 11. Frontend Implementation

The frontend uses React.js and Vite.

It provides an interface for:

- Entering workload parameters.
- Submitting prediction requests.
- Communicating with the backend.
- Displaying prediction results and probabilities.

## 12. Model Storage

The trained model and classification threshold are stored separately.

| File | Purpose |
|---|---|
| `sla_random_forest_model.pkl` | Trained Random Forest model |
| `sla_threshold.json` | Selected classification threshold |

Joblib is used for model serialization.

Git LFS is used to track the large `.pkl` model file in GitHub.

## 13. Project Structure

```text
sla-guard/
│
├── backend/
│   ├── main.py
│   ├── train_model.py
│   ├── sla_random_forest_model.pkl
│   └── sla_threshold.json
│
├── public/
├── src/
├── .gitattributes
├── .gitignore
├── package.json
├── package-lock.json
├── index.html
├── vite.config.js
└── README.md
```

## 14. Installation and Execution

### Clone the Repository

```bash
git clone https://github.com/Saloni1007/sla-guard.git
cd sla-guard
```

### Retrieve the Model Using Git LFS

```bash
git lfs install
git lfs pull
```

### Start the Backend

```powershell
cd backend
py -m uvicorn main:app --reload --port 8001
```

### Start the Frontend

Open another terminal in the project root:

```bash
npm install
npm run dev
```

Open the local URL displayed by Vite in your browser.

The backend must remain active while using the frontend.

## 15. API Testing

The backend can be tested using FastAPI Swagger documentation:

`http://127.0.0.1:8001/docs`

Testing procedure:

1. Open the Swagger documentation.
2. Select the `/predict` endpoint.
3. Select Try it out.
4. Enter the required input features.
5. Execute the request.
6. Inspect the prediction response.

## 16. Example Prediction

An example local API prediction produced:

```text
Violation Probability: 72.83%
Classification Threshold: 30%
Prediction: SLA Violation
```

Since the estimated probability exceeds the selected threshold, the model predicts an SLA violation.

This is an individual prediction example, not a model performance metric.

## 17. Limitations

- The model uses synthetic data rather than real production cloud workloads.
- Real-world generalization has not been established.
- Reported metrics are validation results.
- Lower thresholds may increase false-positive predictions.
- The application does not automatically prevent SLA violations.
- It does not directly modify cloud resource allocation.

## 18. Future Scope

Possible future improvements include:

- Testing with real-world cloud workload datasets.
- Comparing Random Forest with other classification algorithms.
- Experimenting with SMOTE and SMOTE-ENN.
- Improving prediction performance.
- Adding prediction history and dashboard analytics.
- Integrating with cloud monitoring systems.
- Investigating automated resource management.

## 19. Conclusion

SLA Guard demonstrates the application of machine learning to potential SLA violation prediction in cloud computing workloads.

By combining a Random Forest classifier, FastAPI backend, and React frontend, the project provides an integrated system for workload analysis and probability-based predictions.

The current implementation serves as a prototype that can be extended through real-world dataset testing, alternative algorithms, and cloud integration.

## 20. GitHub Repository

**Project Name:** SLA Guard

**GitHub Repository:**  
https://github.com/Saloni1007/sla-guard

The repository contains the project source code, model training implementation, saved model, backend, frontend, and relevant configuration files.
