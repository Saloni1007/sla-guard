
# SLA GUARD
## Machine Learning-Based SLA Violation Prediction in Cloud Computing

---

## 1. Introduction

Service Level Agreements (SLAs) are formal agreements between cloud service providers and customers that define expected service quality, performance, availability, and reliability.

In cloud computing environments, multiple workloads share limited computing resources such as CPU, memory, storage, and network bandwidth. Due to high resource utilization, workload congestion, network latency, and resource contention, cloud applications may fail to meet their agreed service requirements.

SLA Guard is a machine learning-based web application developed to predict potential SLA violations in cloud computing workloads. It analyzes workload and resource-related parameters and uses a trained Random Forest classification model to estimate whether a workload is likely to violate its SLA.

The main objective is to provide an early indication of potential SLA violations, helping users identify workloads that may require attention.

## 2. Problem Statement

Cloud computing environments handle multiple workloads simultaneously. Resource shortages, high CPU utilization, memory pressure, increasing queue lengths, and network delays can negatively affect application performance.

Traditional monitoring systems generally identify problems after performance degradation has occurred. This can make it difficult to respond before an SLA violation happens.

Therefore, there is a need for a predictive system that can analyze workload characteristics and identify potential SLA violations in advance.

SLA Guard addresses this problem using machine learning-based classification.

## 3. Project Objectives

The major objectives of SLA Guard are:

- To develop a machine learning-based SLA violation prediction system.
- To analyze cloud workload and resource utilization parameters.
- To identify patterns associated with potential SLA violations.
- To train a Random Forest classification model.
- To integrate the trained model with a FastAPI backend.
- To develop an interactive web interface using React.
- To display prediction results and violation probabilities.
- To provide an accessible interface for testing different workload conditions.

## 4. Technologies and Tools Used

| Component | Technology |
|---|---|
| Frontend | React.js |
| Build Tool | Vite |
| Styling | CSS |
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn |
| Algorithm | Random Forest Classifier |
| Model Serialization | Joblib / Pickle |
| Data Processing | Pandas, NumPy |
| API Testing | FastAPI Swagger UI |
| Version Control | Git, GitHub |

## 5. System Architecture

SLA Guard follows a frontend-backend machine learning architecture.

### Working Flow

1. The user enters cloud workload parameters through the React interface.
2. The frontend sends the input data to the FastAPI backend through an HTTP POST request.
3. The backend validates the received input.
4. The trained Random Forest model processes the input features.
5. The model generates a prediction and estimates the probability of an SLA violation.
6. The probability is compared against the selected classification threshold.
7. The backend returns the prediction result to the frontend.
8. The React interface displays the result to the user.

### Main Components

**Frontend:** Collects user inputs and displays prediction results.

**Backend:** Handles API requests, validates input data, loads the trained model, and generates predictions.

**Machine Learning Model:** Analyzes workload features and classifies potential SLA violations.

**Model File:** The trained model is stored in a serialized `.pkl` file and loaded by the backend.

## 6. Dataset Description

The project uses a synthetic cloud workload dataset containing 50,000 records.

The dataset consists of 12 input features and one target variable.

### Input Features

| No. | Feature | Description |
|---|---|---|
| 1 | cpu_requested | CPU resources requested by the workload |
| 2 | memory_requested | Memory resources requested |
| 3 | cpu_utilization | CPU utilization of the workload |
| 4 | memory_utilization | Memory utilization of the workload |
| 5 | machine_cpu_utilization | CPU utilization of the machine |
| 6 | machine_memory_utilization | Memory utilization of the machine |
| 7 | queue_length | Number of workloads waiting in the queue |
| 8 | priority | Workload priority |
| 9 | scheduling_class | Scheduling category of the workload |
| 10 | network_latency_ms | Network latency in milliseconds |
| 11 | estimated_runtime_sec | Estimated workload execution time |
| 12 | resource_contention | Indicator of competition for shared resources |

### Target Variable

**SLA_Violation**

- 0 – No SLA violation
- 1 – SLA violation

The dataset is synthetic and is used for model development and experimentation. Its results should not be interpreted as proof of performance on real cloud infrastructure.

## 7. Machine Learning Methodology

### Random Forest Classifier

SLA Guard uses the Random Forest algorithm, an ensemble learning technique that combines multiple decision trees to make a classification decision.

Each decision tree produces a prediction, and the forest combines the predictions to generate the final result.

Random Forest was selected because it can model nonlinear relationships, handle multiple input features, and reduce the overfitting risk associated with relying on a single decision tree.

### Model Configuration

The model uses the following configuration:

- Number of estimators: 300
- Maximum features: sqrt
- Minimum samples per leaf: 2
- Class weighting: balanced_subsample
- Random state: 42

### Dataset Splitting

The dataset was divided into three subsets:

| Dataset | Percentage | Records |
|---|---:|---:|
| Training | 60% | 30,000 |
| Validation | 20% | 10,000 |
| Testing | 20% | 10,000 |
| Total | 100% | 50,000 |

The training set is used to train the model, the validation set is used for threshold selection, and the test set is reserved for final evaluation.

### Classification Threshold

The model generates a probability score for the positive class.

Instead of using the default classification threshold of 0.50, the project uses a threshold of approximately 0.30.

The threshold was selected using validation data with a target recall of 0.85.

This approach prioritizes detecting a larger proportion of potential SLA violations, while accepting that some predictions may be false alarms.

## 8. Model Evaluation

Model evaluation considers classification metrics such as:

- **Accuracy:** Measures the proportion of correctly classified records.
- **Precision:** Measures how many predicted violations are actually violations.
- **Recall:** Measures how many actual violations are correctly identified.
- **F1-score:** Represents the harmonic mean of precision and recall.

The validation threshold selection produced the following results at approximately 0.30:

| Metric | Validation Result |
|---|---:|
| Precision | 57.72% |
| Recall | 85.31% |

These are validation results, not final test-set results.

The relatively high recall indicates that the selected threshold identifies a large proportion of actual violation cases in the validation data. The precision also indicates that false alarms remain an area for improvement.

## 9. Backend Implementation

The backend is developed using FastAPI.

The backend is responsible for:

- Loading the trained Random Forest model.
- Loading the classification threshold configuration.
- Receiving workload inputs through the prediction endpoint.
- Validating incoming data.
- Generating predictions.
- Returning prediction probabilities and classification results.

The trained model is stored in:

`backend/sla_random_forest_model.pkl`

The threshold configuration is stored in:

`backend/sla_threshold.json`

The main backend implementation is:

`backend/main.py`

The model training implementation is:

`backend/train_model.py`

## 10. Frontend Implementation

The frontend is developed using React.js and Vite.

The interface provides an interactive dashboard through which users can:

- View the project dashboard
