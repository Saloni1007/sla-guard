# SLA GUARD: Machine Learning-Based SLA Violation Prediction in Cloud Computing

SLA Guard is a machine learning-based web application designed to predict potential Service Level Agreement (SLA) violations in cloud computing workloads.

The system analyzes workload characteristics and resource utilization parameters using a trained Random Forest classification model. It estimates the probability of an SLA violation and classifies the workload based on a selected probability threshold.

The application combines a React-based frontend, a FastAPI backend, and a serialized machine learning model to provide an interactive prediction system.

> **Important: This is a predictive prototype trained using synthetic workload data.** Its validation performance should not be interpreted as proof of performance on real cloud infrastructure. The application predicts potential violations but does not automatically prevent them or modify cloud resources.

---

## 1. Research problem and motivation

Cloud computing environments provide shared computing resources to multiple applications and users. These resources include CPU, memory, storage, and network bandwidth.

Cloud service providers are expected to maintain agreed service quality and performance levels through Service Level Agreements (SLAs).

However, cloud workloads are dynamic. Resource utilization and workload requirements can change continuously depending on application demand.

Several conditions can increase the possibility of SLA violations:

- High CPU utilization.
- High memory utilization.
- Insufficient requested resources.
- Increasing workload queue length.
- Network latency.
- Resource contention.
- Longer estimated execution time.
- Inefficient workload scheduling.

Traditional monitoring systems generally focus on observing current resource conditions. A predictive approach can additionally analyze workload characteristics and estimate whether a workload may violate its SLA.

The objective of SLA Guard is to demonstrate how machine learning can be used to identify potential SLA violation cases from workload-related parameters.

### Main contribution

The project implements an end-to-end predictive workflow:

1. Workload-related features are provided to the application.
2. The frontend sends the input data to the backend.
3. The backend loads the trained Random Forest model.
4. The model estimates the probability of an SLA violation.
5. A validation-selected classification threshold is applied.
6. The prediction is returned to the frontend for display.

The system provides a foundation for further experimentation with predictive cloud resource management.

---

## 2. System overview and workflow

SLA Guard consists of three major components:

1. React frontend.
2. FastAPI backend.
3. Random Forest machine learning model.

### Frontend

The frontend is developed using React.js and Vite.

It provides an interactive interface for entering workload parameters and viewing prediction results.

### Backend

The backend is developed using FastAPI.

It receives the workload information, validates the input, loads the trained model, and generates the prediction response.

### Machine learning model

The machine learning component uses a Random Forest classifier trained on a synthetic cloud workload dataset.

The model generates a probability estimate for the positive class, which represents an SLA violation.

### Prediction workflow

1. The user opens the SLA Guard application.
2. The user enters the required workload parameters.
3. The React frontend collects the input values.
4. The frontend sends a JSON request to the FastAPI backend.
5. The backend validates the received input.
6. The trained Random Forest model processes the workload features.
7. The model generates a probability estimate.
8. The probability is compared with the configured classification threshold.
9. The backend returns the prediction response.
10. The frontend displays the predicted SLA status and probability.

The model is loaded from a saved file rather than being retrained for every prediction request.

---

## 3. Dataset description

### Dataset overview

The project uses a synthetic cloud workload dataset containing:

- **Total records:** 50,000
- **Input features:** 12
- **Target variable:** 1
- **Total columns:** 13

Each record represents a workload observation containing information about resource requirements, resource utilization, scheduling characteristics, and performance-related parameters.

The dataset is used for training, validation, and testing of the machine learning model.

### Input features

The model uses the following 12 input features.

| Feature | Description |
|---|---|
| `cpu_requested` | CPU resources requested by the workload |
| `memory_requested` | Memory resources requested by the workload |
| `cpu_utilization` | CPU utilization of the workload |
| `memory_utilization` | Memory utilization of the workload |
| `machine_cpu_utilization` | CPU utilization of the machine |
| `machine_memory_utilization` | Memory utilization of the machine |
| `queue_length` | Number of workloads waiting in the processing queue |
| `priority` | Priority assigned to the workload |
| `scheduling_class` | Scheduling category of the workload |
| `network_latency_ms` | Network latency measured in milliseconds |
| `estimated_runtime_sec` | Estimated workload execution time in seconds |
| `resource_contention` | Indicator of competition for shared computing resources |

### Target variable

The target variable is:

`SLA_Violation`

It represents the classification outcome.

| Target value | Meaning |
|---|---|
| 0 | No SLA violation |
| 1 | SLA violation |

The model learns relationships between the workload features and the target variable.

### Dataset interpretation

The dataset is synthetic and is used for model development and experimentation.

The generated records should not be interpreted as actual measurements collected from a production cloud cluster.

The model's performance on real-world cloud workload data remains to be established.

---

## 4. Data preparation and splitting

The dataset is divided into three separate subsets.

The purpose of this division is to separate model training from threshold selection and final evaluation.

### Training dataset

The training dataset contains 60% of the complete dataset.

It is used to train the Random Forest classifier and learn relationships between workload characteristics and SLA violation labels.

### Validation dataset

The validation dataset contains 20% of the complete dataset.

It is used to evaluate different classification thresholds and select a threshold according to the desired recall.

The validation dataset is not used to train the final model.

### Testing dataset

The testing dataset contains the remaining 20% of the dataset.

It is reserved for final evaluation and should not be used during training or threshold selection.

### Dataset split summary

| Dataset | Percentage | Number of records |
|---|---:|---:|
| Training | 60% | 30,000 |
| Validation | 20% | 10,000 |
| Testing | 20% | 10,000 |
| **Total** | **100%** | **50,000** |

This separation helps evaluate the model on observations that were not used for fitting or threshold selection.

---

## 5. Machine learning model

### Random Forest Classifier

SLA Guard uses the Random Forest classification algorithm from Scikit-learn.

Random Forest is an ensemble learning algorithm that combines multiple decision trees to generate a classification result.

Instead of depending on a single decision tree, the algorithm builds several trees using different bootstrap samples and feature subsets.

Each tree generates a prediction, and the forest combines the predictions to estimate the final class probability.

### Working principle

The Random Forest algorithm follows these steps:

1. Different bootstrap samples are generated from the training dataset.
2. A decision tree is trained using each sample.
3. A subset of features is considered at each decision split.
4. Each tree learns patterns between workload features and SLA violation labels.
5. The predictions of the individual trees are combined.
6. The model produces a class probability estimate.

For SLA Guard, the probability of the positive class is used for threshold-based classification.

### Why Random Forest was selected

Random Forest was selected because:

- It can capture nonlinear relationships between input features.
- It combines multiple decision trees.
- It supports binary classification.
- It can handle multiple workload-related features.
- It provides class probability estimates.
- It supports class weighting for imbalanced datasets.
- It generally reduces dependence on the behavior of a single decision tree.

### Model configuration

The implemented model uses the following hyperparameters:

| Parameter | Value | Purpose |
|---|---|---|
| `n_estimators` | 300 | Number of decision trees |
| `max_features` | `sqrt` | Features considered at each split |
| `min_samples_leaf` | 2 | Minimum samples required in a leaf |
| `class_weight` | `balanced_subsample` | Adjusts class weights for bootstrap samples |
| `random_state` | 42 | Supports reproducibility |
| `n_jobs` | -1 | Uses available CPU processing resources |

### Explanation of hyperparameters

**Number of estimators**

The model uses 300 decision trees. The predictions from these trees are combined to produce the final classification probability.

**Maximum features**

The `sqrt` setting uses a square-root-based subset of available features when considering candidate splits.

**Minimum samples per leaf**

The minimum number of samples required in a leaf node is set to 2.

This helps avoid creating leaf nodes based on a single training observation.

**Class weighting**

The model uses `balanced_subsample`.

This adjusts class weights according to the class distribution in individual bootstrap samples.

It does not generate additional dataset records.

**Random state**

The random state is set to 42 to support reproducibility when the same data and configuration are used.

**Parallel processing**

The `n_jobs=-1` setting allows Scikit-learn to use available CPU processing resources.

---

## 6. Class imbalance handling

Binary classification datasets may contain different numbers of observations belonging to each class.

This is known as class imbalance.

In SLA violation prediction, identifying actual violation cases is important because missed violations may reduce the usefulness of the predictive system.

The current implementation uses:

`class_weight="balanced_subsample"`

This approach adjusts the importance of classes during Random Forest training.

The weights are calculated based on the class distribution of individual bootstrap samples.

### Why class weighting is used

Class weighting helps the model account for differences in class frequencies without directly changing the number of observations in the dataset.

It is different from oversampling methods because it does not create synthetic minority-class observations.

### Future experimentation

Other imbalance-handling methods could be investigated, including:

- Random oversampling.
- Random undersampling.
- SMOTE.
- Edited Nearest Neighbours.
- SMOTE-ENN.

These methods have not been used in the current implementation and would require separate experiments.

---

## 7. Classification threshold optimization

### Understanding classification probability

The Random Forest classifier produces a probability estimate for the positive class.

The positive class represents an SLA violation.

A classification threshold is used to convert the probability into a final binary prediction.

The default threshold is commonly 0.50.

However, the project uses a lower threshold to
