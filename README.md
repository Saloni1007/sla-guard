# SLA Guard
## Machine Learning-Based SLA Violation Prediction System for Cloud Computing

---

## Table of Contents

1. Project Overview
2. Problem Statement
3. Motivation
4. Project Objectives
5. Scope of the Project
6. Technology Stack
7. System Architecture
8. Overall System Workflow
9. Dataset Description
10. Dataset Features
11. Target Variable
12. Data Preparation
13. Dataset Splitting Strategy
14. Machine Learning Approach
15. Random Forest Classifier
16. Model Configuration
17. Handling Class Imbalance
18. Classification Probability
19. Classification Threshold Optimization
20. Model Training Workflow
21. Model Validation
22. Evaluation Metrics
23. Validation Results
24. Backend Implementation
25. FastAPI Framework
26. Model Loading and Prediction
27. Prediction API
28. Frontend Implementation
29. Frontend-Backend Communication
30. Dashboard and Prediction Output
31. Model Serialization
32. Project Directory Structure
33. Installation Requirements
34. Local Setup
35. Running the Application
36. API Testing
37. Example Prediction Workflow
38. Advantages
39. Limitations
40. Future Scope
41. Conclusion
42. GitHub Repository

---

# 1. Project Overview

SLA Guard is a machine learning-based web application developed to predict potential Service Level Agreement (SLA) violations in cloud computing workloads.

Cloud computing platforms provide computing resources such as CPU, memory, storage, and network connectivity to users and applications. These resources are shared among multiple workloads, making resource management an important aspect of cloud infrastructure.

An SLA defines the expected level of service between a service provider and a customer. It may specify performance requirements, availability, response time, and other service conditions.

When a cloud workload does not meet its expected service requirements, an SLA violation may occur.

SLA Guard uses machine learning to analyze workload-related parameters and estimate the probability of an SLA violation.

The system uses a Random Forest classification model trained on a synthetic cloud workload dataset.

The trained model is integrated into a FastAPI backend, while a React-based frontend provides an interface for entering workload parameters and viewing prediction results.

The primary purpose of the application is to demonstrate how machine learning can support the early identification of workloads that may be associated with SLA violations.

### Key Features

- Machine learning-based SLA violation prediction.
- Analysis of CPU and memory utilization.
- Consideration of workload queue length and priority.
- Network latency and runtime-based input analysis.
- Random Forest classification model.
- Classification threshold optimization.
- FastAPI-based prediction service.
- React-based interactive interface.
- Probability-based prediction output.
- Local model integration using Joblib.

---

# 2. Problem Statement

Cloud computing environments execute multiple workloads simultaneously.

Each workload may require different amounts of CPU, memory, network resources, and execution time.

When resource demand increases or available resources become insufficient, workload performance may be affected.

Several factors can contribute to SLA violations, including:

- High CPU utilization.
- High memory utilization.
- Increased queue length.
- Network latency.
- Resource contention.
- Long execution time.
- Inappropriate resource allocation.

Traditional resource monitoring systems generally focus on observing current resource conditions.

However, identifying a workload that may lead to an SLA violation before the violation occurs can be useful for resource management.

The problem addressed by SLA Guard is:

**To develop a machine learning-based system that analyzes cloud workload parameters and predicts whether a workload is likely to be associated with an SLA violation.**

The problem is treated as a binary classification task.

The output represents one of two classes:

- No SLA violation.
- SLA violation.

---

# 3. Motivation

The increasing adoption of cloud computing has made efficient resource utilization and service reliability important requirements.

Cloud applications often operate under changing workloads.

Resource requirements may increase or decrease depending on the number of users, application complexity, and processing demands.

A workload that performs normally under one resource condition may experience performance issues under another.

Monitoring individual resource parameters can provide useful information, but analyzing multiple parameters together may provide a broader view of workload conditions.

Machine learning offers a way to learn relationships between workload characteristics and previously observed target labels.

The motivation behind SLA Guard is to explore this approach through a complete web-based prediction system.

The project combines:

- Cloud workload feature analysis.
- Supervised machine learning.
- Probability-based classification.
- Backend API development.
- Frontend integration.

The application demonstrates how a trained model can be made accessible through a web interface rather than being restricted to a standalone Python script.

---

# 4. Project Objectives

The major objectives of SLA Guard are listed below.

### 4.1 Prediction of SLA Violations

Develop a classification model capable of predicting potential SLA violations using workload-related parameters.

### 4.2 Workload Analysis

Analyze multiple input features, including CPU utilization, memory utilization, queue length, and network latency.

### 4.3 Machine Learning Integration

Train a Random Forest classifier and integrate the saved model into a backend application.

### 4.4 Threshold Optimization

Investigate the classification threshold to improve the identification of positive SLA violation cases.

### 4.5 Web Application Development

Build a user interface that allows users to provide workload parameters and view prediction results.

### 4.6 API Development

Develop a FastAPI backend to receive input data, perform predictions, and return structured responses.

### 4.7 Model Accessibility

Make the trained machine learning model available through a locally running web application.

---

# 5. Scope of the Project

SLA Guard focuses on predicting SLA violations from workload-related input parameters.

The current implementation includes:

1. A synthetic cloud workload dataset.
2. Data preparation and feature selection.
3. Random Forest model training.
4. Validation-based threshold optimization.
5. Model serialization.
6. FastAPI backend integration.
7. React frontend integration.
8. Prediction probability and classification output.

The system currently operates as a prediction prototype.

It does not directly interact with a real cloud infrastructure or automatically modify resource allocation.

It also does not automatically prevent SLA violations.

The project focuses on identifying potential violations through machine learning predictions.

---

# 6. Technology Stack

The application uses a combination of frontend, backend, and machine learning technologies.

| Component | Technology | Purpose |
|---|---|---|
| Frontend | React.js | User interface |
| Frontend tooling | Vite | Development server and build tooling |
| Backend | Python | Backend programming language |
| API framework | FastAPI | Prediction API |
| Machine learning | Scikit-learn | Model development |
| Model | Random Forest | Binary classification |
| Data processing | Pandas | Dataset handling |
| Numerical processing | NumPy | Numerical operations |
| Model storage | Joblib | Model serialization |
| Version control | Git | Source code management |
| Repository hosting | GitHub | Project storage |

---

# 7. System Architecture

SLA Guard follows a frontend-backend-machine-learning architecture.

The frontend is responsible for collecting workload information and displaying prediction results.

The backend receives the submitted data and communicates with the trained machine learning model.

The model processes the input features and generates a probability estimate for the positive class.

The backend applies the selected classification threshold and returns the prediction.

### Main Components

**1. Frontend Layer**

- Developed using React.js.
- Provides input fields for workload parameters.
- Sends prediction requests to the backend.
- Displays the prediction response.

**2. Backend Layer**

- Developed using FastAPI.
- Receives requests from the frontend.
- Loads the trained model.
- Processes input features.
- Returns prediction results.

**3. Machine Learning Layer**

- Uses a Random Forest classifier.
- Accepts 12 input features.
- Estimates the probability of an SLA violation.
- Produces a binary classification after threshold application.

**4. Model Storage Layer**

- Stores the trained model in a `.pkl` file.
- Stores the selected classification threshold separately.

### Architecture Flow

```text
             USER
               |
               v
       REACT FRONTEND
               |
               v
        API REQUEST
               |
               v
       FASTAPI BACKEND
               |
               v
       INPUT PROCESSING
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
       API RESPONSE
               |
               v
       REACT DASHBOARD
```

---

# 8. Overall System Workflow

The application follows a sequence of steps whenever a user requests a prediction.

### Step 1: Input Collection

The user enters workload-related parameters through the frontend.

### Step 2: Request Submission

The frontend sends the entered values to the backend using an API request.

### Step 3: Input Processing

The backend receives the submitted data and prepares it in the feature order expected by the model.

### Step 4: Model Execution

The trained Random Forest model processes the input features.

### Step 5: Probability Estimation

The model calculates the probability associated with the SLA violation class.

### Step 6: Threshold Application

The selected classification threshold is applied to convert the probability into a binary prediction.

### Step 7: Response Generation

The backend prepares the prediction result and returns it to the frontend.

### Step 8: Result Display

The frontend displays the prediction and associated probability.

---

# 9. Dataset Description

The project uses a synthetic cloud workload dataset containing 50,000 records and 13 columns.

The dataset consists of:

- 12 input features.
- 1 target variable.

The dataset represents workload and resource-related characteristics that may be relevant to SLA violation prediction.

The dataset is synthetic and is used for experimentation and model development.

It should not be considered a collection of actual production cloud workload observations.

### Dataset Summary

| Property | Description |
|---|---|
| Dataset name | sla_guard_50000_unique_logs.csv |
| Number of records | 50,000 |
| Total columns | 13 |
| Input features | 12 |
| Target variable | SLA_Violation |
| Learning type | Supervised learning |
| Problem type | Binary classification |

### Dataset Purpose

The dataset is used to train a model that learns patterns between workload-related features and the target classification.

Each record represents an individual workload observation.

The model uses the input columns to estimate the target class.

---

# 10. Dataset Features

The dataset contains 12 input features.

These features describe resource requirements, utilization, scheduling information, and workload conditions.

### 10.1 cpu_requested

Represents the CPU resources requested by a workload.

It describes the amount of CPU capacity required for executing the workload.

### 10.2 memory_requested

Represents the memory resources requested by a workload.

Memory requirements can influence workload execution and resource availability.

### 10.3 cpu_utilization

Represents CPU utilization associated with the workload.

High CPU utilization may indicate increased processing demand.

### 10.4 memory_utilization

Represents memory utilization associated with the workload.

It provides information about memory consumption.

### 10.5 machine_cpu_utilization

Represents CPU utilization at the machine level.

It provides information about the overall CPU load of the machine executing the workload.

### 10.6 machine_memory_utilization

Represents machine-level memory utilization.

It describes memory consumption at the machine level.

### 10.7 queue_length

Represents the number of workloads or tasks waiting in the processing queue.

An increased queue length may indicate higher workload demand or processing delays.

### 10.8 priority

Represents the priority assigned to a workload.

Priority information can be useful when analyzing scheduling behaviour.

### 10.9 scheduling_class

Represents the scheduling category assigned to a workload.

It provides information about the workload's scheduling characteristics.

### 10.10 network_latency_ms

Represents network latency measured in milliseconds.

Network latency can affect communication and application response time.

### 10.11 estimated_runtime_sec

Represents the estimated execution time of a workload in seconds.

It provides an estimate of how long the workload is expected to run.

### 10.12 resource_contention

Represents the resource contention feature in the dataset.

Resource contention can occur when multiple workloads compete for shared resources.

---

# 11. Target Variable

The target column used for model training is:

`SLA_Violation`

It represents the class that the model is trained to predict.

The target variable contains two possible classes.

| Value | Meaning |
|---|---|
| 0 | No SLA violation |
| 1 | SLA violation |

The project uses binary classification because the prediction problem has two possible outcomes.

The trained model estimates the probability of the positive class, which represents an SLA violation.

---

# 12. Data Preparation

Data preparation is an important stage of the machine learning workflow.

Before training the classifier, the dataset must be separated into input features and the target variable.

### Feature Selection

The 12 workload-related columns are used as input features.

The target column `SLA_Violation` is separated from the input data.

### Feature Consistency

The same feature names and ordering must be maintained during model training and prediction.

This is important because the model expects input data in the format used during training.

### Data Integrity

The dataset is used as the basis for training, validation, and testing.

Any preprocessing applied during training must also be reflected in the prediction pipeline where required.

---

# 13. Dataset Splitting Strategy

The dataset is divided into three subsets.

The selected split is:

| Subset | Percentage | Number of Records |
|---|---:|---:|
| Training | 60% | 30,000 |
| Validation | 20% | 10,000 |
| Testing | 20% | 10,000 |
| Total | 100% | 50,000 |

### 13.1 Training Dataset

The training dataset is used to train the Random Forest classifier.

It allows the model to learn relationships between input features and target labels.

### 13.2 Validation Dataset

The validation dataset is used to evaluate model behaviour during development.

It is also used for classification threshold selection.

### 13.3 Testing Dataset

The test dataset is reserved for final evaluation.

It should not be used to select the classification threshold or tune the model.

Keeping the test dataset separate helps avoid reporting validation performance as final test performance.

---

# 14. Machine Learning Approach

SLA Guard uses supervised machine learning.

In supervised learning, a model is trained using input data along with corresponding target labels.

The model learns patterns from the training examples and uses the learned patterns to make predictions on new inputs.

The project treats SLA violation prediction as a binary classification problem.

### Why Classification?

The required output is one of two possible classes:

- No SLA violation.
- SLA violation.

The model therefore predicts a class rather than a continuous numerical value.

### Selected Algorithm

The selected algorithm is the Random Forest Classifier.

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

The combined predictions of the trees are used to generate the final classification probability.

---

# 15. Random Forest Classifier

Random Forest is an ensemble machine learning algorithm based on decision trees.

Instead of relying on a single decision tree, it constructs multiple trees during training.

Each tree learns patterns from the training data.

The model combines the predictions of these trees to produce an overall result.

### Working Principle

1. Multiple decision trees are constructed.
2. Each tree learns from training samples.
3. Random subsets of features are considered during tree construction.
4. Individual trees generate predictions.
5. Their outputs are combined to produce the final result.

### Why Random Forest Was Selected

Random Forest was selected because:

- It supports classification problems.
- It can model nonlinear relationships.
- It combines multiple decision trees.
- It can generate class probability estimates.
- It provides a practical baseline for structured tabular data.

The selection does not establish that Random Forest is the best possible algorithm for this dataset. Comparing additional models would be required to make that conclusion.

---

# 16. Model Configuration

The Random Forest classifier uses the following configuration:

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

### Parameter Explanation

**n_estimators = 300**

The model constructs 300 decision trees.

Increasing the number of trees can affect model stability and computational cost.

**max_features = "sqrt"**

The square root of the total number of input features is considered when searching for the best split at each tree node.

**min_samples_leaf = 2**

A leaf node must contain at least two samples.

This parameter helps control the complexity of individual trees.

**class_weight = "balanced_subsample"**

Class weights are adjusted using the class distribution in individual bootstrap samples.

This helps the classifier account for differences in class frequency.

**random_state = 42**

A fixed random state supports reproducibility of the experiment.

**n_jobs = -1**

The model uses all available CPU processors for parallel processing where supported.

---

# 17. Handling Class Imbalance

Class imbalance occurs when one target class contains significantly more records than another.

In classification problems, this can affect how a model learns and how its predictions should be evaluated.

For SLA violation prediction, identifying positive violation cases is important.

A model that predicts the majority class too frequently may fail to identify a substantial number of violation cases.

### Technique Used

The current implementation uses:

`class_weight="balanced_subsample"`

This parameter adjusts class weights during Random Forest training.

It helps account for the distribution of classes in the training samples.

### Other Possible Techniques

The following techniques have not been implemented in the current model but may be investigated in future experiments:

- Random oversampling.
- Random undersampling.
- SMOTE.
- Edited Nearest Neighbours.
- SMOTE-ENN.

These methods should be tested independently before deciding whether they improve model performance.

---

# 18. Classification Probability

The Random Forest classifier can generate a probability estimate for each class.

For SLA Guard, the positive class represents an SLA violation.

The model estimates the probability that an input workload belongs to the positive class.

For example, a model output of 0.72 represents an estimated positive-class probability of 72%.

The probability is then used by the classification threshold to determine the final prediction.

A probability estimate is not a guarantee that a violation will occur.

It is a model-generated estimate based on patterns learned from the training dataset.

---

# 19. Classification Threshold Optimization

A classification threshold is used to convert a probability estimate into a binary prediction.

The default classification threshold commonly used in binary classification is 0.50.

However, the default threshold is not necessarily suitable for every classification problem.

SLA Guard uses a lower threshold of approximately 0.30.

### Threshold Decision

The prediction can be described as:

```text
If violation probability >= threshold:
    Predict SLA Violation (1)

Otherwise:
    Predict No SLA Violation (0)
```

### Why a Lower Threshold?

The project places importance on identifying potential SLA violation cases.

A lower threshold can classify more workloads as positive.

This may increase recall, although it can also increase false-positive predictions and reduce precision.

Therefore, threshold selection involves balancing multiple evaluation metrics.

### Threshold Selection Process

1. Train the model using training data.
2. Generate probability estimates on validation data.
3. Evaluate different classification thresholds.
4. Calculate precision and recall at each threshold.
5. Select a threshold based on the recall requirement and precision.
6. Save the selected threshold for use by the backend.

The threshold was selected using validation data rather than the final test dataset.

---

# 20. Model Training Workflow

The model training workflow consists of the following stages:

**Stage 1: Dataset Loading**

Load the synthetic dataset using Pandas.

**Stage 2: Feature and Target Separation**

Separate the 12 input features from the target column.

**Stage 3: Dataset Splitting**

Divide the dataset into training, validation, and testing subsets.

**Stage 4: Model Initialization**

Create the Random Forest classifier using the selected parameters.

**Stage 5: Model Training**

Train the classifier using the training dataset.

**Stage 6: Probability Prediction**

Generate probability estimates using validation data.

**Stage 7: Threshold Optimization**

Evaluate different thresholds and select the preferred threshold.

**Stage 8: Model Saving**

Save the trained model and selected threshold for backend integration.

---

# 21. Model Validation

Validation is performed to examine how the trained model behaves on data not used for fitting the model.

The validation dataset is separate from the training dataset.

The project uses validation data for threshold optimization.

The selected threshold is approximately 0.30.

The validation metrics are calculated after applying this threshold.

### Importance of Validation

Validation helps in:

- Examining model prediction behaviour.
- Comparing classification thresholds.
- Understanding precision and recall.
- Selecting an operating threshold.
- Reducing the risk of using the test dataset during development.

The validation results should not be presented as final test results.

---

# 22. Evaluation Metrics

The project considers classification metrics to understand the model's prediction performance.

### 22.1 Precision

Precision measures the proportion of predicted positive cases that are actually positive.

Formula:

```text
Precision = TP / (TP + FP)
```

Where:

- TP = True Positives.
- FP = False Positives.

Higher precision means fewer false-positive predictions among the predicted positive cases.

### 22.2 Recall

Recall measures the proportion of actual positive cases correctly identified by the model.

Formula:

```text
Recall = TP / (TP + FN)
```

Where:

- TP = True Positives.
- FN = False Negatives.

Recall is important when missing an actual SLA violation case is costly.

### 22.3 F1-Score

F1-score is the harmonic mean of precision and recall.

Formula:

```text
F1 = 2 * (Precision * Recall) /
     (Precision + Recall)
```

It provides a combined measure of precision and recall.

### 22.4 Confusion Matrix

A confusion matrix summarizes classification outcomes.

| | Predicted No Violation | Predicted Violation |
|---|---|---|
| Actual No Violation | True Negative | False Positive |
| Actual Violation | False Negative | True Positive |

The confusion matrix helps identify the types of prediction errors made by the classifier.

---

# 23. Validation Results

At the selected classification threshold of approximately 0.30, the reported validation results are:

| Metric | Validation Result |
|---|---:|
| Precision | 57.72% |
| Recall | 85.31% |
| Selected threshold | Approximately 0.30 |

### Interpretation

**Precision: 57.72%**

Approximately 57.72% of the cases predicted as SLA violations were actual positive cases in the validation dataset.

**Recall: 85.31%**

Approximately 85.31% of the actual SLA violation cases in the validation dataset were identified by the model.

### Performance Consideration

The results show a recall-oriented operating point.

This means the selected threshold prioritizes identifying a larger proportion of positive cases.

However, the precision value also indicates that some predicted violations may be false positives.

The choice of threshold depends on the relative cost of false positives and false negatives.

**Important:** These are validation results. They must not be reported as final test-set results.

---

# 24. Backend Implementation

The backend is developed using Python and FastAPI.

It acts as the communication layer between the React frontend and the machine learning model.

### Responsibilities

The backend performs the following tasks:

- Receives prediction requests.
- Accepts workload-related input features.
- Loads the trained Random Forest model.
- Reads the saved classification threshold.
- Prepares input data for prediction.
- Generates probability estimates.
- Applies the classification threshold.
- Returns the prediction response.

The backend uses the saved model instead of training the classifier whenever a user requests a prediction.

This separates model development from model inference.

---

# 25. FastAPI Framework

FastAPI is a Python web framework used to build APIs.

It is used in SLA Guard to expose the trained machine learning model through an HTTP endpoint.

### Why FastAPI?

FastAPI provides:

- Request handling.
- Data validation.
- Structured API responses.
- Automatic interactive API documentation.
- Integration with Python machine learning libraries.

The backend can be tested using its automatically generated Swagger documentation.

---

# 26. Model Loading and Prediction

The trained model is stored as a serialized file.

The backend loads this file using Joblib.

The saved model contains the trained classifier and can be used to generate predictions without repeating the complete training process.

### Prediction Process

1. Receive input features.
2. Prepare the feature values.
3. Load or access the trained model.
4. Generate positive-class probability.
5. Read the selected threshold.
6. Compare probability with threshold.
7. Return the prediction.

The input feature order must match the order expected by the trained model.

---

# 27. Prediction API

The backend provides a prediction endpoint:

`POST /predict`

The endpoint accepts workload-related parameters and returns the model's prediction result.

### API Purpose

The endpoint connects the frontend to the machine learning model.

It allows the application to perform inference without exposing the model implementation directly to the frontend.

### Local API Address

```text
http://127.0.0.1:8001/predict
```

### API Documentation

```text
http://127.0.0.1:8001/docs
```

The Swagger interface can be used to inspect the endpoint and submit test requests.

---

# 28. Frontend Implementation

The frontend is developed using React.js and Vite.

React is used to create the user interface, while Vite provides the development environment.

### Frontend Responsibilities

- Display the application interface.
- Collect workload input parameters.
- Submit input data to the backend.
- Receive prediction responses.
- Display the predicted class.
- Display the violation probability.

The frontend provides a more accessible way to interact with the machine learning model.

Users do not need to execute Python code manually to obtain a prediction.

---

# 29. Frontend-Backend Communication

The frontend communicates with the backend through HTTP requests.

The general communication process is:

1. The user enters the required workload values.
2. React prepares the request data.
3. The frontend sends the data to the FastAPI endpoint.
4. FastAPI validates and processes the request.
5. The model generates the prediction.
6. The backend returns a response.
7. React updates the displayed result.

The backend and frontend must both be running locally for the complete application workflow to function.

Cross-Origin Resource Sharing (CORS) configuration is
