# SLA Guard
## Machine Learning-Based SLA Violation Prediction System for Cloud Computing

---

## Table of Contents

1. Project Overview
2. Problem Statement
3. Motivation
4. Project Objectives
5. Technology Stack
6. System Architecture
7. System Workflow
8. Dataset Description
9. Dataset Features
10. Target Variable
11. Data Preparation and Splitting
12. Machine Learning Approach
13. Random Forest Classifier
14. Model Configuration
15. Handling Class Imbalance
16. Classification Threshold Optimization
17. Model Training Workflow
18. Evaluation Metrics
19. Model Validation Results
20. Backend Implementation
21. Frontend Implementation
22. Frontend-Backend Communication
23. Model Storage
24. Project Structure
25. Installation and Setup
26. Running the Application
27. API Testing
28. Example Prediction
29. Advantages
30. Limitations
31. Future Scope
32. Conclusion
33. GitHub Repository

---

# 1. Project Overview

SLA Guard is a machine learning-based web application developed to predict potential Service Level Agreement (SLA) violations in cloud computing workloads.

Cloud computing environments execute multiple workloads simultaneously, sharing resources such as CPU, memory, and network bandwidth. When resource demand increases or resources become insufficient, workload performance may be affected.

A Service Level Agreement defines the expected level of service between a cloud service provider and its customers. Failure to meet these service requirements may result in an SLA violation.

SLA Guard uses machine learning to analyze workload-related parameters and estimate the probability of an SLA violation.

The project uses a Random Forest classification model trained on a synthetic cloud workload dataset. The trained model is integrated with a FastAPI backend and a React-based frontend.

### Key Features

- Machine learning-based SLA violation prediction.
- Analysis of CPU and memory utilization.
- Workload queue length and priority analysis.
- Network latency and execution time consideration.
- Random Forest classification model.
- Probability-based prediction.
- Classification threshold optimization.
- FastAPI-based prediction service.
- React-based user interface.

---

# 2. Problem Statement

Cloud computing platforms manage multiple workloads with different resource requirements.

Each workload may require a particular amount of CPU, memory, and network resources. When several workloads compete for limited resources, performance may be affected.

Factors such as high resource utilization, increased queue length, network latency, and resource contention can contribute to workload performance problems.

Traditional monitoring focuses on observing current resource conditions. However, identifying workloads that may be associated with SLA violations is also important.

The problem addressed by SLA Guard is:

**To develop a machine learning-based system that analyzes cloud workload parameters and predicts whether a workload is likely to be associated with an SLA violation.**

The problem is formulated as a binary classification task with two possible outputs:

- 0 – No SLA violation.
- 1 – SLA violation.

---

# 3. Motivation

Cloud computing has become an important part of modern IT infrastructure. Applications depend on cloud resources to maintain their expected performance and availability.

Workload requirements may change depending on application complexity, user demand, and resource availability.

A workload that performs normally under one resource condition may experience performance issues under another.

Analyzing multiple workload parameters together can provide useful information about workload conditions.

The motivation behind SLA Guard is to explore how machine learning can support the identification of potential SLA violation cases.

The project combines workload analysis, supervised machine learning, backend API development, and frontend integration into one application.

---

# 4. Project Objectives

The main objectives of SLA Guard are:

1. Develop a machine learning model for predicting potential SLA violations.
2. Analyze workload and resource utilization parameters.
3. Train a Random Forest classification model.
4. Investigate classification threshold optimization.
5. Integrate the trained model with a FastAPI backend.
6. Develop a React-based interface for submitting workload information.
7. Display prediction results and violation probabilities.
8. Demonstrate a complete local machine learning web application.

---

# 5. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Frontend | React.js | User interface |
| Frontend tooling | Vite | Development environment |
| Backend | Python | Backend programming |
| API Framework | FastAPI | Prediction API |
| Machine Learning | Scikit-learn | Model development |
| Algorithm | Random Forest | Classification |
| Data Processing | Pandas | Dataset handling |
| Numerical Processing | NumPy | Numerical operations |
| Model Storage | Joblib | Model serialization |
| Version Control | Git | Source code management |
| Repository | GitHub | Project hosting |

---

# 6. System Architecture

SLA Guard follows a frontend-backend-machine-learning architecture.

The frontend collects workload parameters and sends them to the backend. The backend processes the input and passes it to the trained Random Forest model.

The model generates a probability estimate for the SLA violation class. The backend applies the selected threshold and returns the prediction to the frontend.

### Main Components

**Frontend Layer**
- Developed using React.js.
- Collects workload input parameters.
- Sends requests to the backend.
- Displays prediction results.

**Backend Layer**
- Developed using FastAPI.
- Receives prediction requests.
- Loads the trained machine learning model.
- Processes input features.
- Returns prediction results.

**Machine Learning Layer**
- Uses a Random Forest classifier.
- Processes 12 input features.
- Estimates the probability of an SLA violation.
- Generates a binary classification.

**Model Storage Layer**
- Stores the trained model in a `.pkl` file.
- Stores the selected classification threshold separately.

### Architecture Diagram

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
    PROBABILITY ESTIMATE
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

# 7. System Workflow

The application follows the workflow described below:

1. The user enters workload-related parameters through the frontend.
2. The frontend prepares the input data.
3. The data is sent to the FastAPI backend.
4. The backend receives and processes the input.
5. The trained Random Forest model generates a probability estimate.
6. The selected classification threshold is applied.
7. The backend generates a binary prediction.
8. The prediction response is returned to the frontend.
9. The frontend displays the result.

This workflow allows users to access the machine learning model through a web interface without executing the training code manually.

---

# 8. Dataset Description

The project uses a synthetic cloud workload dataset containing 50,000 records and 13 columns.

The dataset consists of 12 input features and one target variable.

It represents workload and resource-related characteristics that may be relevant to SLA violation prediction.

### Dataset Summary

| Property | Description |
|---|---|
| Dataset | sla_guard_50000_unique_logs.csv |
| Records | 50,000 |
| Total Columns | 13 |
| Input Features | 12 |
| Target Variable | SLA_Violation |
| Learning Type | Supervised Learning |
| Problem Type | Binary Classification |

The dataset is synthetic and is used for model development and experimentation. It should not be interpreted as real-world cloud infrastructure data.

---

# 9. Dataset Features

The dataset contains the following 12 input features.

| Feature | Description |
|---|---|
| cpu_requested | CPU resources requested by a workload |
| memory_requested | Memory resources requested by a workload |
| cpu_utilization | CPU utilization associated with the workload |
