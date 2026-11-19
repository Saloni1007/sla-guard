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
| memory_utilization | Memory utilization associated with the workload |
| machine_cpu_utilization | CPU utilization at machine level |
| machine_memory_utilization | Memory utilization at machine level |
| queue_length | Number of workloads waiting in the queue |
| priority | Priority assigned to the workload |
| scheduling_class | Scheduling category of the workload |
| network_latency_ms | Network latency measured in milliseconds |
| estimated_runtime_sec | Estimated execution time in seconds |
| resource_contention | Resource contention feature |

### Importance of Features

These features describe different aspects of workload behaviour.

- CPU and memory features represent resource requirements and utilization.
- Queue length provides information about pending workloads.
- Priority and scheduling class describe scheduling characteristics.
- Network latency represents communication delay.
- Estimated runtime represents expected workload execution time.
- Resource contention represents competition for shared resources.

The model uses these features together to learn patterns associated with the target variable.

---

# 10. Target Variable

The target variable used for training is:

`SLA_Violation`

It represents whether a workload belongs to the SLA violation class.

| Value | Meaning |
|---|---|
| 0 | No SLA violation |
| 1 | SLA violation |

The model learns to predict this target using the 12 input features.

Since the target contains two classes, the problem is treated as binary classification.

---

# 11. Data Preparation and Splitting

The dataset is divided into input features and the target variable.

The 12 workload-related columns are used as input features, while `SLA_Violation` is used as the target.

The dataset is divided into three subsets:

| Dataset | Percentage | Records |
|---|---:|---:|
| Training | 60% | 30,000 |
| Validation | 20% | 10,000 |
| Testing | 20% | 10,000 |
| Total | 100% | 50,000 |

### Training Dataset

The training dataset is used to train the Random Forest classifier.

### Validation Dataset

The validation dataset is used to examine model performance and select the classification threshold.

### Testing Dataset

The testing dataset is reserved for final evaluation.

It should remain separate from model development and threshold selection to provide an independent assessment.

Maintaining the same feature names and order during training and prediction is important for consistent model input.

---

# 12. Machine Learning Approach

SLA Guard uses supervised machine learning.

In supervised learning, a model learns patterns from input data and corresponding target labels.

The trained model can then use these learned patterns to make predictions on new input data.

The project treats SLA violation prediction as a binary classification problem.

The selected algorithm is the Random Forest Classifier.

---

# 13. Random Forest Classifier

Random Forest is an ensemble learning algorithm based on multiple decision trees.

Instead of relying on a single decision tree, it constructs multiple trees and combines their outputs to produce a final prediction.

### Working Principle

1. Multiple decision trees are constructed.
2. Each tree learns patterns from training samples.
3. Random subsets of features are considered during tree construction.
4. Individual trees generate predictions.
5. Their outputs are combined to produce the final result.

### Why Random Forest?

Random Forest was selected because:

- It supports classification tasks.
- It can learn nonlinear relationships.
- It combines multiple decision trees.
- It can generate class probability estimates.
- It provides a practical approach for structured tabular data.

The project does not establish that Random Forest is superior to all other algorithms. Additional model comparisons would be required to make that conclusion.

---

# 14. Model Configuration

The Random Forest classifier uses the following parameters:

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

**max_features = "sqrt"**

The square root of the total number of features is considered when searching for the best split at each tree node.

**min_samples_leaf = 2**

Each leaf node must contain at least two samples.

**class_weight = "balanced_subsample"**

Class weights are adjusted according to the class distribution in individual bootstrap samples.

**random_state = 42**

A fixed random state supports reproducibility.

**n_jobs = -1**

Uses all available CPU processors for parallel processing where supported.

---

# 15. Handling Class Imbalance

Class imbalance occurs when one target class contains significantly more records than another.

In classification problems, this can affect how a model learns and predicts.

SLA Guard uses:

`class_weight="balanced_subsample"`

This parameter adjusts class weights during Random Forest training to account for differences in class frequency.

### Other Possible Techniques

The following techniques are potential future experiments and are not part of the current trained model:

- Random oversampling.
- Random undersampling.
- SMOTE.
- Edited Nearest Neighbours.
- SMOTE-ENN.

Their effectiveness would need to be evaluated experimentally before incorporating them into the model.

---

# 16. Classification Threshold Optimization

A classification threshold converts a predicted probability into a binary class.

The commonly used default threshold is 0.50. However, this threshold may not always be suitable for a particular classification problem.

SLA Guard uses a lower threshold of approximately 0.30.

### Threshold Decision

```text
If violation probability >= threshold:
    Predict SLA Violation (1)

Otherwise:
    Predict No SLA Violation (0)
```

### Why a Lower Threshold?

The project prioritizes identifying a larger proportion of potential violation cases.

A lower threshold can increase recall, but it may also increase false-positive predictions and reduce precision.

Therefore, threshold selection involves balancing multiple classification metrics.

### Threshold Selection Process

1. Train the model using training data.
2. Generate probability estimates on validation data.
3. Evaluate different classification thresholds.
4. Calculate precision and recall at each threshold.
5. Select a threshold based on the recall requirement and precision.
6. Save the selected threshold for backend use.

The threshold is selected using validation data, not the final test dataset.

---

# 17. Model Training Workflow

The model training process consists of the following stages:

**Stage 1: Dataset Loading**

Load the synthetic dataset using Pandas.

**Stage 2: Feature Selection**

Separate the 12 input features from the target column.

**Stage 3: Dataset Splitting**

Divide the dataset into training, validation, and testing subsets.

**Stage 4: Model Initialization**

Initialize the Random Forest classifier using the selected parameters.

**Stage 5: Model Training**

Train the classifier using the training dataset.

**Stage 6: Probability Prediction**

Generate probability estimates using validation data.

**Stage 7: Threshold Optimization**

Evaluate different thresholds and select the preferred threshold.

**Stage 8: Model Saving**

Save the trained model and classification threshold for backend integration.

---

# 18. Evaluation Metrics

The project uses classification metrics to understand the model's prediction behaviour.

### Precision

Precision measures the proportion of predicted positive cases that are actually positive.

```text
Precision = TP / (TP + FP)
```

Where:

- TP = True Positives.
- FP = False Positives.

### Recall

Recall measures the proportion of actual positive cases correctly identified by the model.

```text
Recall = TP / (TP + FN)
```

Where:

- TP = True Positives.
- FN = False Negatives.

Recall is particularly important when missing an actual SLA violation case is costly.

### F1-Score

F1-score is the harmonic mean of precision and recall.

```text
F1 = 2 * (Precision * Recall) /
     (Precision + Recall)
```

### Confusion Matrix

A confusion matrix summarizes classification outcomes.

| | Predicted No Violation | Predicted Violation |
|---|---|---|
| Actual No Violation | True Negative | False Positive |
| Actual Violation | False Negative | True Positive |

It helps identify the types of prediction errors made by the classifier.

---

# 19. Model Validation Results

At the selected classification threshold of approximately 0.30, the reported validation results are:

| Metric | Validation Result |
|---|---:|
| Precision | 57.72% |
| Recall | 85.31% |
| Selected Threshold | Approximately 0.30 |

### Interpretation

**Precision: 57.72%**

Approximately 57.72% of the cases predicted as SLA violations were actual positive cases in the validation dataset.

**Recall: 85.31%**

Approximately 85.31% of the actual SLA violation cases in the validation dataset were identified by the model.

### Performance Consideration

The selected threshold represents a recall-oriented operating point.

It prioritizes identifying more positive cases while accepting that some predictions may be false positives.

These values represent validation performance, not final test-set performance.

---

# 20. Backend Implementation

The backend is developed using Python and FastAPI.

It acts as the communication layer between the React frontend and the machine learning model.

### Backend Responsibilities

- Receive prediction requests.
- Accept workload-related input features.
- Load the trained Random Forest model.
- Read the selected classification threshold.
- Prepare input data in the required feature order.
- Generate probability estimates.
- Apply the classification threshold.
- Return prediction results.

The backend uses the saved model for prediction rather than retraining the classifier for every request.

---

# 21. FastAPI Framework

FastAPI is a Python framework used to develop APIs.

In SLA Guard, it exposes the machine learning model through an HTTP endpoint.

### Why FastAPI?

- Request handling.
- Input validation.
- Structured API responses.
- Automatic interactive API documentation.
- Integration with Python machine learning libraries.

The API can be tested through the Swagger documentation generated by FastAPI.

---

# 22. Frontend Implementation

The frontend is developed using React.js and Vite.

React provides the user interface, while Vite provides the frontend development environment.

### Frontend Responsibilities

- Display the application interface.
- Collect workload input parameters.
- Submit data to the backend.
- Receive prediction responses.
- Display the predicted class.
- Display the violation probability.

The frontend provides a convenient way to interact with the machine learning model without manually executing Python scripts.

---

# 23. Frontend-Backend Communication

The frontend communicates with the backend through HTTP requests.

The general process is:

1. The user enters workload parameters.
2. React prepares the request data.
3. The frontend sends the request to FastAPI.
4. FastAPI processes the input.
5. The model generates a prediction.
6. The backend returns a response.
7. React displays the result.

Both frontend and backend services must be running locally for the complete application workflow to function.

CORS configuration is used where necessary to allow requests from the frontend development server.

---

# 24. Model Storage

The trained Random Forest model is saved using Joblib.

### Saved Files

**Trained Model**

`backend/sla_random_forest_model.pkl`

**Classification Threshold**

`backend/sla_threshold.json`

The model file stores the trained classifier, while the JSON file stores the selected classification threshold.

This allows the backend to load the trained model and threshold independently.

### Git Large File Storage

The trained model file is large.

Git LFS is used to track the `.pkl` file in the GitHub repository.

This allows large model files to be managed separately from ordinary Git objects.

---

# 25. Project Structure

The repository is organized into backend and frontend components.

```text
sla-guard/
│
├── backend/
│   ├── main.py
│   ├── train_model.py
│   ├── verify_model.py
│   ├── inspect_model.py
│   ├── check_model_history.py
│   ├── sla_random_forest_model.pkl
│   └── sla_threshold.json
│
├── public/
│
├── src/
│
├── .gitattributes
├── .gitignore
├── README.md
├── index.html
├── package.json
├── package-lock.json
├── eslint.config.js
└── vite.config.js
```

### Important Files

| File | Purpose |
|---|---|
| backend/main.py | FastAPI application |
| backend/train_model.py | Model training |
| backend/sla_random_forest_model.pkl | Trained model |
| backend/sla_threshold.json | Selected threshold |
| src/ | React source code |
| public/ | Public frontend assets |
| package.json | Frontend dependencies and scripts |
| vite.config.js | Vite configuration |
| .gitignore | Files excluded from Git |
| .gitattributes | Git LFS tracking configuration |

---

# 26. Installation and Setup

The project requires:

- Python.
- Node.js and npm.
- Git.
- Git LFS.
- A code editor such as Visual Studio Code.

### Clone the Repository

```bash
git clone https://github.com/Saloni1007/sla-guard.git
```

Move into the project directory:

```bash
cd sla-guard
```

If Git LFS is being used, install and retrieve the tracked model file:

```bash
git lfs install
git lfs pull
```

### Backend Setup

Move into the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
py -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install the backend dependencies according to the dependency file provided with the project.

### Frontend Setup

Open another terminal in the project root.

Install frontend dependencies:

```bash
npm install
```

---

# 27. Running the Application

The backend and frontend must run in separate terminals.

### Start the Backend

```powershell
cd backend
py -m uvicorn main:app --reload --port 8001
```

The backend will run at:

`http://127.0.0.1:8001`

### Start the Frontend

Open another terminal in the project root:

```bash
npm run dev
```

Vite will display the local development URL.

Open that address in a browser to access SLA Guard.

The backend must remain active while the frontend sends prediction requests.

---

# 28. API Testing

The backend can be tested independently through FastAPI's Swagger interface.

Open:

`http://127.0.0.1:8001/docs`

### Testing Procedure

1. Locate the `/predict` endpoint.
2. Expand the endpoint.
3. Select Try it out.
4. Enter the required workload parameters.
5. Execute the request.
6. Inspect the returned response.

The response fields should match the implementation in `backend/main.py`.

This testing process helps verify that the backend can receive inputs and return model predictions.

---

# 29. Example Prediction

Consider a workload submitted through the application.

The user enters the required workload parameters, and the frontend sends them to the backend.

The Random Forest model generates a probability estimate for the SLA violation class.

For example:

```text
Violation Probability: 72.83%
Classification Threshold: 30%
```

Since the estimated probability is greater than the threshold, the predicted class is:

```text
SLA Violation
```

This represents an example of a successful local API prediction.

It is not a model accuracy or recall measurement.

The probability is a model-generated estimate and does not guarantee that an actual SLA violation will occur.

---

# 30. Advantages

The main advantages of SLA Guard include:

- **Workload Analysis:** Considers multiple resource-related parameters.
- **Machine Learning Integration:** Uses a trained Random Forest classifier.
- **Probability-Based Prediction:** Provides an estimated violation probability.
- **Threshold Optimization:** Allows a recall-oriented classification threshold.
- **Full-Stack Development:** Combines React and FastAPI.
- **Reusable Model:** Loads the saved classifier without retraining for every prediction.
- **Local Execution:** Can be tested without deploying to a cloud platform.

---

# 31. Limitations

The current implementation has several limitations.

### Synthetic Dataset

The model is trained on synthetic data, which may not represent the complexity of actual cloud workloads.

### Real-World Generalization

Performance on real cloud infrastructure has not been established.

### Validation Performance

The reported precision and recall are validation results. Final test performance must be evaluated separately.

### False Positives

A lower threshold may increase false-positive predictions.

### False Negatives

The model may still fail to identify some actual violation cases.

### No Automatic Prevention

The application predicts potential SLA violations but does not automatically modify resource allocation, workload scheduling, or cloud infrastructure.

### Input Quality

Incorrect or unrealistic input values may produce unreliable predictions.

---

# 32. Future Scope

SLA Guard can be extended through additional development and experiments.

### Real-World Dataset Integration

Future work can investigate real cloud workload datasets to evaluate how well the model generalizes beyond synthetic data.

### Alternative Algorithms

Other classification algorithms could be compared, including:

- Logistic Regression.
- Decision Tree.
- Gradient Boosting.
- XGBoost.

### Advanced Imbalance Handling

SMOTE, SMOTE-ENN, oversampling, and undersampling could be evaluated as possible improvements.

### Threshold Comparison

Different thresholds can be compared to study their effect on precision, recall, and F1-score.

### Dashboard Improvements

The dashboard could include additional analytical information, such as prediction history and workload summaries.

### Cloud Integration

A future version could connect to a cloud monitoring platform to obtain workload information.

### Automated Resource Management

Prediction results could potentially support resource allocation and workload scheduling decisions after additional development and testing.

---

# 33. Conclusion

SLA Guard demonstrates a machine learning-based approach to predicting potential SLA violations in cloud computing workloads.

The project uses a Random Forest classifier trained on a synthetic dataset containing 50,000 records and 12 input features.

The classification threshold is optimized using validation data to prioritize recall.

The trained model is integrated with a FastAPI backend and a React-based frontend.

The application provides a complete workflow from workload input collection to probability estimation and prediction display.

SLA Guard serves as a prototype that demonstrates the integration of machine learning with a web application for cloud workload analysis.

Further improvements involving real-world datasets, alternative classification algorithms, and cloud integration could help extend its practical applicability.

---

# GitHub Repository

**Project Name:** SLA Guard

**GitHub Repository:**  
https://github.com/Saloni1007/sla-guard

The repository contains the project source code, backend implementation, model training code, trained model, and classification threshold file.

---

## Project Information

| Item | Details |
|---|---|
| Project Name | SLA Guard |
| Domain | Cloud Computing |
| Subdomain | Machine Learning |
| Problem Type | Binary Classification |
| Algorithm | Random Forest |
| Dataset | Synthetic Cloud Workload Dataset |
| Dataset Size | 50,000 Records |
| Input Features | 12 |
| Frontend | React.js |
| Backend | FastAPI |
| Model Storage | Joblib |
| Version Control | Git and Git LFS |
| Repository | GitHub |
