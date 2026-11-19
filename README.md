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

However, the project uses a lower threshold to prioritize recall.

### Threshold selection strategy

The classification threshold was selected using the validation dataset.

The threshold search considered values from approximately 0.05 to 0.95 with increments of 0.01.

The target recall was set to 0.85.

The selection process follows these steps:

1. Generate probability predictions on validation data.
2. Evaluate different classification thresholds.
3. Calculate precision and recall for each threshold.
4. Identify thresholds that achieve the target recall.
5. Select the threshold with the highest precision among the qualifying thresholds.
6. If no threshold satisfies the recall target, use the best F2-score as a fallback.

### Selected threshold

The selected threshold is approximately:

`0.30`

The value is stored in the project's threshold configuration.

### Classification rule

Let:

- `P` represent the predicted probability of an SLA violation.
- `T` represent the selected classification threshold.

The decision rule is:

- If `P >= T`, predict an SLA violation.
- If `P < T`, predict no SLA violation.

With the selected threshold of approximately 0.30:

| Predicted probability | Classification |
|---|---|
| 0.20 | No SLA violation |
| 0.30 | SLA violation |
| 0.50 | SLA violation |
| 0.75 | SLA violation |

### Why a lower threshold is used

A lower threshold makes the model more likely to classify a workload as a potential violation.

This can increase recall because more actual violation cases may be identified.

However, lowering the threshold can also increase false-positive predictions.

Therefore, threshold selection involves a trade-off between identifying actual violations and avoiding unnecessary alarms.

---

## 8. Model evaluation and validation results

The model is evaluated using classification metrics.

The main metrics considered are accuracy, precision, recall, F1-score, and F2-score.

### Accuracy

Accuracy measures the proportion of correctly classified observations.

`Accuracy = Correct Predictions / Total Predictions`

### Precision

Precision measures the proportion of predicted violation cases that are actually violations.

`Precision = TP / (TP + FP)`

A higher precision indicates fewer false alarms among the predicted violation cases.

### Recall

Recall measures the proportion of actual violation cases that are correctly identified.

`Recall = TP / (TP + FN)`

A higher recall indicates that fewer actual violation cases are missed.

### F1-score

F1-score is the harmonic mean of precision and recall.

It provides a combined measure of these two metrics.

### F2-score

F2-score gives more importance to recall than precision.

It is useful when identifying actual SLA violation cases is a higher priority.

### Validation results

The validation threshold selection produced the following results at approximately 0.30.

| Metric | Validation result |
|---|---:|
| Precision | 57.72% |
| Recall | 85.31% |
| Target recall | 85% |
| Selected threshold | 0.30 |

### Interpretation

The validation recall of 85.31% indicates that the model identified approximately 85 out of every 100 actual SLA violation cases in the validation dataset.

The validation precision of 57.72% indicates that approximately 58 out of every 100 predictions classified as violations were actual violations.

The results demonstrate the trade-off between identifying more violation cases and generating false alarms.

**Important:** These values are validation results from threshold selection. They are not final test-set results.

The final testing metrics must be calculated separately on the reserved test dataset before being reported.

---

## 9. Backend implementation

### FastAPI backend

The backend is developed using FastAPI, a Python framework used to build APIs.

It acts as the communication layer between the React frontend and the trained machine learning model.

### Backend responsibilities

The backend performs the following operations:

1. Receives workload information from the frontend.
2. Validates incoming input parameters.
3. Loads the saved Random Forest model.
4. Loads the threshold configuration.
5. Processes the input features.
6. Generates the predicted class probability.
7. Applies the classification threshold.
8. Returns the prediction response.

### Model loading

The trained model is stored in:

`backend/sla_random_forest_model.pkl`

The backend loads this model using Joblib.

The model is loaded when the application starts, rather than being retrained for every prediction request.

The threshold configuration is stored separately in:

`backend/sla_threshold.json`

This file contains the selected classification threshold and related metadata.

### Prediction endpoint

The backend exposes a prediction endpoint:

`POST /predict`

The endpoint receives workload-related parameters in JSON format.

The received features are passed to the trained model, which generates a probability estimate and classification result.

### API documentation

FastAPI provides interactive API documentation through Swagger UI.

When the backend is running locally, the documentation can be accessed at:

`http://127.0.0.1:8001/docs`

This interface allows users to inspect and test the available API endpoints.

---

## 10. Frontend implementation

### React and Vite

The frontend is developed using React.js and Vite.

React is used to build the interactive user interface, while Vite provides the frontend development and build environment.

### Frontend responsibilities

The frontend provides the interface for interacting with the prediction system.

Its responsibilities include:

- Displaying the application dashboard.
- Collecting workload parameters.
- Sending prediction requests.
- Receiving responses from the backend.
- Displaying the predicted SLA status.
- Showing probability-related information.
- Presenting relevant model and prediction details.

### Frontend-backend communication

The frontend communicates with FastAPI using HTTP requests.

The prediction process follows these steps:

1. User enters workload parameters.
2. React collects the input values.
3. The application creates a JSON request.
4. The request is sent to the backend prediction endpoint.
5. FastAPI validates and processes the input.
6. The model generates the prediction.
7. The backend returns a JSON response.
8. React updates the interface with the returned result.

The frontend must be configured with the correct backend API address.

---

## 11. Prediction example

Consider a workload submitted through the SLA Guard prediction interface.

The Random Forest model generates a probability estimate for the positive class.

An example prediction response contains:

| Field | Example value |
|---|---|
| Prediction | 1 |
| Violation probability | 72.83% |
| Classification threshold | 30% |
| Model | Random Forest |

Since the predicted probability is greater than the selected threshold, the system classifies the workload as:

**SLA Violation Predicted**

### Interpretation

The model's estimated probability is 72.83%, while the selected decision threshold is 30%.

Because:

`72.83% >= 30%`

the final classification is 1.

This is an illustrative example from a successful local API test. It does not represent a general performance metric or a guaranteed real-world probability of failure.

---

## 12. Dataset provenance and interpretation

The current SLA Guard dataset is synthetic.

The dataset contains 50,000 records and 12 input features, with `SLA_Violation` as the target variable.

The synthetic data is used to develop and test the machine learning prediction workflow.

The following distinction is important:

| Component | Current implementation |
|---|---|
| Workload dataset | Synthetic |
| Input features | 12 workload-related parameters |
| Target variable | `SLA_Violation` |
| Training data | Synthetic workload records |
| Validation data | Reserved synthetic validation records |
| Test data | Reserved synthetic test records |
| Prediction model | Random Forest classifier |
| Prediction output | Binary classification and probability estimate |

The results demonstrate the behavior of the model on the dataset used in the project.

They do not establish that the model will achieve the same performance on real cloud infrastructure.

---

## 13. Model storage and serialization

The trained Random Forest model is saved as a serialized Python object.

The model file is:

`backend/sla_random_forest_model.pkl`

The `.pkl` file stores the trained model so that it can be loaded and reused without repeating the training process.

The backend uses Joblib to load the model.

### Why model serialization is used

- Avoids retraining for every prediction request.
- Allows the trained model to be reused.
- Separates model training from application execution.
- Makes integration between the training script and backend easier.

### Threshold configuration

The project also stores threshold information in:

`backend/sla_threshold.json`

The configuration file allows the backend to use the selected classification threshold during prediction.

### Git LFS

The trained model is approximately 150 MB.

Because it exceeds GitHub's standard 100 MB individual-file limit, Git Large File Storage (Git LFS) is used to store the model.

The repository therefore includes Git LFS tracking for the serialized model.

---

## 14. Project structure

The repository contains the frontend implementation, backend implementation, machine learning files, and supporting configuration.

```text
sla-guard/
│
├── backend/
│   ├── main.py
│   ├── train_model.py
│   ├── sla_random_forest_model.pkl
│   ├── sla_threshold.json
│   └── requirements.txt
│
├── src/
│   ├── App.jsx
│   ├── App.css
│   └── main.jsx
│
├── public/
│
├── package.json
├── package-lock.json
├── vite.config.js
├── index.html
├── README.md
├── .gitignore
└── .gitattributes
```

### Important files

| File | Purpose |
|---|---|
| `backend/main.py` | FastAPI application and prediction endpoint |
| `backend/train_model.py` | Model training implementation |
| `backend/sla_random_forest_model.pkl` | Saved Random Forest model |
| `backend/sla_threshold.json` | Classification threshold configuration |
| `backend/requirements.txt` | Python backend dependencies |
| `src/App.jsx` | Main React component |
| `src/App.css` | Frontend styling |
| `src/main.jsx` | React entry point |
| `package.json` | Frontend dependencies and scripts |
| `vite.config.js` | Vite configuration |
| `.gitignore` | Files excluded from Git |
| `.gitattributes` | Git LFS tracking configuration |
| `README.md` | Project documentation |

---

## 15. Installation and reproduction

### Prerequisites

The following software is required:

- Python.
- Node.js and npm.
- Git.
- Git LFS.

Git LFS is required to download the trained model from the repository.

### Clone the repository

```bash
git clone https://github.com/Saloni1007/sla-guard.git
```

Navigate to the project directory:

```bash
cd sla-guard
```

Ensure that Git LFS is installed and the model file has been downloaded successfully.

### Backend setup

Navigate to the backend directory:

```bash
cd backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
venv\Scripts\activate
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI backend:

```bash
uvicorn main:app --reload --port 8001
```

The backend will be available at:

`http://127.0.0.1:8001`

### API testing

Open the FastAPI documentation:

`http://127.0.0.1:8001/docs`

Use the `/predict` endpoint to submit workload parameters and inspect the returned prediction.

### Frontend setup

Open another terminal in the main project directory.

Install frontend dependencies:

```bash
npm install
```

Start the Vite development server:

```bash
npm run dev
```

Vite will display a local development URL.

Open the displayed URL in a browser to access the SLA Guard application.

The frontend must be configured to communicate with the running FastAPI backend.

---

## 16. Testing and verification

The project can be verified at the frontend, backend, and model levels.

### Backend verification

- Check that FastAPI starts successfully.
- Verify that the serialized model loads.
- Verify that the threshold configuration loads.
- Open the Swagger documentation.
- Submit valid input to `/predict`.
- Check that the API returns a successful response.

### Frontend verification

- Verify that the dashboard loads.
- Enter workload parameters.
- Submit a prediction request.
- Check that the frontend communicates with the backend.
- Verify that the returned prediction is displayed correctly.

### Model verification

- Check that all 12 features are supplied.
- Verify the expected feature ordering and input format.
- Check that the model generates probability estimates.
- Verify that the threshold is applied correctly.
- Confirm that the final prediction is consistent with the probability and threshold.

### Example API verification

A successful local test returned HTTP status 200 and a prediction response.

The tested example produced:

- Predicted class: 1.
- Violation probability: 72.83%.
- Classification threshold: 30%.

This verifies that the local prediction endpoint can load the model and return a prediction for the tested input.

---

## 17. Limitations

The current implementation has several limitations.

### 1. Synthetic dataset

The model is trained using synthetic cloud workload records.

Synthetic data may not fully represent the complex and changing behavior of real cloud environments.

Therefore, real-world generalization has not been established.

### 2. Limited algorithm comparison

Random Forest is the primary model used in the current implementation.

A systematic comparison with other classifiers has not yet been completed.

### 3. False-positive predictions

The classification threshold prioritizes recall.

This can help identify more actual violation cases but may also increase false alarms.

### 4. No automatic violation prevention

The system predicts potential SLA violations but does not automatically change resource allocation or workload scheduling.

### 5. No continuous cloud monitoring

The current application processes user-provided workload parameters rather than continuously receiving live cloud infrastructure metrics.

### 6. Limited explainability

The current prediction interface does not provide detailed explanations for every individual prediction.

### 7. Dependency and model compatibility

The saved model depends on compatible Python and Scikit-learn environments.

Version differences between the training and execution environments may produce compatibility warnings or errors.

---

## 18. Future work

Several improvements can be explored in future versions.

### Real-world dataset evaluation

The model can be evaluated using real cloud workload datasets to examine whether it generalizes beyond synthetic records.

### Machine learning algorithm comparison

Random Forest can be compared with:

- Logistic Regression.
- Decision Tree.
- XGBoost.
- Naive Bayes.

All models should be evaluated using the same data splits and metrics for a fair comparison.

### Advanced class imbalance handling

Additional experiments can investigate:

- SMOTE.
- Edited Nearest Neighbours.
- SMOTE-ENN.
- Alternative class-weighting strategies.

### Improved threshold selection

Threshold selection can be further studied to balance precision and recall according to the cost of missed violations and false alarms.

### Explainable AI

Techniques such as SHAP can be integrated to explain which workload features influence individual predictions.

### Real-time cloud monitoring

The system can be extended to collect live workload information from cloud monitoring services.

### Automated resource recommendations

Future versions could recommend resource adjustments or workload rescheduling when the model identifies a high risk of SLA violation.

### Model deployment and scalability

The backend could be deployed as a scalable service and tested under concurrent prediction requests.

---

## 19. Conclusion

SLA Guard demonstrates the application of machine learning to the problem of SLA violation prediction in cloud computing.

The system analyzes 12 workload-related input features using a Random Forest classification model trained on a synthetic dataset containing 50,000 records.

A probability-based classification threshold is selected using validation data to prioritize the identification of potential SLA violation cases.

The React frontend and FastAPI backend provide an integrated application through which users can submit workload parameters and receive prediction results.

The validation results demonstrate the model's behavior on the synthetic dataset, while also highlighting the trade-off between recall and precision.

The project provides a foundation for further research into predictive cloud workload analysis. Future work can focus on real-world data evaluation, algorithm comparison, improved explainability, continuous monitoring, and automated resource optimization.

---

## 20. GitHub repository

The complete source code and supporting project files are maintained in the following GitHub repository.

**Project name:** SLA Guard

**GitHub repository:**

https://github.com/Saloni1007/sla-guard

The repository includes:

- React frontend source code.
- FastAPI backend implementation.
- Random Forest model training script.
- Serialized trained machine learning model.
- Classification threshold configuration.
- Project configuration files.
- Documentation and setup instructions.

The trained model is stored using Git Large File Storage (Git LFS) because of its file size.

The repository is intended to provide access to the implementation and relevant files for project review and reproducibility.
