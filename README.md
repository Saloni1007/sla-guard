# SLA Guard – Cloud SLA Violation Prediction System

## Project Overview
SLA Guard is a machine learning-based web application designed to predict potential Service Level Agreement (SLA) violations in cloud computing workloads.

It analyzes workload and resource-related parameters and predicts whether a workload may violate its SLA.

## Technologies Used
- Frontend: React.js, Vite, CSS
- Backend: Python, FastAPI
- Machine Learning: Scikit-learn, Random Forest Classifier
- Model Storage: Joblib
- Version Control: Git and GitHub

## Machine Learning Model
The project uses a Random Forest Classifier trained on a synthetic cloud workload dataset containing 50,000 records.

It uses 12 input features:
1. CPU requested
2. Memory requested
3. CPU utilization
4. Memory utilization
5. Machine CPU utilization
6. Machine memory utilization
7. Queue length
8. Priority
9. Scheduling class
10. Network latency
11. Estimated runtime
12. Resource contention

**Target variable:** `SLA_Violation`

- 0 – No SLA violation predicted
- 1 – SLA violation predicted

The classification threshold is approximately 0.30.

## Project Structure
```text
sla-guard/
├── backend/
│   ├── main.py
│   ├── train_model.py
│   ├── sla_random_forest_model.pkl
│   ├── sla_threshold.json
│   └── requirements.txt
├── src/
├── public/
├── package.json
├── README.md
└── .gitignore
```

## How to Run the Project

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

API documentation: `http://127.0.0.1:8001/docs`

### Frontend Setup
Open another terminal in the project root:

```bash
npm install
npm run dev
```

Open the local URL displayed by Vite.

## Dataset
The project uses a synthetic cloud workload dataset containing 50,000 records, 12 input features, and the target variable `SLA_Violation`.

## Limitations and Future Improvements
- Evaluate using real-world cloud workload data.
- Compare Random Forest with other ML algorithms.
- Improve precision and reduce false alarms.
- Integrate real-time cloud monitoring.
- Explore explainable AI techniques.
- Develop automated resource allocation recommendations.

## GitHub Repository
[View SLA Guard on GitHub](https://github.com/Saloni1007/sla-guard)
