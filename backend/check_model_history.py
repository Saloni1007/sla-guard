
from pathlib import Path
import joblib

model_path = Path(__file__).parent / "sla_random_forest_model.pkl"
model = joblib.load(model_path)

print("MODEL FILE:", model_path)
print("MODEL CLASS:", type(model).__name__)
print("SCIKIT-LEARN PARAMETERS:")
for key, value in model.get_params().items():
    print(f"{key}: {value}")

print("\nMODEL FEATURE NAMES:")
print(getattr(model, "feature_names_in_", "Not recorded"))

print("\nMODEL CLASSES:")
print(model.classes_)

print("\nROOT SAMPLES PER TREE (first 10):")
for i, tree in enumerate(model.estimators_[:10], start=1):
    print(i, tree.tree_.n_node_samples[0])

print("\nTRAINING DATA ROWS:")
print(
    "Not recoverable from standard RandomForestClassifier metadata."
)
