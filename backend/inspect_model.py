
from pathlib import Path
import joblib

model_path = Path(__file__).parent / "sla_random_forest_model.pkl"
model = joblib.load(model_path)

print("MODEL TYPE:")
print(type(model))

print("\nMODEL PARAMETERS:")
print(model.get_params())

print("\nNUMBER OF TREES:")
print(len(model.estimators_))

print("\nFEATURE NAMES:")
print(getattr(model, "feature_names_in_", "Not stored"))

print("\nNUMBER OF FEATURES:")
print(getattr(model, "n_features_in_", "Not stored"))

print("\nCLASSES:")
print(model.classes_)

print("\nTRAINING SAMPLE COUNTS:")
for i, tree in enumerate(model.estimators_[:5], start=1):
    print(f"Tree {i}: {tree.tree_.n_node_samples[0]}")
