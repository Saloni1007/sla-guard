
from pathlib import Path
import joblib
import sklearn

MODEL_PATH = Path(__file__).parent / "sla_random_forest_model.pkl"

print("=" * 50)
print("SLA GUARD - MODEL VERIFICATION")
print("=" * 50)

print("\n1. Checking model file...")

if not MODEL_PATH.exists():
    print("ERROR: Model file not found!")
    print("Expected location:", MODEL_PATH)
    raise SystemExit(1)

print("Model file exists:", MODEL_PATH.name)
print("File size:", round(MODEL_PATH.stat().st_size / 1024, 2), "KB")

print("\n2. Loading saved model...")

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as error:
    print("ERROR loading model:", error)
    raise SystemExit(1)

print("\n3. Model information")
print("Model type:", type(model).__name__)
print("Scikit-learn version:", sklearn.__version__)

if hasattr(model, "n_features_in_"):
    print("Number of input features:", model.n_features_in_)

if hasattr(model, "feature_names_in_"):
    print("Feature names:")
    for name in model.feature_names_in_:
        print(" -", name)
else:
    print("Feature names were not saved in the model.")

if hasattr(model, "classes_"):
    print("Classes:", model.classes_)

if hasattr(model, "n_estimators"):
    print("Number of trees:", model.n_estimators)

if hasattr(model, "estimators_"):
    print("Trees actually loaded:", len(model.estimators_))

    if len(model.estimators_) > 0:
        first_tree = model.estimators_[0].tree_
        print("First tree node count:", first_tree.node_count)
        print(
            "Training samples recorded at first tree root:",
            first_tree.n_node_samples[0]
        )

print("\n4. Testing prediction capability")

if not hasattr(model, "predict"):
    print("ERROR: The saved object has no predict() method.")
    raise SystemExit(1)

print("Model has a predict() method.")

print("\nIMPORTANT:")
print("This check verifies the saved model's structure.")
print("It does NOT prove the model was trained on real cloud data.")
print("We must inspect the training code and dataset to establish that.")

print("\nVerification finished.")
