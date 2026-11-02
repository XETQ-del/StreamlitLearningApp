
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from joblib import dump
import os

# Load the Iris flower dataset
iris = load_iris()

# Train a new model
model = SVC()
model.fit(iris.data, iris.target)

# Make sure the data folder exists
os.makedirs("data", exist_ok=True)

# Save the trained model
dump(model, "data/iris_model.joblib")

print("Iris model created successfully!")
