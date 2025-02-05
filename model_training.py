import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load labeled dataset
df = pd.read_csv('../data/labeled_dataset.csv')
X = df.drop(['Cluster', 'Label'], axis=1)
y = df['Label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    "SVM": SVC(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42)
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} trained successfully.")

# Save trained models (optional)
import joblib
for name, model in trained_models.items():
    joblib.dump(model, f"../models/{name.lower().replace(' ', '_')}.pkl")
