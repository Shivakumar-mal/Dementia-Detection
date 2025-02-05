import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Load test data
df = pd.read_csv('../data/labeled_dataset.csv')
X = df.drop(['Cluster', 'Label'], axis=1)
y = df['Label']

# Load trained models
models = {
    "SVM": joblib.load("../models/svm.pkl"),
    "Gradient Boosting": joblib.load("../models/gradient_boosting.pkl"),
    "Logistic Regression": joblib.load("../models/logistic_regression.pkl")
}

# Evaluate models
for name, model in models.items():
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, pos_label='Dementia')
    f1 = f1_score(y, y_pred, pos_label='Dementia')

    print(f"{name}: Accuracy = {accuracy:.2f}, Precision = {precision:.2f}, F1-Score = {f1:.2f}")
