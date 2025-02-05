import matplotlib.pyplot as plt
import numpy as np

# Accuracy, precision, and F1 scores from evaluation.py
labels = ['SVM', 'Gradient Boosting', 'Logistic Regression']
accuracy_scores = [0.85, 0.88, 0.83]  # Replace with actual values
precision_scores = [0.78, 0.80, 0.75]  # Replace with actual values
f1_scores = [0.82, 0.85, 0.80]  # Replace with actual values

x = np.arange(len(labels))

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.bar(x, accuracy_scores, color='blue', alpha=0.7)
plt.xticks(x, labels)
plt.ylabel('Accuracy')
plt.title('Model Accuracy')

plt.subplot(1, 3, 2)
plt.bar(x, precision_scores, color='green', alpha=0.7)
plt.xticks(x, labels)
plt.ylabel('Precision')
plt.title('Model Precision')

plt.subplot(1, 3, 3)
plt.bar(x, f1_scores, color='red', alpha=0.7)
plt.xticks(x, labels)
plt.ylabel('F1-Score')
plt.title('Model F1-Score')

plt.tight_layout()
plt.show()
