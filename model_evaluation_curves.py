import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import joblib
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import os  # Import the os module for folder creation

# Load trained model
model = joblib.load("random_forest.pkl")

# Load test data
data = np.load("test_data.npz", allow_pickle=True)
X_test, y_test_encoded = data["X_test"], data["y_test"]

# Get model predictions (probabilities)
y_probs = model.predict_proba(X_test)

# Binarize the labels for multi-class ROC and PR curves
label_binarizer = LabelBinarizer()
y_test_bin = label_binarizer.fit_transform(y_test_encoded)
n_classes = y_test_bin.shape[1]

# Create folders if they don't exist
if not os.path.exists("roc_curves"):
    os.makedirs("roc_curves")
if not os.path.exists("pr_curves"):
    os.makedirs("pr_curves")
if not os.path.exists("multi_class_roc"): #create new folders
    os.makedirs("multi_class_roc")
if not os.path.exists("multi_class_pr"): #create new folders
    os.makedirs("multi_class_pr")

# ROC Curve (Multi-class) - One plot per class
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Class {i}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"roc_curves/roc_curve_class_{i}.png")
    plt.show()

# Precision-Recall Curve (Multi-class) - One plot per class
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_probs[:, i])
    average_precision = average_precision_score(y_test_bin[:, i], y_probs[:, i])

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Class {i}')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(f"pr_curves/pr_curve_class_{i}.png")
    plt.show()

# Multi-class ROC Curve - Combined plot
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve (One-vs-Rest)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("multi_class_roc/multi_class_roc_curve.png") #save in new folder
plt.show()

# Multi-class Precision-Recall Curve - Combined plot
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_probs[:, i])
    average_precision = average_precision_score(y_test_bin[:, i], y_probs[:, i])
    plt.plot(recall, precision, lw=2, label=f'Class {i} (AP = {average_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multi-class Precision-Recall Curve (One-vs-Rest)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("multi_class_pr/multi_class_pr_curve.png") #save in new folder
plt.show()

# Save details to text file
with open("model_evaluation_details.txt", "w") as f:
    f.write("ROC Curve Details:\n")
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        f.write(f"Class {i} ROC curve (AUC = {roc_auc:.2f})\n")

    f.write("\nPrecision-Recall Curve Details:\n")
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_probs[:, i])
        average_precision = average_precision_score(y_test_bin[:, i], y_probs[:, i])
        f.write(f"Class {i} PR curve (AP = {average_precision:.2f})\n")