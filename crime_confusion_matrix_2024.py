import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = joblib.load("random_forest.pkl")

# Load test data
data = np.load("test_data.npz", allow_pickle=True)
X_test, y_test_encoded = data["X_test"], data["y_test"]

# Get model predictions
y_pred_encoded = model.predict(X_test)

# Encode string labels to numeric values
label_encoder = LabelEncoder()
label_encoder.fit(y_test_encoded) # only fit on the y_test_encoded.

# Handle unseen labels in predictions
y_pred_numeric = []
for label in y_pred_encoded:
    if label in label_encoder.classes_:
        y_pred_numeric.append(label_encoder.transform([label])[0])
    else:
        # Handle unseen label (e.g., assign a default value or skip)
        print(f"Warning: Unseen label '{label}' in predictions.")
        y_pred_numeric.append(-1)  # Assign -1 as a default value

y_pred_numeric = np.array(y_pred_numeric)

y_test_numeric = label_encoder.transform(y_test_encoded)

# Generate confusion matrix
cm = confusion_matrix(y_test_numeric, y_pred_numeric)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix - Crime Prediction 2024")
plt.tight_layout()
plt.savefig("crime_confusion_matrix_2024.png")
plt.show()

# Normalized confusion matrix (percentages)
cm_normalized = confusion_matrix(y_test_numeric, y_pred_numeric, normalize='true')

# Display normalized confusion matrix
disp_normalized = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=label_encoder.classes_)
disp_normalized.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Normalized Confusion Matrix - Crime Prediction 2024")
plt.tight_layout()
plt.savefig("crime_normalized_confusion_matrix_2024.png")
plt.show()

# Save the confusion matrix as a .txt file
with open("crime_confusion_matrix_2024.txt", "w") as f:
    f.write("Confusion Matrix - Crime Prediction 2024\n")
    f.write(np.array2string(cm, separator=', '))
    f.write("\n\nNormalized Confusion Matrix - Crime Prediction 2024\n")
    f.write(np.array2string(cm_normalized, separator=', '))