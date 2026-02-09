import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------
# Step 1: Setup Paths
# --------------------------
BASE_DIR = os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "Orignal-Dataset")
MODEL_PATH = os.path.join(BASE_DIR, "crop_recommendation_model.h5")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------
# Step 2: Load Trained Model
# --------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# --------------------------
# Step 3: Prepare Validation Dataset
# --------------------------
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

val_ds = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),   # MUST match training size
    batch_size=32,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

# --------------------------
# Step 4: Evaluate Model
# --------------------------
loss, accuracy = model.evaluate(val_ds)
print(f"\n📊 Validation Accuracy: {accuracy * 100:.2f}%")
print(f"📉 Validation Loss: {loss:.4f}")

# --------------------------
# Step 5: Predictions
# --------------------------
y_pred = model.predict(val_ds)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_ds.classes
class_labels = list(val_ds.class_indices.keys())

# --------------------------
# Step 6: Confusion Matrix
# --------------------------
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Soil Classification")
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()

# --------------------------
# Step 7: Save Evaluation Report
# --------------------------
report = classification_report(
    y_true,
    y_pred_classes,
    target_names=class_labels
)

report_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")

with open(report_path, "w") as f:
    f.write("Soil Classification Model Evaluation Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Validation Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Validation Loss: {loss:.4f}\n\n")
    f.write("Detailed Classification Report:\n")
    f.write(report)
    f.write("\n\nConfusion Matrix saved at:\n" + cm_path)

print("\n✅ Evaluation completed successfully.")
print(f"📄 Report saved at: {report_path}")
print(f"🖼️ Confusion matrix saved at: {cm_path}")
